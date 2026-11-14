// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "CpuCoprocessorSession.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

#include "BalboaLink.hpp"
#include "Error.hpp"
#include "HsbLink.hpp"
#include "RcLink.hpp"
#include "RealtimeThread.hpp"
#include "WireProtocol.hpp"

#include <unistd.h>

namespace catalyst::transport::cpu_verbs {
using namespace catalyst::transport::common;

CpuCoprocessorSession::CpuCoprocessorSession(std::string dev, int gid_idx, frame::RingGeometry ring)
    : CpuCoprocessorSession(std::make_unique<RcLink>(std::move(dev), gid_idx, "cpu_verbs",
                                                     RcLink::Side::Coprocessor, ring)) {}

CpuCoprocessorSession::CpuCoprocessorSession(std::unique_ptr<PeerLink> link)
    : link_(std::move(link)), ring_(link_->ring()) {}

int CpuCoprocessorSession::connect(const ConnectInfo &info) {
    link_->connect(info);
    return 0;
}

MemRegion CpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(kind == MemKind::CpuRam, "cpu_verbs: only MemKind::CpuRam supported");
    // The peer writes a ring of frames into the region, which may be larger than asked for. It is
    // zeroed, so no stale seq looks like a request, and page-aligned (64 KiB pages on Grace).
    size = std::max(size, ring_.region_bytes());
    const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t rounded = (size + page - 1) / page * page;
    void *buf = std::aligned_alloc(page, rounded);
    TP_CHECK_ERRNO(buf, "aligned_alloc(%zu)", rounded);
    std::memset(buf, 0, rounded);
    regions_.emplace_back(link_->pd(), buf, size, link_->ring_iova(buf), MemAccess::REMOTE_WRITE,
                          std::shared_ptr<void>(buf, std::free));
    const MemoryRegion &mr = regions_.back();
    return MemRegion{
        .addr = buf,
        .size = size,
        .lkey = mr.lkey(),
        .rkey = mr.rkey(),
        .kind = MemKind::CpuRam,
    };
}

PeerRef CpuCoprocessorSession::exchange_keys(const MemRegion &local) {
    return link_->exchange(local);
}

void CpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                              const PeerRef &peer) {
    TP_CHECK(desc.transport == "rdma", "cpu_verbs: only transport \"rdma\" supported");
    ring_buf_ = static_cast<std::uint8_t *>(local.addr);
    link_->establish(local, peer);
}

void CpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    coproc_fn_ = fn;
    coproc_ctx_ = ctx;
}

void CpuCoprocessorSession::start() {
    stop();
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    completed_.store(0, std::memory_order_relaxed);
    last_word_.store(0, std::memory_order_relaxed);
    // jthread injects the stop_token. A data-path TP_CHECK throws TransportError;
    // it must not escape the thread function (that would std::terminate).
    // Capture it into error_ and publish via failed_ (release) so collect()
    // can rethrow the real exception.
    auto body = [this](std::stop_token st) {
        try {
            run(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    };
    engine_ = std::jthread(body);
}

int CpuCoprocessorSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                   std::size_t n) {
    while (completed_.load(std::memory_order_acquire) == 0) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_); // surface the engine's real error
        }
        if (!engine_.joinable() || engine_.get_stop_token().stop_requested()) {
            break;
        }
        std::this_thread::yield();
    }
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }
    // Stopped before any round completed -> no data (non-exceptional).
    if (completed_.load(std::memory_order_acquire) == 0) {
        return -1;
    }
    if (n > 0 && replies && replies[0]) {
        const std::uint64_t w = last_word_.load(std::memory_order_relaxed);
        const std::size_t cap = replies_bytes ? replies_bytes[0] : sizeof(w);
        TP_CHECK(cap <= sizeof(w), "reply capacity (%zu) exceeds the %zu B payload", cap,
                 sizeof(w));
        std::memcpy(replies[0], &w, cap);
    }
    return 0;
}

void CpuCoprocessorSession::stop() {
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
}

// The loop is compiled once per link and frame mode. The links are final, so the loop calls the
// link directly, with no virtual call and no peer or mode check on the hot path.
void CpuCoprocessorSession::run(std::stop_token st) {
    pin_thread(pin_cpu_, pin_realtime_);
    switch (link_->peer()) {
    case Peer::CATALYST:
        if (ring_.mode == frame::Mode::BL16) {
            serve<frame::Mode::BL16>(static_cast<RcLink &>(*link_), st);
        } else {
            serve<frame::Mode::BLV>(static_cast<RcLink &>(*link_), st);
        }
        break;
    // HSB and BALBOA carry only BL16 frames.
    case Peer::HSB:
        serve<frame::Mode::BL16>(static_cast<HsbLink &>(*link_), st);
        break;
    case Peer::BALBOA:
        serve<frame::Mode::BL16>(static_cast<BalboaLink &>(*link_), st);
        break;
    }
}

// Coprocessor: wait for a message, run the coprocessor function into the send frame in place, then
// send the result. A null fn is the built-in echo (passthrough).
// A request is seen in the ring itself: the seq of the slot the peer writes next reaches the
// expected value the moment the NIC's write lands, seq last. The peer fills the slots in order and
// the controller's seq counts from 1, so request n is in slot n mod the ring's slots with seq n
// + 1. A later seq is taken too, so in a one-slot ring a lost request does not stall the loop.
template <frame::Mode M, class Link>
void CpuCoprocessorSession::serve(Link &link, std::stop_token st) {
    std::uint64_t slot = 0;   // the slot polled next
    std::uint32_t expect = 1; // the lowest seq to take
    for (std::uint64_t spin = 0;; ++spin) {
        std::uint8_t *req = ring_buf_ + ring_.slot_offset(slot);
        const std::uint32_t seq = std::atomic_ref<std::uint32_t>(
                                      *reinterpret_cast<std::uint32_t *>(req + ring_.poll_offset()))
                                      .load(std::memory_order_acquire);
        frame::FrameView r; // the incoming message
        if (frame::seq_reached(seq, expect) && frame::poll<M>(ring_, req, seq, r)) {
            const std::size_t len = M == frame::Mode::BL16 ? PAYLOAD_DATA_BYTES : r.len;
            std::uint64_t word = 0;
            std::memcpy(&word, r.data, std::min<std::size_t>(len, sizeof(word)));
            last_word_.store(word, std::memory_order_relaxed);
            completed_.fetch_add(1, std::memory_order_release);
            // The reply goes back under the request's fn_id. Its payload area is zero-filled for
            // the function to fill, so the controller reads zeros past what it wrote; the built-in
            // echo copies the request's payload instead.
            void *send = link.send_frame();
            std::uint8_t *reply = frame::data<M>(ring_, send);
            const std::size_t cap = frame::data_capacity<M>(ring_);
            std::memset(reply, 0, cap);
            std::size_t nb = len;
            if (coproc_fn_) {
                nb = coproc_fn_(r.data, len, r.fn_id, reply, cap, coproc_ctx_);
                TP_CHECK(nb > 0 && nb <= cap,
                         "coprocessor function wrote %zu bytes, expected 1..%zu", nb, cap);
            } else {
                std::memcpy(reply, r.data, nb);
            }
            frame::stage<M>(ring_, send, r.fn_id, reply, nb);
            link.send(seq); // the controller matches the reply by it
            ++slot;
            expect = seq + 1;
            continue;
        }

        // Idle. PAUSE leaves the spin faster when the line changes.
        cpu_relax();
        link.idle();
        if ((spin & STOP_CHECK_MASK) == 0 && st.stop_requested()) {
            break;
        }
    }
    link.drain();
}

} // namespace catalyst::transport::cpu_verbs
