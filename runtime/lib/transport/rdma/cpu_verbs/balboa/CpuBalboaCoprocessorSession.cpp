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

#include "CpuBalboaCoprocessorSession.hpp"

#include <cstdlib>
#include <cstring>
#include <thread>

#include "Error.hpp"
#include "RealtimeThread.hpp"
#include "WireProtocol.hpp"

#include <unistd.h>

namespace catalyst::transport::cpu_verbs {
using namespace catalyst::transport::common;

namespace {
// How often the idle spin looks at the stop token.
constexpr std::uint64_t STOP_CHECK_MASK = 0xFFFFF;
} // namespace

int CpuBalboaCoprocessorSession::connect(const ConnectInfo &info) {
    link_.connect(info.oob_port);
    return 0;
}

MemRegion CpuBalboaCoprocessorSession::alloc_memory(std::size_t /*size*/, MemKind kind) {
    TP_CHECK(kind == MemKind::CpuRam, "cpu_verbs (wire=balboa): only MemKind::CpuRam supported");
    // The ring is fixed by the wire, whatever the core asks for, and there is one per session.
    const std::size_t bytes = ring_.region_bytes();
    if (!ring_mr_) {
        // page-aligned, as a region at IOVA 0 needs (64 KiB pages on Grace)
        const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
        void *buf = std::aligned_alloc(page, (bytes + page - 1) / page * page);
        TP_CHECK_ERRNO(buf, "aligned_alloc(%zu)", bytes);
        std::memset(buf, 0, bytes); // no stale seq can look like a request
        ring_buf_ = static_cast<std::uint8_t *>(buf);
        ring_mr_.emplace(link_.pd(), buf, bytes, /*iova=*/0, MemAccess::REMOTE_WRITE,
                         std::shared_ptr<void>(buf, std::free));
    }
    return MemRegion{
        .addr = ring_buf_,
        .size = bytes,
        .lkey = ring_mr_->lkey(),
        .rkey = ring_mr_->rkey(),
        .kind = MemKind::CpuRam,
    };
}

PeerRef CpuBalboaCoprocessorSession::exchange_keys(const MemRegion & /*local*/) {
    TP_CHECK(ring_mr_.has_value(), "cpu_verbs (wire=balboa): exchange_keys before alloc_memory");
    link_.exchange(ring_mr_->rkey(), ring_);
    return PeerRef{};
}

void CpuBalboaCoprocessorSession::establish_channel(const ChannelDesc &desc,
                                                    const MemRegion & /*local*/,
                                                    const PeerRef & /*peer*/) {
    TP_CHECK(desc.transport == "rdma", "cpu_verbs: only transport \"rdma\" supported");
    link_.establish();
}

void CpuBalboaCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    coproc_fn_ = fn;
    coproc_ctx_ = ctx;
}

void CpuBalboaCoprocessorSession::start() {
    stop();
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    completed_.store(0, std::memory_order_relaxed);
    last_word_.store(0, std::memory_order_relaxed);
    // A data-path TP_CHECK throws; it must not escape the thread function, so it is captured
    // into error_ and published through failed_ for collect() to rethrow.
    engine_ = std::jthread([this](std::stop_token st) {
        try {
            run(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    });
}

// A request is seen in the ring itself: the seq of the slot BALBOA writes next turns to the
// expected value the moment the NIC's write lands. BALBOA writes round n into slot n mod the
// ring's slots and the controller's seq counts from 1, so request n has seq n + 1. RC delivers in
// order and retransmits a lost WRITE, so the session never has to find its place again.
void CpuBalboaCoprocessorSession::run(std::stop_token st) {
    pin_thread(pin_cpu_, pin_realtime_);
    std::uint32_t slot = 0;
    std::uint32_t expect = 1;
    Payload reply{};

    for (std::uint64_t spin = 0;; ++spin) {
        frame::FrameView req;
        if (frame::poll<frame::Mode::BL16>(ring_, ring_buf_ + ring_.slot_offset(slot), expect,
                                           req)) {
            std::uint64_t word = 0;
            std::memcpy(&word, req.data, sizeof(word));
            last_word_.store(word, std::memory_order_relaxed);
            completed_.fetch_add(1, std::memory_order_release);

            reply.value = 0;
            reply.decoder_id = req.fn_id;
            if (coproc_fn_) {
                const std::size_t nb = coproc_fn_(req.data, req.len, req.fn_id, &reply.value,
                                                  PAYLOAD_DATA_BYTES, coproc_ctx_);
                TP_CHECK(nb > 0 && nb <= PAYLOAD_DATA_BYTES,
                         "coprocessor function wrote %zu bytes, expected 1..%zu", nb,
                         PAYLOAD_DATA_BYTES);
            } else {
                reply.value = word; // built-in echo
            }
            reply.seq_num = expect; // the controller matches the reply by it
            link_.send(&reply);

            slot = (slot + 1) & (ring_.slots - 1);
            ++expect;
            continue;
        }

        // Idle. PAUSE leaves the spin faster when the line changes.
        cpu_relax();
        if (link_.sends_inflight() >= BalboaLink::SIGNAL_EVERY) {
            link_.reap_send();
        }
        if ((spin & STOP_CHECK_MASK) == 0 && st.stop_requested()) {
            return;
        }
    }
}

int CpuBalboaCoprocessorSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                         std::size_t n) {
    while (completed_.load(std::memory_order_acquire) == 0) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_);
        }
        if (!engine_.joinable() || engine_.get_stop_token().stop_requested()) {
            break;
        }
        std::this_thread::yield();
    }
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }
    if (completed_.load(std::memory_order_acquire) == 0) {
        return -1; // stopped before any round completed
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

void CpuBalboaCoprocessorSession::stop() {
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
}

} // namespace catalyst::transport::cpu_verbs
