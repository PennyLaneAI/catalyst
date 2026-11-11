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

#include "GpuCoprocessorSession.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

#include "Error.hpp"
#include "GpuLaunchers.hpp"
#include "RcLink.hpp"
#include "RealtimeThread.hpp"

#include <unistd.h>

namespace catalyst::transport::gpu_verbs {
using namespace catalyst::transport;
using namespace catalyst::transport::common;
using namespace catalyst::transport::coproc; // GpuRuntime, launchers

GpuCoprocessorSession::GpuCoprocessorSession(std::string dev, int gid_idx, int gpu_device,
                                             frame::RingGeometry ring)
    : GpuCoprocessorSession(
          gpu_device, std::make_unique<RcLink>(std::move(dev), gid_idx, "gpu_verbs",
                                               RcLink::Side::Coprocessor, ring, FWD_QP_SHAPE)) {}

GpuCoprocessorSession::GpuCoprocessorSession(int gpu_device, std::unique_ptr<PeerLink> link)
    : link_(std::move(link)), ring_(link_->ring()), gpu_(gpu_device) {}

int GpuCoprocessorSession::connect(const ConnectInfo &info) {
    link_->connect(info);
    return 0;
}

MemRegion GpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(kind == MemKind::GpuHbm, "gpu_verbs: only MemKind::GpuHbm supported");
    // The peer writes a ring of frames into the region, which may be larger than asked for; a small
    // ring still takes a whole page, as the dma-buf export needs. It comes zeroed, so no stale seq
    // looks like a request.
    hbm_ = gpu_.alloc_hbm_ring(std::max({size, ring_.region_bytes(), std::size_t{4096}}));
    hbm_ring_.emplace(link_->pd(), /*offset=*/std::uint64_t{0}, hbm_.size,
                      link_->ring_iova(hbm_.ptr), hbm_.dmabuf_fd, MemAccess::REMOTE_WRITE);
    ::close(hbm_.dmabuf_fd); // MR holds its own reference
    hbm_.dmabuf_fd = -1;
    return MemRegion{
        .addr = hbm_.ptr,
        .size = hbm_.size,
        .lkey = hbm_ring_->lkey(),
        .rkey = hbm_ring_->rkey(),
        .kind = MemKind::GpuHbm,
    };
}

PeerRef GpuCoprocessorSession::exchange_keys(const MemRegion &local) {
    return link_->exchange(local);
}

void GpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                              const PeerRef &peer) {
    TP_CHECK(desc.transport == "rdma", "gpu_verbs: only transport \"rdma\" supported");
    handoff_ = gpu_.alloc_handoff(ring_.region_bytes());
    link_->establish(local, peer);
}

void GpuCoprocessorSession::set_coprocessor_launcher(CoprocessorLauncherFn fn, void *ctx) {
    // Store the launcher; start() invokes it once to launch the on-device decode
    // kernel. nullptr selects the built-in echo launcher.
    coproc_launcher_ = fn;
    coproc_ctx_ = ctx;
}

// The loop is compiled once per link and frame mode. The links are final, so the loop calls the
// link directly, with no virtual call and no peer or mode check on the hot path.
void GpuCoprocessorSession::run(std::stop_token st) {
    pin_thread(pin_cpu_, pin_realtime_);
    switch (link_->peer()) {
    case Peer::CATALYST:
        if (ring_.mode == frame::Mode::BL16) {
            serve<frame::Mode::BL16>(static_cast<RcLink &>(*link_), st);
        } else {
            serve<frame::Mode::BLV>(static_cast<RcLink &>(*link_), st);
        }
        break;
    }
}

// Coprocessor: wait for the fused kernel to publish its n-th reply frame in handoff slot n, then
// send it, inline when it fits, under that request's seq.
template <frame::Mode M, class Link>
void GpuCoprocessorSession::serve(Link &link, std::stop_token st) {
    std::uint64_t served = 0;
    for (std::uint64_t spin = 0;; ++spin) {
        const std::uint32_t seq = static_cast<std::uint32_t>(served + 1);
        frame::FrameView reply; // the kernel's reply, published with one store
        if (frame::poll<M>(ring_, handoff_.host + ring_.slot_offset(served), seq, reply)) {
            std::int64_t first = 0;
            std::memcpy(&first, reply.data, std::min<std::size_t>(reply.len, sizeof(first)));
            last_word_.store(first, std::memory_order_relaxed);
            void *send = link.send_frame();
            frame::stage<M>(ring_, send, reply.fn_id, reply.data, reply.len);
            // The kernel writes only the reply's bytes; zero the rest of the payload area, as
            // every other coprocessor's reply is.
            std::memset(frame::data<M>(ring_, send) + reply.len, 0,
                        frame::data_capacity<M>(ring_) - reply.len);
            link.send(seq); // the controller matches the reply by it
            completed_.fetch_add(1, std::memory_order_release);
            ++served;
            continue;
        }

        cpu_relax();
        link.idle();
        if ((spin & STOP_CHECK_MASK) == 0 && st.stop_requested()) {
            break;
        }
    }
    link.drain();
}

void GpuCoprocessorSession::start() {
    stop();
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    completed_.store(0, std::memory_order_relaxed);
    last_word_.store(0, std::memory_order_relaxed);
    if (handoff_.stop_host) {
        *handoff_.stop_host = 0;
    }
    // Launch the persistent kernel once via the bound launcher (nullptr selects
    // the built-in echo launcher). The kernel runs until the stop flag is set
    // (total = 0); the session keeps owning the stream/stop/sync + reply thread.
    // CoprocLaunchDesc is the HIP-free bridge to the launcher.
    CoprocLaunchDesc L{
        .ring = hbm_.ptr,
        .ring_slots = ring_.slots,
        .handoff = handoff_.dev,
        .stop = handoff_.stop_dev,
        .total = 0,
        .stream = gpu_.stream(),
        .frame_mode = static_cast<std::uint32_t>(ring_.mode),
        .slot_stride = ring_.stride,
    };
    CoprocessorLauncherFn launch = coproc_launcher_ ? coproc_launcher_ : &default_echo_launcher;
    const int lrc = launch(&L, coproc_ctx_);
    TP_CHECK(lrc == 0, "gpu_verbs: coprocessor kernel launch failed");
    // A data-path TP_CHECK throws TransportError; capture it here so it doesn't
    // escape the thread (which would std::terminate), and publish via failed_.
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

int GpuCoprocessorSession::collect(void *const *outputs, const std::uint64_t *output_bytes,
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
        return -1;
    }
    if (n > 0 && outputs && outputs[0]) {
        const std::int64_t w = last_word_.load(std::memory_order_relaxed);
        const std::size_t cap = output_bytes ? output_bytes[0] : sizeof(w);
        TP_CHECK(cap <= sizeof(w), "output capacity (%zu) exceeds the %zu B payload", cap,
                 sizeof(w));
        std::memcpy(outputs[0], &w, cap);
    }
    return 0;
}

void GpuCoprocessorSession::stop() {
    if (handoff_.stop_host) {
        *handoff_.stop_host = 1; // let the fused kernel exit its loop
    }
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
    gpu_.sync(); // wait for the kernel to observe stop and finish
}

} // namespace catalyst::transport::gpu_verbs
