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
#include "RealtimeThread.hpp"

#include <unistd.h>

namespace catalyst::transport::gpu_verbs {
using namespace catalyst::transport;
using namespace catalyst::transport::common;
using namespace catalyst::transport::coproc; // GpuRuntime, launchers

namespace {
// The coprocessor only receives on the fwd queue pair (the controller writes into its ring), so
// that queue pair needs almost no send queue and no inline capacity.
constexpr QpShape FWD_QP_SHAPE{.sq_depth = 16, .cq_depth = 256, .inline_max = 0};
} // namespace

GpuCoprocessorSession::GpuCoprocessorSession(std::string dev, int gid_idx, int gpu_device,
                                             frame::RingGeometry ring, common::CuqrConfig cuqr)
    : Base(std::move(dev), gid_idx, "gpu_verbs", FWD_QP_SHAPE, ring, cuqr), gpu_(gpu_device) {
    // mlx5 hands a small SEND to the CPU to copy into its receive buffer, which in GPU memory it
    // cannot, so requests must be writes with immediate (a cudaq-realtime client picks its own).
    TP_CHECK(ring.mode != frame::Mode::CUQR || cuqr.write_imm || cuqr.cudaq_peer,
             "gpu_verbs: CUQR requests reach GPU memory only by write with immediate; set "
             "cuqr_tx=write_imm");
}

MemRegion GpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(kind == MemKind::GpuHbm, "gpu_verbs: only MemKind::GpuHbm supported");
    // The peer writes a ring of frames into the region, which may be larger than asked for; a small
    // ring still takes a whole page, as the dma-buf export needs.
    hbm_ = gpu_.alloc_hbm_ring(std::max({size, ring_.region_bytes(), std::size_t{4096}}));
    hbm_ring_.emplace(pd_, /*offset=*/static_cast<std::uint64_t>(0), hbm_.size,
                      // a cudaq-realtime peer addresses the ring from 0
                      /*iova=*/cuqr_.cudaq_peer ? 0 : reinterpret_cast<std::uint64_t>(hbm_.ptr),
                      hbm_.dmabuf_fd, MemAccess::REMOTE_WRITE);
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

void GpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                              const PeerRef &peer) {
    Base::establish_channel(desc, local, peer);
    const std::size_t flags = cuqr() ? 2 * ring_.slots * sizeof(std::uint32_t) : 0;
    handoff_ = gpu_.alloc_handoff(ring_.region_bytes() + flags);
}

void GpuCoprocessorSession::set_coprocessor_launcher(CoprocessorLauncherFn fn, void *ctx) {
    // Store the launcher; start() invokes it once to launch the on-device decode
    // kernel. nullptr selects the built-in echo launcher.
    coproc_launcher_ = fn;
    coproc_ctx_ = ctx;
}

bool GpuCoprocessorSession::post_reply(std::uint64_t cursor, const frame::FrameView &reply) {
    std::int64_t first = 0;
    std::memcpy(&first, reply.data, std::min<std::size_t>(reply.len, sizeof(first)));
    last_word_.store(first, std::memory_order_relaxed);
    void *send = send_frame();
    frame::stage(ring_, send, reply.fn_id, reply.data, reply.len, &reply); // CUQR echoes `reply`
    // The kernel writes only the reply's bytes; zero the rest of the payload area, as every other
    // coprocessor's reply is.
    std::memset(frame::data(ring_, send) + reply.len, 0, frame::data_capacity(ring_) - reply.len);
    const bool signaled = (cursor % SIGNAL_EVERY == 0);
    post_frame(bwd_qp_->get(), cursor, /*inline_data=*/true, signaled);
    return signaled;
}

// Coprocessor: wait for the fused kernel to publish each message's reply frame into the
// host-mapped handoff ring, then send it on the bwd QP, inline when it fits. Sends are
// selectively signaled (1 in SIGNAL_EVERY) and the bwd CQ is reaped in batches
// at those points, so completions stay off the per-message critical path.
void GpuCoprocessorSession::run_coprocessor(std::stop_token st) {
    pin_thread(pin_cpu_, pin_realtime_);
    cuqr() ? serve<true>(st) : serve<false>(st);
}

// CUQR frames carry no word written last, so this thread also relays each arrival from the receive
// completion queue to the kernel (rx_flags) and waits for its reply flag (tx_flags). This hop is
// not optimized; doc/dev/backline.rst lists ways to remove it.
template <bool CUQR> void GpuCoprocessorSession::serve(std::stop_token st) {
    int signaled_outstanding = 0;
    for (std::uint64_t c = 0; !st.stop_requested(); ++c) {
        const auto expect = static_cast<std::uint32_t>(c + 1);
        const std::uint64_t slot = c & (ring_.slots - 1);
        if constexpr (CUQR) {
            if (!wait_recv(c, st)) {
                break;
            }
            std::atomic_ref<std::uint32_t>(rx_flags_host()[slot])
                .store(expect, std::memory_order_release);
        }
        void *frame = handoff_.host + ring_.slot_offset(c);
        frame::FrameView reply;
        std::uint32_t spins = 0;
        while (CUQR ? std::atomic_ref<std::uint32_t>(tx_flags_host()[slot])
                              .load(std::memory_order_acquire) != expect
                    : !frame::poll(ring_, frame, expect, reply)) {
            if (++spins == STOP_CHECK_SPINS) {
                spins = 0;
                if (st.stop_requested()) {
                    reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/true);
                    return;
                }
            }
            cpu_relax();
        }
        if (CUQR && !frame::poll(ring_, frame, expect, reply)) {
            continue; // the kernel found no CUQR frame: dropped unanswered
        }
        if (post_reply(c, reply)) {
            ++signaled_outstanding;
            reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/false); // lazy batch reap
        }
        completed_.fetch_add(1, std::memory_order_release);
    }
    reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/true);
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
    if (cuqr()) { // a restarted session counts from message 0 again
        std::memset(rx_flags_host(), 0, 2 * ring_.slots * sizeof(std::uint32_t));
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
        .rx_flags = cuqr() ? handoff_.dev + ring_.region_bytes() : nullptr,
        .tx_flags = cuqr()
                        ? handoff_.dev + ring_.region_bytes() + ring_.slots * sizeof(std::uint32_t)
                        : nullptr,
    };
    CoprocessorLauncherFn launch = coproc_launcher_ ? coproc_launcher_ : &default_echo_launcher;
    const int lrc = launch(&L, coproc_ctx_);
    TP_CHECK(lrc == 0, "gpu_verbs: coprocessor kernel launch failed");
    // A data-path TP_CHECK throws TransportError; capture it here so it doesn't
    // escape the thread (which would std::terminate), and publish via failed_.
    auto body = [this](std::stop_token st) {
        try {
            run_coprocessor(st);
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
