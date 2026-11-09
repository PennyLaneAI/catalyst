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

#include "GpuHsbCoprocessorSession.hpp"

#include <algorithm>
#include <cstring>
#include <thread>

#include "Error.hpp"
#include "GpuLaunchers.hpp"
#include "RealtimeThread.hpp"
#include "WireProtocol.hpp"

#include <unistd.h>

namespace catalyst::transport::gpu_verbs {
using namespace catalyst::transport::common;

namespace {
// How often the idle spin looks at the stop token.
constexpr std::uint64_t STOP_CHECK_MASK = 0xFFFFF;
} // namespace

int GpuHsbCoprocessorSession::connect(const ConnectInfo &info) {
    link_.connect(info.oob_port);
    return 0;
}

MemRegion GpuHsbCoprocessorSession::alloc_memory(std::size_t /*size*/, MemKind kind) {
    TP_CHECK(kind == MemKind::GpuHbm, "gpu_verbs (wire=hsb): only MemKind::GpuHbm supported");
    // The ring is fixed by the wire, whatever the core asks for, and there is one per session.
    if (!ring_mr_) {
        // zeroed: no stale seq looks like a request; at least a page, as the dma-buf export needs
        hbm_ = gpu_.alloc_hbm_ring(std::max(ring_.region_bytes(), std::size_t{4096}));
        ring_mr_.emplace(link_.pd(), /*offset=*/std::uint64_t{0}, hbm_.size, /*iova=*/0,
                         hbm_.dmabuf_fd, MemAccess::REMOTE_WRITE);
        ::close(hbm_.dmabuf_fd); // the MR holds its own reference
        hbm_.dmabuf_fd = -1;
    }
    return MemRegion{
        .addr = hbm_.ptr,
        .size = hbm_.size,
        .lkey = ring_mr_->lkey(),
        .rkey = ring_mr_->rkey(),
        .kind = MemKind::GpuHbm,
    };
}

PeerRef GpuHsbCoprocessorSession::exchange_keys(const MemRegion & /*local*/) {
    TP_CHECK(ring_mr_.has_value(), "gpu_verbs (wire=hsb): exchange_keys before alloc_memory");
    link_.exchange(ring_mr_->rkey(), ring_);
    return PeerRef{};
}

void GpuHsbCoprocessorSession::establish_channel(const ChannelDesc &desc,
                                                 const MemRegion & /*local*/,
                                                 const PeerRef & /*peer*/) {
    TP_CHECK(desc.transport == "rdma", "gpu_verbs: only transport \"rdma\" supported");
    handoff_ = gpu_.alloc_handoff(ring_.region_bytes());
    link_.establish();
}

void GpuHsbCoprocessorSession::set_coprocessor_launcher(CoprocessorLauncherFn fn, void *ctx) {
    coproc_launcher_ = fn;
    coproc_ctx_ = ctx;
}

void GpuHsbCoprocessorSession::start() {
    stop();
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    completed_.store(0, std::memory_order_relaxed);
    last_word_.store(0, std::memory_order_relaxed);
    if (handoff_.stop_host) {
        *handoff_.stop_host = 0;
    }
    // The persistent kernel, once, until the stop flag is set (total = 0).
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
    TP_CHECK(launch(&L, coproc_ctx_) == 0, "gpu_verbs (wire=hsb): kernel launch failed");
    engine_ = std::jthread([this](std::stop_token st) {
        try {
            run(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    });
}

// GpuCoprocessorSession::run_coprocessor on the HSB wire: wait for the kernel to publish round n's
// reply frame in handoff slot n, send it with the request's seq (n + 1), and keep the receive and
// send queues serviced while nothing is due. Like the RC session, it follows the kernel's cursor,
// so a request lost on the wire stalls the session; the CPU session resyncs instead.
void GpuHsbCoprocessorSession::run(std::stop_token st) {
    pin_thread(pin_cpu_, pin_realtime_);
    std::uint64_t served = 0;
    std::uint64_t recvd = 0;
    Payload reply{};
    std::uint32_t cq_slots[HsbLink::RECV_BATCH];

    for (std::uint64_t spin = 0;; ++spin) {
        const std::uint32_t expect = static_cast<std::uint32_t>(served + 1);
        frame::FrameView r; // the kernel's reply, published with one 16 B store
        if (frame::poll<frame::Mode::BL16>(ring_, handoff_.host + ring_.slot_offset(served), expect,
                                           r)) {
            std::memcpy(&reply.value, r.data, sizeof(reply.value));
            last_word_.store(static_cast<std::int64_t>(reply.value), std::memory_order_relaxed);
            reply.decoder_id = r.fn_id;
            reply.seq_num = expect;
            link_.send(&reply);
            completed_.fetch_add(1, std::memory_order_release);
            ++served;
            continue;
        }

        cpu_relax();
        // only need to check if WRITE_WITH_IMM is enabled and there are requests in flight
        if (link_.imm() && recvd < served) {
            recvd += static_cast<std::uint64_t>(link_.reap_recv(cq_slots));
        }
        if (link_.sends_inflight() >= HsbLink::SIGNAL_EVERY) {
            link_.reap_send();
        }
        if ((spin & STOP_CHECK_MASK) == 0 && st.stop_requested()) {
            return;
        }
    }
}

int GpuHsbCoprocessorSession::collect(void *const *outputs, const std::uint64_t *output_bytes,
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

void GpuHsbCoprocessorSession::stop() {
    if (handoff_.stop_host) {
        *handoff_.stop_host = 1; // let the kernel exit its loop
    }
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
    gpu_.sync(); // wait for the kernel to see the stop flag and finish
}

} // namespace catalyst::transport::gpu_verbs
