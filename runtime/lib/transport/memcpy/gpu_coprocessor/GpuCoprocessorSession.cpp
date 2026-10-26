// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "GpuCoprocessorSession.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>

#include "Error.hpp"
#include "GpuLaunchers.hpp"
#include "HipCheck.hpp"
#include "WireProtocol.hpp"

#include <hip/hip_runtime.h>

namespace catalyst::transport::memcpy {

GpuCoprocessorSession::GpuCoprocessorSession(const std::string &config, int gpu_device,
                                             bool per_message)
    : pair_key_(parse_pair_key(config)), gpu_device_(gpu_device), per_message_(per_message),
      worker_(common::frame::geometry_from_config(config)),
      slots_(common::frame::geometry_from_config(config).slots), reply_ring_(slots_) {
    // The persistent kernels read 16 B frames, so a launch-once session carries 8 B messages.
    // TODO: lift this once the persistent kernels read frames through the frame codec.
    TP_CHECK(per_message_ ||
                 common::frame::geometry_from_config(config).mode == common::frame::Mode::BL16,
             "A launch-once GPU coprocessor reads 16 B frames, but its config asks for another "
             "frame layout; set coproc_fn=per_message in its config for larger messages");
}

GpuCoprocessorSession::~GpuCoprocessorSession() {
    try {
        stop();
    } catch (...) {
    }
    if (ring_host_) {
        (void)hipHostFree(ring_host_);
    }
    if (link_) {
        // Wait for any in-flight kick, then unbind so no future call reaches a dying `this`.
        std::lock_guard<std::mutex> lock(link_->mu);
        link_->process_message = nullptr;
    }
}

void GpuCoprocessorSession::ensure_gpu_state() {
    if (!gpu_) {
        gpu_ = std::make_unique<coproc::GpuRuntime>(gpu_device_);
    }
    if (!ring_host_) {
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&ring_host_),
                                slots_ * sizeof(common::PayloadSlot),
                                hipHostMallocMapped | hipHostMallocCoherent),
                  "hipHostMalloc(local gpu request ring)");
        std::memset(ring_host_, 0, slots_ * sizeof(common::PayloadSlot));
        void *ring_dev = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&ring_dev, ring_host_, 0),
                  "hipHostGetDevicePointer(local gpu request ring)");
        ring_dev_ = static_cast<common::PayloadSlot *>(ring_dev);
    }
    if (!handoff_.host) {
        handoff_ = gpu_->alloc_handoff(slots_);
    }
}

int GpuCoprocessorSession::connect(const ConnectInfo & /*info*/) {
    // Only bind `link_` after the duplicate-binding check succeeds. Otherwise the destructor of
    // a rejected second coprocessor would clear the incumbent's binding on the way out.
    auto candidate = acquire_memcpy_link(pair_key_);
    std::lock_guard<std::mutex> lock(candidate->mu);
    TP_CHECK(!candidate->process_message, "Coprocessor already bound to pair '%s'",
             pair_key_.c_str());
    candidate->process_message = [this](const void *payload, std::size_t len, std::uint32_t fn_id,
                                        void *out, std::size_t out_cap) {
        return this->process_message(payload, len, fn_id, out, out_cap);
    };
    link_ = std::move(candidate);
    return 0;
}

MemRegion GpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(kind == MemKind::CpuRam, "CPU device can only allocate CpuRam");
    caller_memory_regions_.push_back(size ? std::make_unique<std::byte[]>(size)
                                          : std::unique_ptr<std::byte[]>{});
    return MemRegion{
        .addr = size ? caller_memory_regions_.back().get() : nullptr,
        .size = static_cast<std::uint64_t>(size),
        .lkey = 0,
        .rkey = 0,
        .kind = kind,
    };
}

PeerRef GpuCoprocessorSession::exchange_keys(const MemRegion & /*local*/) { return PeerRef{}; }

void GpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion & /*local*/,
                                              const PeerRef & /*peer*/) {
    TP_CHECK(desc.transport == "memcpy", "Only transport=memcpy is supported");
}

void GpuCoprocessorSession::start() {
    stop();
    if (per_message_) {
        const int device = gpu_device_;
        worker_.start([device] { HIP_CHECK(hipSetDevice(device), "hipSetDevice(coprocessor)"); });
        return;
    }
    ensure_gpu_state();
    // Reset per-session state: cursors, ring slots, handoff slots, stop flag, error.
    process_cursor_ = 0;
    std::memset(ring_host_, 0, slots_ * sizeof(common::PayloadSlot));
    std::fill_n(handoff_.host, slots_, coproc::HandoffSlot{});
    std::fill(reply_ring_.begin(), reply_ring_.end(), common::PayloadSlot{});
    if (handoff_.stop_host) {
        *handoff_.stop_host = 0;
    }
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    // Launch the persistent decode kernel once; it polls ring_dev_ for seq_num and publishes
    // handoff slots until stop_dev flips (total=0 -> run until stop).
    CoprocLaunchDesc desc{
        .ring = ring_dev_,
        .ring_slots = slots_,
        .handoff = handoff_.dev,
        .stop = handoff_.stop_dev,
        .total = 0,
        .stream = gpu_->stream(),
    };
    CoprocessorLauncherFn launch = launcher_ ? launcher_ : &coproc::default_echo_launcher;
    TP_CHECK(launch(&desc, launcher_ctx_) == 0, "GPU persistent kernel launch failed");
    kernel_running_ = true;
    // Engine thread: handoff -> reply_ring. Captures exceptions so a throw doesn't terminate.
    engine_ = std::jthread([this](std::stop_token st) {
        try {
            run(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    });
}

int GpuCoprocessorSession::collect(void *const * /*replies*/,
                                   const std::uint64_t * /*replies_bytes*/, std::size_t /*n*/) {
    throw std::logic_error("Coprocessor collect unused");
}

void GpuCoprocessorSession::stop() {
    worker_.stop();
    if (kernel_running_ && handoff_.stop_host) {
        *handoff_.stop_host = 1; // let the persistent kernel exit its poll loop
    }
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
    if (gpu_) {
        gpu_->sync(); // wait for the kernel to observe stop and finish
    }
    kernel_running_ = false;
}

// Engine loop: for each cursor c, wait for the kernel to publish handoff[c % slots].seq == c+1,
// then republish into reply_ring[c % slots] with the same seq. HandoffSlot is 16-B aligned so the
// kernel's single-store makes correction+seq visible together.
void GpuCoprocessorSession::run(std::stop_token st) {
    constexpr std::uint32_t STOP_CHECK_SPINS = 4096;
    for (std::uint64_t c = 0; !st.stop_requested(); ++c) {
        const std::size_t idx = c & (slots_ - 1);
        const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
        volatile std::uint32_t *hseq = &handoff_.host[idx].seq;
        std::uint32_t spins = 0;
        while (*hseq != expect) {
            if (++spins == STOP_CHECK_SPINS) {
                spins = 0;
                if (st.stop_requested()) {
                    return;
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);

        const std::int64_t correction = handoff_.host[idx].correction;
        common::PayloadSlot &out = reply_ring_[idx];
        out.p.value = 0;
        std::memcpy(&out.p.value, &correction, sizeof(correction));
        // TODO: echo the request's fn_id once the kernel hands it back with the correction.
        out.p.decoder_id = 0;
        std::atomic_thread_fence(std::memory_order_release);
        out.p.seq_num = expect; // publish
    }
}

void GpuCoprocessorSession::set_coprocessor_launcher(CoprocessorLauncherFn fn, void *ctx) {
    TP_CHECK(!per_message_, "A per-message GPU coprocessor takes a CoprocessorFn, not a launcher");
    launcher_ = fn;
    launcher_ctx_ = ctx;
}

void GpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    TP_CHECK(per_message_,
             "A launch-once GPU coprocessor takes a launcher; set coproc_fn=per_message "
             "in its config to bind a CoprocessorFn");
    worker_.bind(fn, ctx);
}

std::size_t GpuCoprocessorSession::process_message(const void *payload, std::size_t len,
                                                   std::uint32_t fn_id, void *out,
                                                   std::size_t out_cap) {
    if (per_message_) {
        return worker_.process_message(payload, len, fn_id, out, out_cap);
    }
    TP_CHECK(len <= common::PAYLOAD_DATA_BYTES,
             "A launch-once GPU coprocessor carries 8 B messages, got %zu B; set "
             "coproc_fn=per_message in its config for larger ones",
             len);
    TP_CHECK(kernel_running_, "Call start() before process_message");
    TP_CHECK(out_cap <= sizeof(std::int64_t),
             "A launch-once GPU coprocessor carries 8 B messages, got a %zu B reply; set "
             "coproc_fn=per_message in its config for larger ones",
             out_cap);
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }

    // Frame the request in ring[cursor % slots]; sealing writes seq_num last, so the kernel never
    // observes a matching seq before the payload is visible.
    const std::uint64_t c = process_cursor_++;
    const std::size_t idx = c & (slots_ - 1);
    const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
    common::frame::stage(common::frame::BL16_RING, &ring_host_[idx].p, fn_id, payload, len);
    common::frame::seal(common::frame::BL16_RING, &ring_host_[idx].p, expect);

    // Spin-wait for the engine thread to publish reply_ring_[idx]. The engine bridges the
    // kernel's handoff into this ring so the controller never touches HIP memory directly.
    volatile std::uint32_t *sseq = &reply_ring_[idx].p.seq_num;
    while (*sseq != expect) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_);
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    std::int64_t correction = 0;
    std::memcpy(&correction, &reply_ring_[idx].p.value, sizeof(correction));
    const std::size_t n = std::min(out_cap, sizeof(correction));
    if (n != 0 && out) {
        std::memcpy(out, &correction, n);
    }
    return n;
}

} // namespace catalyst::transport::memcpy
