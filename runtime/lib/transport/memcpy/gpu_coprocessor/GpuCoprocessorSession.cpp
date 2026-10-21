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
      ring_(common::frame::geometry_from_config(config)), worker_(ring_) {}

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
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&ring_host_), ring_.region_bytes(),
                                hipHostMallocMapped | hipHostMallocCoherent),
                  "hipHostMalloc(local gpu request ring)");
        void *ring_dev = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&ring_dev, ring_host_, 0),
                  "hipHostGetDevicePointer(local gpu request ring)");
        ring_dev_ = ring_dev;
    }
    if (!handoff_.host) {
        handoff_ = gpu_->alloc_handoff(ring_.region_bytes());
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
    // Reset per-session state: the cursor, both rings and the stop flag.
    process_cursor_ = 0;
    std::memset(ring_host_, 0, ring_.region_bytes());
    std::memset(handoff_.host, 0, ring_.region_bytes());
    *handoff_.stop_host = 0;
    // Launch the persistent decode kernel once; it polls ring_dev_ for requests and answers each
    // with a reply frame in the handoff ring until stop_dev flips (total=0 -> run until stop).
    CoprocLaunchDesc desc{
        .ring = ring_dev_,
        .ring_slots = ring_.slots,
        .handoff = handoff_.dev,
        .stop = handoff_.stop_dev,
        .total = 0,
        .stream = gpu_->stream(),
        .frame_mode = static_cast<std::uint32_t>(ring_.mode),
        .slot_stride = ring_.stride,
    };
    CoprocessorLauncherFn launch = launcher_ ? launcher_ : &coproc::default_echo_launcher;
    TP_CHECK(launch(&desc, launcher_ctx_) == 0, "GPU persistent kernel launch failed");
    kernel_running_ = true;
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
    if (gpu_) {
        gpu_->sync(); // wait for the kernel to observe stop and finish
    }
    kernel_running_ = false;
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
    TP_CHECK(kernel_running_, "Call start() before process_message");
    // Checked before the cursor advances: a slot that is never sealed would stall the kernel.
    const std::size_t cap = common::frame::data_capacity(ring_);
    TP_CHECK(len <= cap && out_cap <= cap,
             "A %zu B request or %zu B reply exceeds the %zu B a frame of this session carries",
             len, out_cap, cap);

    // Frame the request in its slot; sealing writes the polled word last, so the kernel never
    // sees a complete frame before its payload.
    const std::uint64_t c = process_cursor_++;
    const auto expect = static_cast<std::uint32_t>(c + 1);
    std::uint8_t *request = ring_host_ + ring_.slot_offset(c);
    common::frame::stage(ring_, request, fn_id, payload, len);
    common::frame::seal(ring_, request, expect);

    // Wait for the kernel's reply frame in the same slot of the handoff ring.
    common::frame::FrameView reply;
    while (!common::frame::poll(ring_, handoff_.host + ring_.slot_offset(c), expect, reply)) {
    }
    // The kernel writes only the reply's bytes; zero past them, as every other coprocessor does.
    if (out_cap != 0 && out) {
        const std::size_t n = reply.len < out_cap ? reply.len : out_cap;
        std::memcpy(out, reply.data, n);
        std::memset(static_cast<std::uint8_t *>(out) + n, 0, out_cap - n);
    }
    return out_cap;
}

} // namespace catalyst::transport::memcpy
