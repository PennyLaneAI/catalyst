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

#include "LocalGpuCoprocessorSession.hpp"

#include <cstring>
#include <stdexcept>

#include "GpuLaunchers.hpp"
#include "HipCheck.hpp"
#include "WireProtocol.hpp"

#include <hip/hip_runtime.h>

namespace catalyst::transport::local_copy {

LocalGpuCoprocessorSession::LocalGpuCoprocessorSession(std::string /*unused*/, int gpu_device)
    : gpu_device_(gpu_device) {}

LocalGpuCoprocessorSession::~LocalGpuCoprocessorSession() {
    try {
        stop();
    } catch (...) {
    }
    if (request_slot_host_) {
        (void)hipHostFree(request_slot_host_);
    }
    if (link_) {
        // Wait for any in-flight kick, then unbind so no future call reaches a dying `this`.
        std::lock_guard<std::mutex> lock(link_->mu);
        link_->process_message = nullptr;
    }
}

void LocalGpuCoprocessorSession::ensure_gpu_state() {
    if (!gpu_) {
        gpu_ = std::make_unique<gpu_verbs::GpuRuntime>(gpu_device_);
    }
    if (!request_slot_host_) {
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&request_slot_host_),
                                sizeof(common::PayloadSlot),
                                hipHostMallocMapped | hipHostMallocCoherent),
                  "hipHostMalloc(local gpu request slot)");
        std::memset(request_slot_host_, 0, sizeof(common::PayloadSlot));
        void *slot_dev = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&slot_dev, request_slot_host_, 0),
                  "hipHostGetDevicePointer(local gpu request slot)");
        request_slot_dev_ = static_cast<common::PayloadSlot *>(slot_dev);
    }
    if (!handoff_.host) {
        handoff_ = gpu_->alloc_handoff(1);
    }
}

int LocalGpuCoprocessorSession::connect(const ConnectInfo &info) {
    link_ = acquire_memcpy_link(info);
    std::lock_guard<std::mutex> lock(link_->mu);
    if (link_->process_message) {
        throw std::runtime_error(
            "memcpy: another coprocessor is already bound to this endpoint (peer, oob_port)");
    }
    link_->process_message = [this](const void *in, std::size_t in_len, std::uint32_t decoder_id,
                                    void *out, std::size_t out_cap) {
        return this->process_message(in, in_len, decoder_id, out, out_cap);
    };
    return 0;
}

MemRegion LocalGpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    if (kind != MemKind::CpuRam) {
        throw std::runtime_error(
            "memcpy: local GPU coprocessor expects CpuRam request buffers from the controller");
    }
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

PeerRef LocalGpuCoprocessorSession::exchange_keys(const MemRegion & /*local*/) { return PeerRef{}; }

void LocalGpuCoprocessorSession::establish_channel(const ChannelDesc &desc,
                                                   const MemRegion & /*local*/,
                                                   const PeerRef & /*peer*/) {
    if (desc.transport != "memcpy") {
        throw std::runtime_error("memcpy: local GPU coprocessor supports only transport=memcpy");
    }
}

void LocalGpuCoprocessorSession::start() { ensure_gpu_state(); }

int LocalGpuCoprocessorSession::collect(void *const * /*replies*/,
                                        const std::uint64_t * /*replies_bytes*/,
                                        std::size_t /*n*/) {
    throw std::logic_error("memcpy: coprocessor collect is not used");
}

void LocalGpuCoprocessorSession::stop() {
    if (gpu_) {
        gpu_->sync();
    }
}

void LocalGpuCoprocessorSession::set_coprocessor_launcher(CoprocessorLauncherFn fn, void *ctx) {
    launcher_ = fn;
    launcher_ctx_ = ctx;
}

std::size_t LocalGpuCoprocessorSession::process_message(const void *in, std::size_t in_len,
                                                        std::uint32_t decoder_id, void *out,
                                                        std::size_t out_cap) {
    if (in_len != common::PAYLOAD_DATA_BYTES) {
        throw std::runtime_error("memcpy: local GPU coprocessor expects one 8-byte payload");
    }
    if (out_cap < sizeof(std::int64_t)) {
        throw std::runtime_error("memcpy: reply buffer too small for GPU correction");
    }

    ensure_gpu_state();

    request_slot_host_[0] = common::PayloadSlot{};
    handoff_.host[0] = gpu_verbs::HandoffSlot{};
    if (handoff_.stop_host) {
        *handoff_.stop_host = 0;
    }

    std::memcpy(&request_slot_host_[0].p.value, in, common::PAYLOAD_DATA_BYTES);
    request_slot_host_[0].p.decoder_id = decoder_id;
    request_slot_host_[0].p.seq_num = 1;

    CoprocLaunchDesc desc{
        .ring = request_slot_dev_,
        .ring_slots = 1,
        .handoff = handoff_.dev,
        .stop = handoff_.stop_dev,
        .total = 1,
        .stream = gpu_->stream(),
    };
    CoprocessorLauncherFn launch = launcher_ ? launcher_ : &gpu_verbs::default_echo_launcher;
    if (launch(&desc, launcher_ctx_) != 0) {
        throw std::runtime_error("memcpy: GPU coprocessor launcher failed");
    }
    gpu_->sync();

    if (handoff_.host[0].seq != 1) {
        throw std::runtime_error("memcpy: GPU coprocessor did not publish a reply");
    }

    const std::int64_t correction = handoff_.host[0].correction;
    std::memcpy(out, &correction, sizeof(correction));
    return sizeof(correction);
}

} // namespace catalyst::transport::local_copy
