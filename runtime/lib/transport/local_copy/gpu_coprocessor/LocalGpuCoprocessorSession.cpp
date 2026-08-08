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

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "GpuLaunchers.hpp"
#include "HipCheck.hpp"
#include "LocalWireProtocol.hpp"
#include "WireProtocol.hpp"

#include <hip/hip_runtime.h>

namespace catalyst::transport::local_copy {
namespace {

std::size_t local_region_bytes(std::size_t payload_bytes) {
    return payload_bytes + std::max(kLocalRequestHeaderBytes, kLocalReplyHeaderBytes);
}

} // namespace

LocalGpuCoprocessorSession::LocalGpuCoprocessorSession(std::string /*unused*/, int gpu_device)
    : gpu_device_(gpu_device) {}

LocalGpuCoprocessorSession::~LocalGpuCoprocessorSession() {
    try {
        stop();
    } catch (...) {
    }
    if (request_ring_host_) {
        (void)hipHostFree(request_ring_host_);
    }
}

void LocalGpuCoprocessorSession::ensure_gpu_state() {
    if (!gpu_) {
        gpu_ = std::make_unique<gpu_verbs::GpuRuntime>(gpu_device_);
    }
    if (!request_ring_host_) {
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&request_ring_host_),
                                common::REGION_BYTES, hipHostMallocMapped | hipHostMallocCoherent),
                  "hipHostMalloc(local gpu request ring)");
        std::memset(request_ring_host_, 0, common::REGION_BYTES);
        void *ring_dev = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&ring_dev, request_ring_host_, 0),
                  "hipHostGetDevicePointer(local gpu request ring)");
        request_ring_dev_ = static_cast<common::PayloadSlot *>(ring_dev);
    }
    if (!handoff_.host) {
        handoff_ = gpu_->alloc_handoff(common::K_RING_SLOTS);
    }
}

int LocalGpuCoprocessorSession::connect(const ConnectInfo &info) {
    pair_ = acquire_endpoint_pair(info);
    pair_->coprocessor = this;
    return 0;
}

MemRegion LocalGpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    if (kind != MemKind::CpuRam) {
        throw std::runtime_error(
            "local_copy: local GPU coprocessor expects CpuRam request buffers from the controller");
    }
    const std::size_t alloc_bytes = local_region_bytes(size);
    caller_memory_regions_.push_back(alloc_bytes ? std::make_unique<std::byte[]>(alloc_bytes)
                                                 : std::unique_ptr<std::byte[]>{});
    return MemRegion{
        .addr = caller_memory_regions_.empty() ? nullptr : caller_memory_regions_.back().get(),
        .size = static_cast<std::uint64_t>(alloc_bytes),
        .lkey = 0,
        .rkey = 0,
        .kind = kind,
    };
}

PeerRef LocalGpuCoprocessorSession::exchange_keys(const MemRegion &local) {
    local_request_ = local;
    if (pair_) {
        pair_->coprocessor_request = local;
        pair_->coprocessor_request_ready = true;
        if (pair_->controller_reply_ready) {
            peer_reply_ = PeerRef{
                .rkey = 0,
                .remote_addr = reinterpret_cast<std::uint64_t>(pair_->controller_reply.addr),
                .size = pair_->controller_reply.size,
            };
        }
    }
    return peer_reply_;
}

void LocalGpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                                   const PeerRef &peer) {
    if (desc.data_path != "memcpy") {
        throw std::runtime_error(
            "local_copy: local GPU coprocessor supports only data_path=memcpy");
    }
    local_request_ = local;
    peer_reply_ = peer;
    if (peer_reply_.remote_addr == 0 && pair_ && pair_->controller_reply_ready) {
        peer_reply_ = PeerRef{
            .rkey = 0,
            .remote_addr = reinterpret_cast<std::uint64_t>(pair_->controller_reply.addr),
            .size = pair_->controller_reply.size,
        };
    }
}

void LocalGpuCoprocessorSession::start() { ensure_gpu_state(); }

int LocalGpuCoprocessorSession::collect(void *const * /*replies*/,
                                        const std::uint64_t * /*replies_bytes*/,
                                        std::size_t /*n*/) {
    throw std::logic_error("LocalGpuCoprocessorSession::collect not implemented yet");
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

int LocalGpuCoprocessorSession::run_once() {
    if (!local_request_.addr || local_request_.size < kLocalRequestHeaderBytes) {
        throw std::runtime_error("local_copy: local request region is not established");
    }
    if (peer_reply_.remote_addr == 0 || peer_reply_.size < kLocalReplyHeaderBytes) {
        throw std::runtime_error("local_copy: peer reply region is not established");
    }
    if (local_request_.kind != MemKind::CpuRam) {
        throw std::runtime_error("local_copy: local GPU coprocessor expects CpuRam request region");
    }
    if (pair_ && pair_->controller_reply.kind != MemKind::CpuRam) {
        throw std::runtime_error("local_copy: local GPU coprocessor expects CpuRam reply region");
    }

    const auto *request_base = static_cast<const std::byte *>(local_request_.addr);
    const auto *request_hdr = reinterpret_cast<const LocalRequestHeader *>(request_base);
    const std::uint64_t request_bytes = request_hdr->bytes;
    if (request_bytes > local_request_.size - kLocalRequestHeaderBytes) {
        throw std::runtime_error("local_copy: request exceeds local request region capacity");
    }
    if (request_bytes != common::PAYLOAD_DATA_BYTES) {
        throw std::runtime_error("local_copy: local GPU coprocessor expects one 8-byte payload");
    }
    if (peer_reply_.size < kLocalReplyHeaderBytes + sizeof(std::int64_t)) {
        throw std::runtime_error("local_copy: peer reply region too small for GPU correction");
    }

    ensure_gpu_state();

    // Reset Payload and Handoff
    request_ring_host_[0] = common::PayloadSlot{};
    handoff_.host[0] = gpu_verbs::HandoffSlot{};
    if (handoff_.stop_host) {
        *handoff_.stop_host = 0;
    }

    std::memcpy(&request_ring_host_[0].p.value, request_base + kLocalRequestHeaderBytes,
                common::PAYLOAD_DATA_BYTES);
    request_ring_host_[0].p.decoder_id = request_hdr->decoder_id;
    request_ring_host_[0].p.seq_num = 1;

    CoprocLaunchDesc desc{
        .ring = request_ring_dev_,
        .ring_slots = static_cast<std::uint32_t>(common::K_RING_SLOTS),
        .handoff = handoff_.dev,
        .stop = handoff_.stop_dev,
        .total = 1,
        .stream = gpu_->stream(),
    };
    CoprocessorLauncherFn launch = launcher_ ? launcher_ : &gpu_verbs::default_echo_launcher;
    if (launch(&desc, launcher_ctx_) != 0) {
        throw std::runtime_error("local_copy: GPU coprocessor launcher failed");
    }
    gpu_->sync();

    if (handoff_.host[0].seq != 1) {
        throw std::runtime_error("local_copy: GPU coprocessor did not publish a reply");
    }

    auto *reply_base =
        reinterpret_cast<std::byte *>(static_cast<std::uintptr_t>(peer_reply_.remote_addr));
    const std::int64_t correction = handoff_.host[0].correction;
    std::memcpy(reply_base + kLocalReplyHeaderBytes, &correction, sizeof(correction));
    auto *reply_hdr = reinterpret_cast<LocalReplyHeader *>(reply_base);
    reply_hdr->bytes = sizeof(correction);
    return 0;
}

} // namespace catalyst::transport::local_copy
