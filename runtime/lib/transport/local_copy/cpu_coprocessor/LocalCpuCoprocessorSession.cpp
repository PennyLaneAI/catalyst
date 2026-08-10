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

#include "LocalCpuCoprocessorSession.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "LocalWireProtocol.hpp"

namespace catalyst::transport::local_copy {
namespace {

std::size_t echo_fn(const void *in, std::size_t in_len, void *out, std::size_t out_cap, void *) {
    const std::size_t n = std::min(in_len, out_cap);
    if (n != 0 && in && out) {
        std::memcpy(out, in, n);
    }
    return n;
}

std::size_t local_region_bytes(std::size_t payload_bytes) {
    return payload_bytes + std::max(kLocalRequestHeaderBytes, kLocalReplyHeaderBytes);
}

} // namespace

int LocalCpuCoprocessorSession::connect(const ConnectInfo &info) {
    pair_ = acquire_endpoint_pair(info);
    pair_->coprocessor = this;
    return 0;
}

MemRegion LocalCpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    if (kind != MemKind::CpuRam) {
        throw std::runtime_error("local_copy: CPU-only for now; alloc_memory expects CpuRam");
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

PeerRef LocalCpuCoprocessorSession::exchange_keys(const MemRegion &local) {
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

void LocalCpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                                   const PeerRef &peer) {
    if (desc.transport != "local") {
        throw std::runtime_error("local_copy: CPU-only coprocessor supports only transport=local");
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

void LocalCpuCoprocessorSession::start() {}

int LocalCpuCoprocessorSession::collect(void *const * /*replies*/,
                                        const std::uint64_t * /*replies_bytes*/,
                                        std::size_t /*n*/) {
    throw std::logic_error("LocalCpuCoprocessorSession::collect not implemented yet");
}

void LocalCpuCoprocessorSession::stop() {}

void LocalCpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    fn_ = fn;
    ctx_ = ctx;
}

int LocalCpuCoprocessorSession::run_once() {
    if (!local_request_.addr || local_request_.size < kLocalRequestHeaderBytes) {
        throw std::runtime_error("local_copy: local request region is not established");
    }
    if (peer_reply_.remote_addr == 0 || peer_reply_.size < kLocalReplyHeaderBytes) {
        throw std::runtime_error("local_copy: peer reply region is not established");
    }

    if (local_request_.kind != MemKind::CpuRam) {
        throw std::runtime_error("local_copy: CPU-only coprocessor expects CpuRam request region");
    }
    if (pair_->controller_reply.kind != MemKind::CpuRam) {
        throw std::runtime_error("local_copy: CPU-only coprocessor expects CpuRam reply region");
    }

    auto *request_base = static_cast<std::byte *>(local_request_.addr);
    const auto *request_hdr = reinterpret_cast<const LocalRequestHeader *>(request_base);
    const std::uint64_t request_bytes = request_hdr->bytes;
    if (request_bytes > local_request_.size - kLocalRequestHeaderBytes) {
        throw std::runtime_error("local_copy: request exceeds local request region capacity");
    }

    const void *request = request_base + kLocalRequestHeaderBytes;

    auto *reply_base =
        reinterpret_cast<std::byte *>(static_cast<std::uintptr_t>(peer_reply_.remote_addr));
    const std::size_t reply_cap =
        static_cast<std::size_t>(peer_reply_.size - kLocalReplyHeaderBytes);
    void *reply = reply_base + kLocalReplyHeaderBytes;
    CoprocessorFn fn = fn_ ? fn_ : &echo_fn;
    const std::size_t written =
        fn(request, static_cast<std::size_t>(request_bytes), reply, reply_cap, ctx_);
    if (written > reply_cap) {
        throw std::runtime_error("local_copy: coprocessor wrote past reply capacity");
    }

    auto *reply_hdr = reinterpret_cast<LocalReplyHeader *>(reply_base);
    reply_hdr->bytes = static_cast<std::uint64_t>(written);
    return 0;
}

} // namespace catalyst::transport::local_copy
