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

#include "LocalCpuControllerSession.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "LocalWireProtocol.hpp"

namespace catalyst::transport::local_copy {
namespace {

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

std::size_t local_region_bytes(std::size_t payload_bytes) {
    return payload_bytes + std::max(kLocalRequestHeaderBytes, kLocalReplyHeaderBytes);
}

} // namespace

int LocalCpuControllerSession::connect(const ConnectInfo &info) {
    pair_ = acquire_endpoint_pair(info);
    pair_->controller = this;
    return 0;
}

MemRegion LocalCpuControllerSession::alloc_memory(std::size_t size, MemKind kind) {
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

PeerRef LocalCpuControllerSession::exchange_keys(const MemRegion &local) {
    local_reply_ = local;
    if (pair_) {
        pair_->controller_reply = local;
        pair_->controller_reply_ready = true;
        if (pair_->coprocessor_request_ready) {
            peer_request_ = PeerRef{
                .rkey = 0,
                .remote_addr = reinterpret_cast<std::uint64_t>(pair_->coprocessor_request.addr),
                .size = pair_->coprocessor_request.size,
            };
        }
    }
    return peer_request_;
}

void LocalCpuControllerSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                                  const PeerRef &peer) {
    if (desc.transport != "local") {
        throw std::runtime_error("local_copy: CPU-only controller supports only transport=local");
    }
    local_reply_ = local;
    peer_request_ = peer;
    if (peer_request_.remote_addr == 0 && pair_ && pair_->coprocessor_request_ready) {
        peer_request_ = PeerRef{
            .rkey = 0,
            .remote_addr = reinterpret_cast<std::uint64_t>(pair_->coprocessor_request.addr),
            .size = pair_->coprocessor_request.size,
        };
    }
}

void LocalCpuControllerSession::start() { rtt_ns_ = 0; }

int LocalCpuControllerSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                       std::size_t n) {
    if (!local_reply_.addr) {
        throw std::runtime_error("local_copy: controller reply region is not established");
    }

    if (local_reply_.kind != MemKind::CpuRam) {
        throw std::runtime_error("local_copy: CPU-only controller expects CpuRam reply region");
    }

    auto *base = static_cast<std::byte *>(local_reply_.addr);
    const auto *hdr = reinterpret_cast<const LocalReplyHeader *>(base);
    const std::uint64_t reply_bytes = hdr->bytes;
    const std::uint64_t cap = replies_bytes ? replies_bytes[0] : out_bytes_;

    if (reply_bytes > out_bytes_) {
        throw std::runtime_error("local_copy: reply exceeds committed output bytes");
    }
    if (reply_bytes > cap) {
        throw std::runtime_error("local_copy: caller reply buffer too small");
    }
    if (reply_bytes > local_reply_.size - kLocalReplyHeaderBytes) {
        throw std::runtime_error("local_copy: reply exceeds local reply region capacity");
    }
    if (n > 0 && replies && replies[0] && reply_bytes != 0) {
        std::memcpy(replies[0], base + kLocalReplyHeaderBytes,
                    static_cast<std::size_t>(reply_bytes));
    }

    rtt_ns_ = now_ns() - kick_ns_;
    return 0;
}

void LocalCpuControllerSession::stop() {}

void LocalCpuControllerSession::commit_work_item(std::uint32_t /*work_item_idx*/,
                                                 std::uint64_t in_bytes, std::uint64_t out_bytes) {
    in_bytes_ = in_bytes;
    out_bytes_ = out_bytes;
    staged_bytes_ = 0;
    request_staging_.resize(static_cast<std::size_t>(in_bytes_));
}

int LocalCpuControllerSession::kick(std::uint32_t /*work_item_idx*/) {
    if (!pair_ || !pair_->coprocessor) {
        throw std::runtime_error("local_copy: no paired coprocessor");
    }
    if (peer_request_.remote_addr == 0) {
        throw std::runtime_error("local_copy: peer request region is not established");
    }
    if (peer_request_.size < kLocalRequestHeaderBytes + staged_bytes_) {
        throw std::runtime_error("local_copy: peer request region too small for staged payload");
    }
    if (!local_reply_.addr || local_reply_.size < kLocalReplyHeaderBytes) {
        throw std::runtime_error("local_copy: local reply region is not established");
    }

    if (pair_->coprocessor_request.kind != MemKind::CpuRam) {
        throw std::runtime_error(
            "local_copy: CPU-only controller expects CpuRam peer request region");
    }
    if (local_reply_.kind != MemKind::CpuRam) {
        throw std::runtime_error("local_copy: CPU-only controller expects CpuRam reply region");
    }

    auto *reply_base = static_cast<std::byte *>(local_reply_.addr);
    auto *reply_hdr = reinterpret_cast<LocalReplyHeader *>(reply_base);
    reply_hdr->bytes = 0;

    auto *base =
        reinterpret_cast<std::byte *>(static_cast<std::uintptr_t>(peer_request_.remote_addr));
    auto *hdr = reinterpret_cast<LocalRequestHeader *>(base);
    hdr->bytes = staged_bytes_;
    hdr->decoder_id = decoder_id_;
    if (staged_bytes_ != 0) {
        std::memcpy(base + kLocalRequestHeaderBytes, request_staging_.data(),
                    static_cast<std::size_t>(staged_bytes_));
    }

    kick_ns_ = now_ns();
    return pair_->coprocessor->run_once();
}

void *LocalCpuControllerSession::data_slot() {
    return request_staging_.empty() ? nullptr : request_staging_.data();
}

void LocalCpuControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                                std::uint32_t decoder_id) {
    if (bytes > in_bytes_) {
        throw std::runtime_error("local_copy: payload exceeds committed input bytes");
    }
    if (bytes != 0 && src == nullptr) {
        throw std::runtime_error("local_copy: null source with non-zero payload size");
    }
    if (bytes != 0) {
        std::memcpy(request_staging_.data(), src, static_cast<std::size_t>(bytes));
    }
    staged_bytes_ = bytes;
    decoder_id_ = decoder_id;
}

} // namespace catalyst::transport::local_copy
