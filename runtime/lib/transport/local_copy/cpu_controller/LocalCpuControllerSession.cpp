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

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace catalyst::transport::local_copy {
namespace {

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
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

PeerRef LocalCpuControllerSession::exchange_keys(const MemRegion &local) {
    local_reply_ = local;
    return PeerRef{};
}

void LocalCpuControllerSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                                  const PeerRef & /*peer*/) {
    if (desc.transport != "memcpy") {
        throw std::runtime_error("local_copy: CPU-only controller supports only transport=memcpy");
    }
    local_reply_ = local;
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

    const std::uint64_t cap = replies_bytes ? replies_bytes[0] : out_bytes_;
    if (reply_bytes_ > cap) {
        throw std::runtime_error("local_copy: caller reply buffer too small");
    }
    if (n > 0 && replies && replies[0] && reply_bytes_ != 0) {
        std::memcpy(replies[0], local_reply_.addr, static_cast<std::size_t>(reply_bytes_));
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
    if (!pair_ || !pair_->run_once) {
        throw std::runtime_error("local_copy: no paired coprocessor");
    }
    if (!local_reply_.addr) {
        throw std::runtime_error("local_copy: local reply region is not established");
    }
    if (local_reply_.kind != MemKind::CpuRam) {
        throw std::runtime_error("local_copy: CPU-only controller expects CpuRam reply region");
    }
    if (local_reply_.size < out_bytes_) {
        throw std::runtime_error("local_copy: local reply region too small for committed output");
    }

    kick_ns_ = now_ns();
    const std::size_t written = pair_->run_once(
        request_staging_.data(), static_cast<std::size_t>(staged_bytes_), decoder_id_,
        local_reply_.addr, static_cast<std::size_t>(out_bytes_));
    reply_bytes_ = static_cast<std::uint64_t>(written);
    return 0;
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
