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

namespace catalyst::transport::local_copy {
namespace {

std::size_t echo_fn(const void *in, std::size_t in_len, void *out, std::size_t out_cap, void *) {
    const std::size_t n = std::min(in_len, out_cap);
    if (n != 0 && in && out) {
        std::memcpy(out, in, n);
    }
    return n;
}

} // namespace

LocalCpuCoprocessorSession::~LocalCpuCoprocessorSession() {
    if (pair_) {
        // Wait for any in-flight kick, then unbind so no future call reaches a dying `this`.
        std::lock_guard<std::mutex> lock(pair_->mu);
        pair_->run_once = nullptr;
    }
}

int LocalCpuCoprocessorSession::connect(const ConnectInfo &info) {
    pair_ = acquire_endpoint_pair(info);
    std::lock_guard<std::mutex> lock(pair_->mu);
    if (pair_->run_once) {
        throw std::runtime_error(
            "memcpy: another coprocessor is already bound to this endpoint (peer, oob_port)");
    }
    pair_->run_once = [this](const void *req, std::size_t req_bytes, std::uint32_t decoder_id,
                             void *reply, std::size_t reply_cap) {
        return this->run_once(req, req_bytes, decoder_id, reply, reply_cap);
    };
    return 0;
}

MemRegion LocalCpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    if (kind != MemKind::CpuRam) {
        throw std::runtime_error("memcpy: CPU-only for now; alloc_memory expects CpuRam");
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

PeerRef LocalCpuCoprocessorSession::exchange_keys(const MemRegion & /*local*/) {
    return PeerRef{};
}

void LocalCpuCoprocessorSession::establish_channel(const ChannelDesc &desc,
                                                   const MemRegion & /*local*/,
                                                   const PeerRef & /*peer*/) {
    if (desc.transport != "memcpy") {
        throw std::runtime_error("memcpy: CPU-only coprocessor supports only transport=memcpy");
    }
}

void LocalCpuCoprocessorSession::start() {}

int LocalCpuCoprocessorSession::collect(void *const * /*replies*/,
                                        const std::uint64_t * /*replies_bytes*/,
                                        std::size_t /*n*/) {
    // Compute is driven inline from the controller's kick(); nothing collects on this side.
    throw std::logic_error("memcpy: coprocessor collect is not used");
}

void LocalCpuCoprocessorSession::stop() {}

void LocalCpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    fn_ = fn;
    ctx_ = ctx;
}

std::size_t LocalCpuCoprocessorSession::run_once(const void *req, std::size_t req_bytes,
                                                 std::uint32_t /*decoder_id*/, void *reply,
                                                 std::size_t reply_cap) {
    CoprocessorFn fn = fn_ ? fn_ : &echo_fn;
    const std::size_t written = fn(req, req_bytes, reply, reply_cap, ctx_);
    if (written > reply_cap) {
        throw std::runtime_error("memcpy: coprocessor wrote past reply capacity");
    }
    return written;
}

} // namespace catalyst::transport::local_copy
