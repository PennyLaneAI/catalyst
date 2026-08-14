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

#include "CpuCoprocessorSession.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace catalyst::transport::memcpy {
namespace {

std::size_t echo_fn(const void *in, std::size_t in_len, void *out, std::size_t out_cap, void *) {
    const std::size_t n = std::min(in_len, out_cap);
    if (n != 0 && in && out) {
        std::memcpy(out, in, n);
    }
    return n;
}

} // namespace

CpuCoprocessorSession::~CpuCoprocessorSession() {
    if (link_) {
        // Wait for any in-flight kick, then unbind so no future call reaches a dying `this`.
        std::lock_guard<std::mutex> lock(link_->mu);
        link_->process_message = nullptr;
    }
}

int CpuCoprocessorSession::connect(const ConnectInfo & /*info*/) {
    // Only bind `link_` after the duplicate-binding check succeeds. If we assigned it up front
    // and threw, the destructor would clear the incumbent coprocessor's binding on the way out.
    auto candidate = acquire_memcpy_link(pair_key_);
    std::lock_guard<std::mutex> lock(candidate->mu);
    if (candidate->process_message) {
        throw std::runtime_error("memcpy: another coprocessor is already bound to session pair '" +
                                 pair_key_ + "'");
    }
    candidate->process_message = [this](const void *in, std::size_t in_len, void *out,
                                        std::size_t out_cap) {
        return this->process_message(in, in_len, out, out_cap);
    };
    link_ = std::move(candidate);
    return 0;
}

MemRegion CpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    if (kind != MemKind::CpuRam) {
        throw std::runtime_error("CPU device can only allocate CpuRam");
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

PeerRef CpuCoprocessorSession::exchange_keys(const MemRegion & /*local*/) { return PeerRef{}; }

void CpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion & /*local*/,
                                              const PeerRef & /*peer*/) {
    if (desc.transport != "memcpy") {
        throw std::runtime_error("memcpy: CPU-only coprocessor supports only transport=memcpy");
    }
}

void CpuCoprocessorSession::start() {}

int CpuCoprocessorSession::collect(void *const * /*replies*/,
                                   const std::uint64_t * /*replies_bytes*/, std::size_t /*n*/) {
    // Compute is driven inline from the controller's kick(); nothing collects on this side.
    throw std::logic_error("memcpy: coprocessor collect is not used");
}

void CpuCoprocessorSession::stop() {}

void CpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    fn_ = fn;
    ctx_ = ctx;
}

std::size_t CpuCoprocessorSession::process_message(const void *in, std::size_t in_len, void *out,
                                                   std::size_t out_cap) {
    // `in` is a wire-shaped Payload (matching cpu_verbs); decoder_id sits at offset 8
    // for any decoder that needs to dispatch on it.
    CoprocessorFn fn = fn_ ? fn_ : &echo_fn;
    const std::size_t out_bytes = fn(in, in_len, out, out_cap, ctx_);
    if (out_bytes > out_cap) {
        throw std::runtime_error("memcpy: coprocessor wrote past reply capacity");
    }
    return out_bytes;
}

} // namespace catalyst::transport::memcpy
