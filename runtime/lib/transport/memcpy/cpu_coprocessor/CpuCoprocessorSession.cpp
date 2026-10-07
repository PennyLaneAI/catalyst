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

#include <stdexcept>

#include "Error.hpp"

namespace catalyst::transport::memcpy {

CpuCoprocessorSession::~CpuCoprocessorSession() {
    // Stop the worker first so a lingering thread cannot touch `this` after teardown.
    try {
        stop();
    } catch (...) {
    }
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
    TP_CHECK(!candidate->process_message, "Coprocessor already bound to pair '%s'",
             pair_key_.c_str());
    candidate->process_message = [this](const void *in, std::size_t in_len, void *out,
                                        std::size_t out_cap) {
        return this->process_message(in, in_len, out, out_cap);
    };
    link_ = std::move(candidate);
    return 0;
}

MemRegion CpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
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

PeerRef CpuCoprocessorSession::exchange_keys(const MemRegion & /*local*/) { return PeerRef{}; }

void CpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion & /*local*/,
                                              const PeerRef & /*peer*/) {
    TP_CHECK(desc.transport == "memcpy", "Only transport=memcpy is supported");
}

void CpuCoprocessorSession::start() { worker_.start(); }

int CpuCoprocessorSession::collect(void *const * /*replies*/,
                                   const std::uint64_t * /*replies_bytes*/, std::size_t /*n*/) {
    // Compute is driven inline from the controller's kick(); nothing collects on this side.
    throw std::logic_error("Coprocessor collect unused");
}

void CpuCoprocessorSession::stop() { worker_.stop(); }

void CpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    worker_.bind(fn, ctx);
}

std::size_t CpuCoprocessorSession::process_message(const void *in, std::size_t in_len, void *out,
                                                   std::size_t out_cap) {
    return worker_.process_message(in, in_len, out, out_cap);
}

} // namespace catalyst::transport::memcpy
