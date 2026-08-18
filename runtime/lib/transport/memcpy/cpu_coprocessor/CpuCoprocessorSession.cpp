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

#include "Error.hpp"

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

void CpuCoprocessorSession::start() {
    stop();
    process_cursor_ = 0;
    request_ring_.fill(common::PayloadSlot{});
    reply_ring_.fill(common::PayloadSlot{});
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    // Capture exceptions into error_ (release-published via failed_) so process_message can
    // surface the real error instead of the thread std::terminate-ing on an escape.
    engine_ = std::jthread([this](std::stop_token st) {
        try {
            run(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    });
}

int CpuCoprocessorSession::collect(void *const * /*replies*/,
                                   const std::uint64_t * /*replies_bytes*/, std::size_t /*n*/) {
    // Compute is driven inline from the controller's kick(); nothing collects on this side.
    throw std::logic_error("Coprocessor collect unused");
}

void CpuCoprocessorSession::stop() {
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
}

void CpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    // The worker thread reads fn_/ctx_ from run() without synchronization, so mutating them
    // while it runs would race and could tear across the two reads (calling fn with the wrong
    // ctx). The interface already documents this as bind-before-start; enforce it here.
    TP_CHECK(!engine_.joinable(), "Bind set_coprocessor_fn before start()");
    fn_ = fn;
    ctx_ = ctx;
}

// Worker loop: consume requests in order, run the bound fn (or echo), and publish the reply.
// Uses the wait-until-seq-matches + acquire/release-fence pattern the ring is designed around.
void CpuCoprocessorSession::run(std::stop_token st) {
    constexpr std::uint32_t STOP_CHECK_SPINS = 4096;
    for (std::uint64_t c = 0; !st.stop_requested(); ++c) {
        const std::size_t idx = c & (common::K_RING_SLOTS - 1);
        const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
        volatile std::uint32_t *rseq = &request_ring_[idx].p.seq_num;
        // Periodically check for stop so a request that never arrives can't hang teardown.
        std::uint32_t spins = 0;
        while (*rseq != expect) {
            if (++spins == STOP_CHECK_SPINS) {
                spins = 0;
                if (st.stop_requested()) {
                    return;
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);

        // Match cpu_verbs's contract: fn writes into PAYLOAD_DATA_BYTES of the reply's value slot;
        // the request frame is handed to the fn intact so decoder_id / seq_num remain readable.
        common::PayloadSlot &out = reply_ring_[idx];
        out.p.value = 0;
        out.p.decoder_id = request_ring_[idx].p.decoder_id;
        CoprocessorFn fn = fn_ ? fn_ : &echo_fn;
        const std::size_t nb = fn(&request_ring_[idx].p, sizeof(common::Payload), &out.p.value,
                                  common::PAYLOAD_DATA_BYTES, ctx_);
        TP_CHECK(nb <= common::PAYLOAD_DATA_BYTES, "Coprocessor fn overran reply");
        std::atomic_thread_fence(std::memory_order_release);
        out.p.seq_num = expect; // publish
    }
}

std::size_t CpuCoprocessorSession::process_message(const void *in, std::size_t in_len, void *out,
                                                   std::size_t out_cap) {
    TP_CHECK(in_len == sizeof(common::Payload), "Expected one wire-shaped Payload");
    TP_CHECK(engine_.joinable(), "Call start() before process_message");
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_); // surface the worker's real error
    }

    // Publish the request into ring[cursor % K]. Copy payload first, then release-fence, then
    // seq_num — so the worker never observes a slot where seq matches but the payload trails.
    const std::uint64_t c = process_cursor_++;
    const std::size_t idx = c & (common::K_RING_SLOTS - 1);
    const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
    std::memcpy(&request_ring_[idx].p, in, sizeof(common::Payload));
    std::atomic_thread_fence(std::memory_order_release);
    request_ring_[idx].p.seq_num = expect;

    // Spin-wait for the worker to publish reply_ring_[idx] with the matching seq.
    volatile std::uint32_t *sseq = &reply_ring_[idx].p.seq_num;
    while (*sseq != expect) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_);
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    const std::size_t n = std::min(out_cap, common::PAYLOAD_DATA_BYTES);
    if (n != 0 && out) {
        std::memcpy(out, &reply_ring_[idx].p.value, n);
    }
    return n;
}

} // namespace catalyst::transport::memcpy
