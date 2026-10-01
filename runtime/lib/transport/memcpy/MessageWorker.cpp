// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "MessageWorker.hpp"

#include <algorithm>
#include <cstring>
#include <initializer_list>

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

MessageWorker::MessageWorker()
    : request_ring_(common::K_RING_SLOTS), reply_ring_(common::K_RING_SLOTS) {}

MessageWorker::~MessageWorker() {
    try {
        stop();
    } catch (...) {
    }
}

void MessageWorker::bind(CoprocessorFn fn, void *ctx) {
    // run() reads fn_ and ctx_ without synchronization, so they are fixed before it starts.
    TP_CHECK(!running(), "Bind set_coprocessor_fn before start()");
    fn_ = fn;
    ctx_ = ctx;
}

void MessageWorker::start(ThreadInit init) {
    stop();
    process_cursor_ = 0;
    for (auto *ring : {&request_ring_, &reply_ring_}) {
        for (Slot &slot : *ring) {
            slot.bytes = slot.cap = slot.seq = 0;
        }
    }
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    // Exceptions are captured into error_ so process_message surfaces the real error instead of
    // the thread terminating the process.
    engine_ = std::jthread([this, init = std::move(init)](std::stop_token st) {
        try {
            if (init) {
                init();
            }
            run(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    });
}

void MessageWorker::stop() {
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
}

void MessageWorker::run(std::stop_token st) {
    constexpr std::uint32_t STOP_CHECK_SPINS = 4096;
    for (std::uint64_t c = 0; !st.stop_requested(); ++c) {
        const std::size_t idx = c & (common::K_RING_SLOTS - 1);
        const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
        const Slot &req = request_ring_[idx];
        volatile const std::uint32_t *rseq = &req.seq;
        // Check for stop periodically, so a request that never arrives cannot hang teardown.
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

        Slot &out = reply_ring_[idx];
        const std::size_t cap = std::max<std::size_t>(req.cap, common::PAYLOAD_DATA_BYTES);
        std::memset(out.data, 0, cap);
        CoprocessorFn fn = fn_ ? fn_ : &echo_fn;
        const std::size_t nb = fn(req.data, req.bytes, out.data, cap, ctx_);
        if (nb == COPROCESSOR_FN_ERROR) {
            out.bytes = FAILED_REPLY;
        } else {
            TP_CHECK(nb <= cap, "Coprocessor fn overran reply");
            out.bytes = static_cast<std::uint32_t>(nb);
        }
        std::atomic_thread_fence(std::memory_order_release);
        out.seq = expect; // publish
    }
}

std::size_t MessageWorker::process_message(const void *in, std::size_t in_len, void *out,
                                           std::size_t out_cap) {
    TP_CHECK(in_len <= common::MAX_FRAME_BYTES, "Request frame of %zu B exceeds %zu B", in_len,
             common::MAX_FRAME_BYTES);
    TP_CHECK(out_cap <= common::MAX_MESSAGE_BYTES, "Reply of %zu B exceeds %zu B", out_cap,
             common::MAX_MESSAGE_BYTES);
    TP_CHECK(running(), "Call start() before process_message");
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }
    // Write the frame, then release-fence, then seq, so the worker never sees a matching seq
    // before the frame it covers.
    const std::uint64_t c = process_cursor_++;
    const std::size_t idx = c & (common::K_RING_SLOTS - 1);
    const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
    Slot &req = request_ring_[idx];
    if (in_len != 0) {
        std::memcpy(req.data, in, in_len);
    }
    req.bytes = static_cast<std::uint32_t>(in_len);
    req.cap = static_cast<std::uint32_t>(out_cap);
    std::atomic_thread_fence(std::memory_order_release);
    req.seq = expect;

    const Slot &rep = reply_ring_[idx];
    volatile const std::uint32_t *sseq = &rep.seq;
    while (*sseq != expect) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_);
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    TP_CHECK(rep.bytes != FAILED_REPLY, "The coprocessor function failed to process message %llu",
             static_cast<unsigned long long>(c));
    if (out_cap != 0 && out) {
        std::memcpy(out, rep.data, out_cap);
    }
    return out_cap;
}

} // namespace catalyst::transport::memcpy
