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

#include "Error.hpp"

namespace catalyst::transport::memcpy {
namespace frame = common::frame;

namespace {

std::size_t echo_fn(const void *payload, std::size_t len, std::uint32_t, void *out,
                    std::size_t out_cap, void *) {
    const std::size_t n = std::min(len, out_cap);
    if (n != 0 && payload && out) {
        std::memcpy(out, payload, n);
    }
    return n;
}

} // namespace

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
    const std::size_t region = ring_.region_bytes();
    storage_.assign(2 * region + ring_.stride, 0);
    const auto addr = reinterpret_cast<std::uintptr_t>(storage_.data());
    request_ring_ = storage_.data() + (ring_.stride - addr % ring_.stride) % ring_.stride;
    reply_ring_ = request_ring_ + region;
    failed_slot_.assign(ring_.slots, 0);
    process_cursor_ = 0;
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    started_ = true;
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
    started_ = false;
}

void MessageWorker::run(std::stop_token st) {
    constexpr std::uint32_t STOP_CHECK_SPINS = 4096;
    const std::size_t cap = frame::data_capacity(ring_);
    for (std::uint64_t c = 0; !st.stop_requested(); ++c) {
        const auto expect = static_cast<std::uint32_t>(c + 1);
        // Check for stop periodically, so a request that never arrives cannot hang teardown.
        frame::FrameView request;
        std::uint32_t spins = 0;
        while (!frame::poll(ring_, request_slot(c), expect, request)) {
            if (++spins == STOP_CHECK_SPINS) {
                spins = 0;
                if (st.stop_requested()) {
                    return;
                }
            }
        }
        std::uint8_t *reply = reply_slot(c);
        std::uint8_t *reply_data = frame::data(ring_, reply);
        std::memset(reply_data, 0, cap);
        CoprocessorFn fn = fn_ ? fn_ : &echo_fn;
        const std::size_t nb = fn(request.data, request.len, request.fn_id, reply_data, cap, ctx_);
        const std::size_t idx = c & (ring_.slots - 1);
        failed_slot_[idx] = nb == COPROCESSOR_FN_ERROR;
        if (!failed_slot_[idx]) {
            TP_CHECK(nb <= cap, "Coprocessor fn wrote %zu B, more than the %zu B reply holds", nb,
                     cap);
        }
        frame::stage(ring_, reply, request.fn_id, reply_data, failed_slot_[idx] ? 0 : nb);
        frame::seal(ring_, reply, expect);
    }
}

std::size_t MessageWorker::process_message(const void *payload, std::size_t len,
                                           std::uint32_t fn_id, void *out, std::size_t out_cap) {
    TP_CHECK(running(), "Call start() before process_message");
    // Checked before the cursor advances: a slot that is never sealed would stall the worker.
    TP_CHECK(len <= frame::data_capacity(ring_) && out_cap <= frame::data_capacity(ring_),
             "A %zu B request or %zu B reply exceeds the %zu B a frame of this session carries",
             len, out_cap, frame::data_capacity(ring_));
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }
    const std::uint64_t c = process_cursor_++;
    const auto expect = static_cast<std::uint32_t>(c + 1);
    // Staging checks the payload fits; sealing writes the polled word last.
    frame::stage(ring_, request_slot(c), fn_id, payload, len);
    frame::seal(ring_, request_slot(c), expect);

    frame::FrameView reply;
    while (!frame::poll(ring_, reply_slot(c), expect, reply)) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_);
        }
    }
    TP_CHECK(!failed_slot_[c & (ring_.slots - 1)],
             "The coprocessor function failed to process message %llu",
             static_cast<unsigned long long>(c));
    if (out_cap != 0 && out) {
        std::memcpy(out, reply.data, out_cap);
    }
    return out_cap;
}

} // namespace catalyst::transport::memcpy
