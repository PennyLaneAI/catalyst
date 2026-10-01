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
#include <new>

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

constexpr std::size_t SLOT_ALIGN = 64;

constexpr std::size_t round_up(std::size_t n, std::size_t to) { return (n + to - 1) / to * to; }

} // namespace

void MessageWorker::Ring::allocate(std::size_t capacity) {
    static_assert(sizeof(Header) == SLOT_ALIGN, "a slot header fills one 64 B line");
    stride_ = sizeof(Header) + round_up(std::max<std::size_t>(capacity, 1), SLOT_ALIGN);
    storage_.assign(common::K_RING_SLOTS * stride_ + SLOT_ALIGN, std::byte{0});
    const auto addr = reinterpret_cast<std::uintptr_t>(storage_.data());
    base_ = storage_.data() + (SLOT_ALIGN - addr % SLOT_ALIGN) % SLOT_ALIGN;
    capacity_ = capacity;
    for (std::size_t i = 0; i < common::K_RING_SLOTS; ++i) {
        new (slot(i)) Header{};
    }
}

void MessageWorker::Ring::release() {
    std::vector<std::byte>().swap(storage_);
    base_ = nullptr;
    stride_ = 0;
    capacity_ = 0;
}

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
    request_ring_.release();
    reply_ring_.release();
    process_cursor_ = 0;
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    init_ = std::move(init);
    started_ = true;
}

void MessageWorker::stop() {
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
    started_ = false;
}

void MessageWorker::launch(std::size_t frame_bytes, std::size_t reply_bytes) {
    const std::size_t reply_capacity =
        std::max<std::size_t>(reply_bytes, common::PAYLOAD_DATA_BYTES);
    try {
        request_ring_.allocate(frame_bytes);
        reply_ring_.allocate(reply_capacity);
    } catch (const std::bad_alloc &) {
        request_ring_.release();
        reply_ring_.release();
        TP_CHECK(false, "Cannot allocate the message rings for %zu B frames and %zu B replies",
                 frame_bytes, reply_capacity);
    }
    // Exceptions are captured into error_ so process_message surfaces the real error instead of
    // the thread terminating the process.
    engine_ = std::jthread([this](std::stop_token st) {
        try {
            if (init_) {
                init_();
            }
            run(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    });
}

void MessageWorker::run(std::stop_token st) {
    constexpr std::uint32_t STOP_CHECK_SPINS = 4096;
    for (std::uint64_t c = 0; !st.stop_requested(); ++c) {
        const std::size_t idx = c & (common::K_RING_SLOTS - 1);
        const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
        const Ring::Header &req = request_ring_.header(idx);
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

        Ring::Header &out = reply_ring_.header(idx);
        std::byte *out_data = reply_ring_.data(idx);
        const std::size_t cap =
            std::max<std::size_t>(static_cast<std::size_t>(req.cap), common::PAYLOAD_DATA_BYTES);
        std::memset(out_data, 0, cap);
        CoprocessorFn fn = fn_ ? fn_ : &echo_fn;
        const std::size_t nb =
            fn(request_ring_.data(idx), static_cast<std::size_t>(req.bytes), out_data, cap, ctx_);
        if (nb == COPROCESSOR_FN_ERROR) {
            out.bytes = FAILED_REPLY;
        } else {
            TP_CHECK(nb <= cap, "Coprocessor fn overran reply");
            out.bytes = nb;
        }
        std::atomic_thread_fence(std::memory_order_release);
        out.seq = expect; // publish
    }
}

std::size_t MessageWorker::process_message(const void *in, std::size_t in_len, void *out,
                                           std::size_t out_cap) {
    TP_CHECK(running(), "Call start() before process_message");
    if (!engine_.joinable()) {
        launch(in_len, out_cap);
    }
    TP_CHECK(in_len <= request_ring_.capacity(),
             "Request frame of %zu B exceeds the %zu B frame this session started with", in_len,
             request_ring_.capacity());
    TP_CHECK(out_cap <= reply_ring_.capacity(),
             "Reply of %zu B exceeds the %zu B reply this session started with", out_cap,
             reply_ring_.capacity());
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }
    // Write the frame, then release-fence, then seq, so the worker never sees a matching seq
    // before the frame it covers.
    const std::uint64_t c = process_cursor_++;
    const std::size_t idx = c & (common::K_RING_SLOTS - 1);
    const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
    Ring::Header &req = request_ring_.header(idx);
    if (in_len != 0) {
        std::memcpy(request_ring_.data(idx), in, in_len);
    }
    req.bytes = in_len;
    req.cap = out_cap;
    std::atomic_thread_fence(std::memory_order_release);
    req.seq = expect;

    const Ring::Header &rep = reply_ring_.header(idx);
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
        std::memcpy(out, reply_ring_.data(idx), out_cap);
    }
    return out_cap;
}

} // namespace catalyst::transport::memcpy
