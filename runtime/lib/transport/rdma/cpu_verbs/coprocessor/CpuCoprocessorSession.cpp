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

#include "CpuCoprocessorSession.hpp"

#include <algorithm>
#include <cstring>
#include <thread>

#include "Frame.hpp"
#include "RealtimeThread.hpp"

namespace catalyst::transport::cpu_verbs {
using namespace catalyst::transport::common;

void CpuCoprocessorSession::set_coprocessor_fn(CoprocessorFn fn, void *ctx) {
    coproc_fn_ = fn;
    coproc_ctx_ = ctx;
}

void CpuCoprocessorSession::start() {
    stop();
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    completed_.store(0, std::memory_order_relaxed);
    last_word_.store(0, std::memory_order_relaxed);
    // jthread injects the stop_token. A data-path TP_CHECK throws TransportError;
    // it must not escape the thread function (that would std::terminate).
    // Capture it into error_ and publish via failed_ (release) so collect()
    // can rethrow the real exception.
    auto body = [this](std::stop_token st) {
        try {
            run(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    };
    engine_ = std::jthread(body);
}

int CpuCoprocessorSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                   std::size_t n) {
    while (completed_.load(std::memory_order_acquire) == 0) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_); // surface the engine's real error
        }
        if (!engine_.joinable() || engine_.get_stop_token().stop_requested()) {
            break;
        }
        std::this_thread::yield();
    }
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }
    // Stopped before any round completed -> no data (non-exceptional).
    if (completed_.load(std::memory_order_acquire) == 0) {
        return -1;
    }
    if (n > 0 && replies && replies[0]) {
        const std::uint64_t w = last_word_.load(std::memory_order_relaxed);
        const std::size_t cap = replies_bytes ? replies_bytes[0] : sizeof(w);
        TP_CHECK(cap <= sizeof(w), "reply capacity (%zu) exceeds the %zu B payload", cap,
                 sizeof(w));
        std::memcpy(replies[0], &w, cap);
    }
    return 0;
}

void CpuCoprocessorSession::stop() {
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
}

// Coprocessor: wait for a message, run the coprocessor function into the send
// buffer in place, then send the result. A null fn is the built-in echo
// (passthrough). Replies are inline + selectively signaled; the bwd CQ is
// reaped in batches at signal points.
void CpuCoprocessorSession::run(std::stop_token st) {
    pin_thread(pin_cpu_, pin_realtime_);
    if (ring_.mode == frame::Mode::BL16) {
        serve<frame::Mode::BL16>(st);
    } else {
        serve<frame::Mode::BLV>(st);
    }
}

// Compiled once per frame mode, so the BL16 loop has no mode checks.
template <frame::Mode M> void CpuCoprocessorSession::serve(std::stop_token st) {
    int signaled_outstanding = 0;
    for (std::uint64_t c = 0; !st.stop_requested(); c++) {
        frame::FrameView r; // the incoming message
        if (!poll_message_arrival<M>(c, st, r)) {
            reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/true);
            return;
        }
        const std::size_t len = M == frame::Mode::BL16 ? PAYLOAD_DATA_BYTES : r.len;
        std::uint64_t word = 0;
        std::memcpy(&word, r.data, std::min<std::size_t>(len, sizeof(word)));
        last_word_.store(word, std::memory_order_relaxed);
        completed_.fetch_add(1, std::memory_order_release);
        // The reply goes back under the request's fn_id. Its payload area is zero-filled for the
        // function to fill, so the controller reads zeros past what it wrote; the built-in echo
        // copies the request's payload instead.
        void *send = send_frame();
        std::uint8_t *reply = frame::data<M>(ring_, send);
        const std::size_t cap = frame::data_capacity<M>(ring_);
        std::memset(reply, 0, cap);
        std::size_t nb = len;
        if (coproc_fn_) {
            nb = coproc_fn_(r.data, len, r.fn_id, reply, cap, coproc_ctx_);
            TP_CHECK(nb > 0 && nb <= cap, "coprocessor function wrote %zu bytes, expected 1..%zu",
                     nb, cap);
        } else {
            std::memcpy(reply, r.data, nb);
        }
        frame::stage<M>(ring_, send, r.fn_id, reply, nb);
        const bool sig = (c % SIGNAL_EVERY == 0);
        post_write<M>(bwd_qp_->get(), c, /*inline_data=*/true, sig);
        if (sig) {
            ++signaled_outstanding;
            reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/false);
        }
    }
    reap(bwd_cq_->get(), signaled_outstanding, /*drain=*/true);
}

} // namespace catalyst::transport::cpu_verbs
