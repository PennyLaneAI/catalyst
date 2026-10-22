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

#include "CpuControllerSession.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stop_token>

#include "Error.hpp"
#include "Frame.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::cpu_verbs {
using namespace catalyst::transport::common;

namespace {
std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}
} // namespace

void CpuControllerSession::start() {
    stop(); // drain any leftovers + reset (idempotent)
    next_send_ = 0;
    next_recv_ = 0;
    signaled_outstanding_ = 0;
    rtt_ns_ = 0;
}

void CpuControllerSession::stop() {
    if (fwd_cq_) {
        reap(fwd_cq_->get(), signaled_outstanding_, /*drain=*/true);
    }
}

// Single work item, fixed-size frame: work_item_idx is ignored; the sizes are
// just recorded (out_bytes_ caps the reply in collect()).
void CpuControllerSession::commit_work_item(std::uint32_t /*work_item_idx*/, std::uint64_t in_bytes,
                                            std::uint64_t out_bytes) {
    const std::size_t cap = frame::data_capacity(ring_);
    TP_CHECK(in_bytes <= cap, "in_bytes (%zu) exceeds the %zu B payload",
             static_cast<std::size_t>(in_bytes), cap);
    TP_CHECK(out_bytes <= cap, "out_bytes (%zu) exceeds the %zu B payload",
             static_cast<std::size_t>(out_bytes), cap);
    in_bytes_ = in_bytes;
    out_bytes_ = out_bytes;
}

void *CpuControllerSession::data_slot() {
    // Current round's outbound slot: the caller writes up to in_bytes_ here, then kick()s.
    return frame::data(ring_, send_frame());
}

void CpuControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                           std::uint32_t decoder_id) {
    TP_CHECK(bytes <= in_bytes_, "payload (%zu B) exceeds the %zu B committed for this round",
             static_cast<std::size_t>(bytes), static_cast<std::size_t>(in_bytes_));
    frame::stage(ring_, send_frame(), decoder_id, src, static_cast<std::size_t>(bytes));
    staged_ = true;
    fn_id_ = decoder_id;
}

int CpuControllerSession::kick(std::uint32_t /*work_item_idx*/) {
    if (!staged_) {
        void *send = send_frame();
        frame::stage(ring_, send, fn_id_, frame::data(ring_, send),
                     static_cast<std::size_t>(in_bytes_));
    }
    staged_ = false;
    kick_ns_ = now_ns();
    const bool sig = (next_send_ % SIGNAL_EVERY == 0);
    if (cuqr()) {
        post_write<true>(fwd_qp_->get(), next_send_, /*inline_data=*/true, sig);
    } else {
        post_write(fwd_qp_->get(), next_send_, /*inline_data=*/true, sig);
    }
    if (sig) {
        ++signaled_outstanding_;
        reap(fwd_cq_->get(), signaled_outstanding_, /*drain=*/false); // lazy: free the SQ
    }
    ++next_send_;
    return 0;
}

void *CpuControllerSession::reply_slot() { return frame::data(ring_, ring_slot(next_recv_)); }

int CpuControllerSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                  std::size_t n) {
    return cuqr() ? collect_frame<true>(replies, replies_bytes, n)
                  : collect_frame<false>(replies, replies_bytes, n);
}

template <bool CUQR>
int CpuControllerSession::collect_frame(void *const *replies, const std::uint64_t *replies_bytes,
                                        std::size_t n) {
    std::stop_token none; // blocking wait for this round's reply
    frame::FrameView r;
    // UC (a cudaq-realtime peer) drops a lost message without telling either side.
    const auto timeout =
        CUQR && cuqr_.cudaq_peer ? CUDAQ_REPLY_TIMEOUT : std::chrono::nanoseconds{0};
    if (!poll_message_arrival<CUQR>(next_recv_, none, r, timeout)) {
        return -1;
    }
    rtt_ns_ = now_ns() - kick_ns_;
    ++next_recv_;
    if constexpr (CUQR) { // next_recv_ is now this round's seq, the request_id the request carried
        TP_CHECK(r.data != nullptr, "cpu_verbs: the coprocessor's reply is not a CUQR frame");
        TP_CHECK(r.request_id == static_cast<std::uint32_t>(next_recv_),
                 "cpu_verbs: a CUQR reply answers request %u, expected %u", r.request_id,
                 static_cast<std::uint32_t>(next_recv_));
        TP_CHECK(r.status == 0, "cpu_verbs: the coprocessor answered with status %d", r.status);
    }
    if (n > 0 && replies && replies[0] && replies[0] != r.data) {
        // The coprocessor zero-fills the reply's payload area, so all of `cap` is defined.
        const std::size_t cap = replies_bytes ? replies_bytes[0] : out_bytes_;
        TP_CHECK(cap <= frame::data_capacity(ring_),
                 "reply capacity (%zu) exceeds the %zu B a frame of this session carries", cap,
                 frame::data_capacity(ring_));
        if constexpr (CUQR) { // a CUQR frame carries only the reply's own bytes
            const std::size_t len = std::min<std::size_t>(r.len, cap);
            std::memcpy(replies[0], r.data, len);
            std::memset(static_cast<std::uint8_t *>(replies[0]) + len, 0, cap - len);
        } else {
            std::memcpy(replies[0], r.data, cap);
        }
    }
    return 0;
}

} // namespace catalyst::transport::cpu_verbs
