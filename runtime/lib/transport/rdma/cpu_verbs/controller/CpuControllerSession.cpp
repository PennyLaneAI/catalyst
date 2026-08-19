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
    TP_CHECK(in_bytes <= PAYLOAD_DATA_BYTES, "in_bytes (%zu) exceeds the %zu B payload",
             static_cast<std::size_t>(in_bytes), PAYLOAD_DATA_BYTES);
    TP_CHECK(out_bytes <= PAYLOAD_DATA_BYTES, "out_bytes (%zu) exceeds the %zu B payload",
             static_cast<std::size_t>(out_bytes), PAYLOAD_DATA_BYTES);
    in_bytes_ = in_bytes;
    out_bytes_ = out_bytes;
}

void *CpuControllerSession::data_slot() {
    // Current round's outbound slot: the caller writes up to in_bytes_ here, then kick()s.
    return &send_payload()->value;
}

void CpuControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                           std::uint32_t decoder_id) {
    TP_CHECK(bytes <= in_bytes_, "payload (%zu B) exceeds the %zu B committed for this round",
             static_cast<std::size_t>(bytes), static_cast<std::size_t>(in_bytes_));
    Payload *send = send_payload();
    send->value = 0;
    std::memcpy(&send->value, src, bytes);
    send->decoder_id = decoder_id;
}

int CpuControllerSession::kick(std::uint32_t /*work_item_idx*/) {
    // The payload was written into data_slot() by the caller; fire one round.
    kick_ns_ = now_ns();
    const bool sig = (next_send_ % SIGNAL_EVERY == 0);
    post_write(fwd_qp_->get(), next_send_, /*inline_data=*/true, sig);
    if (sig) {
        ++signaled_outstanding_;
        reap(fwd_cq_->get(), signaled_outstanding_, /*drain=*/false); // lazy: free the SQ
    }
    ++next_send_;
    return 0;
}

void *CpuControllerSession::reply_slot() {
    auto *ring = reinterpret_cast<common::PayloadSlot *>(local_.addr);
    return &ring[next_recv_ & (common::K_RING_SLOTS - 1)].p.value;
}

int CpuControllerSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                  std::size_t n) {
    std::stop_token none; // blocking wait for this round's reply
    Payload *r = poll_message_arrival(next_recv_, none);
    if (!r) {
        return -1;
    }
    rtt_ns_ = now_ns() - kick_ns_;
    ++next_recv_;
    if (n > 0 && replies && replies[0] && replies[0] != &r->value) {
        const std::size_t cap = replies_bytes ? replies_bytes[0] : out_bytes_;
        TP_CHECK(cap <= PAYLOAD_DATA_BYTES,
                 "reply capacity (%zu) exceeds the %zu B payload; a round carries one "
                 "error index",
                 cap, PAYLOAD_DATA_BYTES);
        std::memcpy(replies[0], &r->value, cap);
    }
    return 0;
}

} // namespace catalyst::transport::cpu_verbs
