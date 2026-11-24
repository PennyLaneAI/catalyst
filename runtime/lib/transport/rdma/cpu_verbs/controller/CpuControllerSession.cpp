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
#include <utility>

#include "Error.hpp"
#include "RealtimeThread.hpp"

namespace catalyst::transport::cpu_verbs {
using namespace catalyst::transport::common;

namespace {
// Page alignment for registered host buffers.
constexpr std::size_t PAGE_ALIGN = 4096;

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}
} // namespace

CpuControllerSession::CpuControllerSession(std::string dev, int gid_idx, frame::RingGeometry ring)
    : link_(std::move(dev), gid_idx, "cpu_verbs", RcLink::Side::Controller, ring), ring_(ring) {}

int CpuControllerSession::connect(const ConnectInfo &info) {
    link_.connect(info);
    return 0;
}

MemRegion CpuControllerSession::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(kind == MemKind::CpuRam, "cpu_verbs: only MemKind::CpuRam supported");
    // The peer writes a ring of frames into the region, which may be larger than asked for.
    size = std::max(size, ring_.region_bytes());
    regions_.push_back(
        MemoryRegion::alloc_host(link_.pd(), size, PAGE_ALIGN, MemAccess::REMOTE_WRITE));
    const MemoryRegion &mr = regions_.back();
    return MemRegion{
        .addr = mr.addr(),
        .size = size,
        .lkey = mr.lkey(),
        .rkey = mr.rkey(),
        .kind = MemKind::CpuRam,
    };
}

PeerRef CpuControllerSession::exchange_keys(const MemRegion &local) {
    return link_.exchange(local);
}

void CpuControllerSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                             const PeerRef &peer) {
    TP_CHECK(desc.transport == "rdma", "cpu_verbs: only transport \"rdma\" supported");
    ring_buf_ = static_cast<std::uint8_t *>(local.addr);
    link_.establish(local, peer);
}

void CpuControllerSession::start() {
    stop(); // drain any leftovers + reset (idempotent)
    next_send_ = 0;
    next_recv_ = 0;
    rtt_ns_ = 0;
}

void CpuControllerSession::stop() { link_.drain(); }

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
    return frame::data(ring_, link_.send_frame());
}

void CpuControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                           std::uint32_t decoder_id) {
    TP_CHECK(bytes <= in_bytes_, "payload (%zu B) exceeds the %zu B committed for this round",
             static_cast<std::size_t>(bytes), static_cast<std::size_t>(in_bytes_));
    frame::stage(ring_, link_.send_frame(), decoder_id, src, static_cast<std::size_t>(bytes));
    staged_ = true;
    fn_id_ = decoder_id;
}

int CpuControllerSession::kick(std::uint32_t /*work_item_idx*/) {
    if (!staged_) {
        void *send = link_.send_frame();
        frame::stage(ring_, send, fn_id_, frame::data(ring_, send),
                     static_cast<std::size_t>(in_bytes_));
    }
    staged_ = false;
    kick_ns_ = now_ns();
    const auto seq = static_cast<std::uint32_t>(next_send_ + 1);
    link_.send(seq);
    ++next_send_;
    return 0;
}

void *CpuControllerSession::reply_slot() {
    return frame::data(ring_, ring_buf_ + ring_.slot_offset(next_recv_));
}

int CpuControllerSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                  std::size_t n) {
    // Blocking wait for this round's reply, which the coprocessor writes into slot next_recv_ of
    // the local ring, seq last.
    void *slot = ring_buf_ + ring_.slot_offset(next_recv_);
    const auto seq = static_cast<std::uint32_t>(next_recv_ + 1);
    frame::FrameView r;
    while (!frame::poll(ring_, slot, seq, r)) {
        cpu_relax();
    }
    rtt_ns_ = now_ns() - kick_ns_;
    ++next_recv_;
    if (n > 0 && replies && replies[0] && replies[0] != r.data) {
        // The coprocessor zero-fills the reply's payload area, so all of `cap` is defined.
        const std::size_t cap = replies_bytes ? replies_bytes[0] : out_bytes_;
        TP_CHECK(cap <= frame::data_capacity(ring_),
                 "reply capacity (%zu) exceeds the %zu B a frame of this session carries", cap,
                 frame::data_capacity(ring_));
        std::memcpy(replies[0], r.data, cap);
    }
    return 0;
}

} // namespace catalyst::transport::cpu_verbs
