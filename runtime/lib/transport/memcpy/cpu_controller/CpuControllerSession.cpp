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

#include "CpuControllerSession.hpp"

#include <chrono>
#include <cstring>

#include "Error.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::memcpy {
namespace {

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

} // namespace

CpuControllerSession::~CpuControllerSession() {
    if (link_) {
        std::lock_guard<std::mutex> lock(link_->mu);
        if (link_->controller == this) {
            link_->controller = nullptr;
        }
    }
}

int CpuControllerSession::connect(const ConnectInfo & /*info*/) {
    link_ = acquire_memcpy_link(pair_key_);
    std::lock_guard<std::mutex> lock(link_->mu);
    TP_CHECK(!link_->controller, "Controller already bound to pair '%s'", pair_key_.c_str());
    link_->controller = this;
    return 0;
}

MemRegion CpuControllerSession::alloc_memory(std::size_t size, MemKind kind) {
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

PeerRef CpuControllerSession::exchange_keys(const MemRegion &local) {
    local_reply_ = local;
    return PeerRef{};
}

void CpuControllerSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                             const PeerRef & /*peer*/) {
    TP_CHECK(desc.transport == "memcpy", "Only transport=memcpy is supported");
    local_reply_ = local;
}

void CpuControllerSession::start() {
    rtt_ns_ = 0;
    next_send_ = 0;
}

int CpuControllerSession::collect(void *const *replies, const std::uint64_t *replies_bytes,
                                  std::size_t n) {
    TP_CHECK(n <= 1, "Only one reply slot supported (n<=1)");
    TP_CHECK(local_reply_.addr, "Reply region not established");

    const std::uint64_t cap = replies_bytes ? replies_bytes[0] : out_bytes_;
    TP_CHECK(reply_bytes_ <= cap, "Caller reply buffer too small");
    if (n > 0 && replies && replies[0] && reply_bytes_ != 0) {
        std::memcpy(replies[0], local_reply_.addr, static_cast<std::size_t>(reply_bytes_));
    }

    rtt_ns_ = now_ns() - kick_ns_;
    return 0;
}

void CpuControllerSession::stop() {}

void CpuControllerSession::commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                            std::uint64_t out_bytes) {
    TP_CHECK(work_item_idx == 0, "Only work_item_idx=0 supported");
    // The wire carries an 8-byte payload; anything larger would be silently truncated in kick().
    // Match the RDMA backend's contract (rdma/cpu_verbs/.../CpuControllerSession.cpp) rather than
    // accepting and dropping bytes.
    TP_CHECK(in_bytes <= common::PAYLOAD_DATA_BYTES && out_bytes <= common::PAYLOAD_DATA_BYTES,
             "In/out_bytes exceeds 8 B payload area");
    TP_CHECK(!committed_, "Only one commit_work_item per session");
    in_bytes_ = in_bytes;
    out_bytes_ = out_bytes;
    staged_bytes_ = 0;
    request_staging_.resize(static_cast<std::size_t>(in_bytes_));
    committed_ = true;
}

int CpuControllerSession::kick(std::uint32_t work_item_idx) {
    TP_CHECK(work_item_idx == 0, "Only work_item_idx=0 supported");
    TP_CHECK(link_, "No paired coprocessor");
    TP_CHECK(local_reply_.size >= out_bytes_, "Reply region too small for committed out_bytes");

    kick_ns_ = now_ns();

    // Synthesize a wire-shaped Payload so `in` looks the same as it does over cpu_verbs:
    // value bytes at offset 0, decoder_id at PAYLOAD_DATA_BYTES, then seq_num.
    common::Payload frame{};
    if (staged_bytes_ != 0) {
        std::memcpy(&frame.value, request_staging_.data(),
                    std::min<std::size_t>(static_cast<std::size_t>(staged_bytes_),
                                          common::PAYLOAD_DATA_BYTES));
    }
    frame.decoder_id = decoder_id_;
    frame.seq_num = static_cast<std::uint32_t>(next_send_ + 1);
    ++next_send_;

    std::size_t reply_bytes = 0;
    {
        // Held across check and call so teardown can't clear process_message mid-call.
        std::lock_guard<std::mutex> lock(link_->mu);
        TP_CHECK(link_->process_message, "No paired coprocessor");
        reply_bytes = link_->process_message(&frame, sizeof(frame), local_reply_.addr,
                                             static_cast<std::size_t>(out_bytes_));
    }
    reply_bytes_ = static_cast<std::uint64_t>(reply_bytes);
    return 0;
}

void *CpuControllerSession::data_slot() {
    return request_staging_.empty() ? nullptr : request_staging_.data();
}

void CpuControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                           std::uint32_t decoder_id) {
    TP_CHECK(bytes <= in_bytes_, "Payload exceeds committed in_bytes");
    TP_CHECK(bytes == 0 || src != nullptr, "Null source with non-zero payload");
    if (bytes != 0) {
        std::memcpy(request_staging_.data(), src, static_cast<std::size_t>(bytes));
    }
    staged_bytes_ = bytes;
    decoder_id_ = decoder_id;
}

} // namespace catalyst::transport::memcpy
