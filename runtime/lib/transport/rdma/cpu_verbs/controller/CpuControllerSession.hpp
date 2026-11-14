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

#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "Frame.hpp"
#include "MemoryRegion.hpp"
#include "RcLink.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::cpu_verbs {

// Controller role: caller-driven. The caller commits a work item,
// writes the outbound payload into data_slot(), kick()s one round,
// then collect()s the reply.
class CpuControllerSession final : public ControllerSession {
  public:
    explicit CpuControllerSession(std::string dev, int gid_idx,
                                  common::frame::RingGeometry ring = common::frame::BL16_RING);
    // Drains the fwd CQ before the link's verbs objects go away. stop() is
    // idempotent, so calling it here and from start() is harmless.
    ~CpuControllerSession() override { stop(); }

    CpuControllerSession(const CpuControllerSession &) = delete;
    CpuControllerSession &operator=(const CpuControllerSession &) = delete;

    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override;
    void stop() override;
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override;
    std::uint64_t last_rtt_ns() const override { return rtt_ns_; }

    // ControllerSession interface.
    void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                          std::uint64_t out_bytes) override;
    int kick(std::uint32_t work_item_idx = 0) override;
    void *data_slot() override;
    void write_data_slot(const void *src, std::uint64_t bytes, std::uint32_t decoder_id) override;
    void *reply_slot() override;

  private:
    common::RcLink link_;
    common::frame::RingGeometry ring_;
    std::vector<common::MemoryRegion> regions_; // handed out by alloc_memory()
    std::uint8_t *ring_buf_ = nullptr; // the ring replies land in, from establish_channel()

    // Sizes from commit_work_item are fixed in CPU controller.
    std::uint64_t in_bytes_ = sizeof(common::Payload::value);
    std::uint64_t out_bytes_ = sizeof(common::Payload::value);
    std::uint64_t next_send_ = 0, next_recv_ = 0;
    // Whether write_data_slot() staged this round's frame; if not, the caller wrote all in_bytes_
    // in place through data_slot(), and kick() stages it under the last fn_id.
    bool staged_ = false;
    std::uint32_t fn_id_ = 0;
    std::uint64_t kick_ns_ = 0, rtt_ns_ = 0;
};

} // namespace catalyst::transport::cpu_verbs
