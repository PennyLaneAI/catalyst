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

#include "CpuSessionBase.hpp"

namespace catalyst::transport::cpu_verbs {

// Controller role: caller-driven. The caller commits a work item,
// writes the outbound payload into data_slot(), kick()s one round,
// then collect()s the reply.
class CpuControllerSession : public CpuSessionBase<ControllerSession> {
    using Base = CpuSessionBase<ControllerSession>;

  public:
    explicit CpuControllerSession(std::string dev, int gid_idx) : Base(std::move(dev), gid_idx) {}
    // Drains the fwd CQ before the base's verbs objects go away. stop() is
    // idempotent, so calling it here and from start() is harmless.
    ~CpuControllerSession() override { stop(); }

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

  protected:
    bool oob_listens() const override { return false; }

  private:
    // Sizes from commit_work_item are fixed in CPU controller.
    std::uint64_t in_bytes_ = sizeof(common::Payload::value);
    std::uint64_t out_bytes_ = sizeof(common::Payload::value);
    std::uint64_t next_send_ = 0, next_recv_ = 0;
    int signaled_outstanding_ = 0;
    std::uint64_t kick_ns_ = 0, rtt_ns_ = 0;
};

} // namespace catalyst::transport::cpu_verbs
