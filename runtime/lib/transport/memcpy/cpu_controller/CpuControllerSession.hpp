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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "MemcpyLink.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::memcpy {

class CpuControllerSession : public ControllerSession {
  public:
    explicit CpuControllerSession(const std::string &config = {})
        : pair_key_(parse_pair_key(config)) {}
    ~CpuControllerSession() override;

    // TransportSession
    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override;
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override;
    void stop() override;
    std::uint64_t last_rtt_ns() const override { return rtt_ns_; }

    // ControllerSession
    void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                          std::uint64_t out_bytes) override;
    int kick(std::uint32_t work_item_idx = 0) override;
    void *data_slot() override;
    void write_data_slot(const void *src, std::uint64_t bytes, std::uint32_t decoder_id) override;
    void *reply_slot() override { return local_reply_.addr; }

  private:
    std::string pair_key_;
    std::shared_ptr<MemcpyLink> link_;

    /// Reply buffer the paired coprocessor writes into during kick().
    MemRegion local_reply_{};

    /// Owns the buffers backing MemRegions handed out by alloc_memory().
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    std::vector<std::byte> request_staging_;

    /// Set on the first commit_work_item(). Subsequent commits are rejected so any pointer a
    /// prior data_slot() handed out cannot dangle behind a request_staging_ reallocation.
    bool committed_ = false;

    std::uint64_t in_bytes_ = sizeof(common::Payload::value);
    std::uint64_t out_bytes_ = sizeof(common::Payload::value);
    std::uint64_t staged_bytes_ = 0;
    std::uint64_t reply_bytes_ = 0;
    std::uint32_t decoder_id_ = 0;

    std::uint64_t kick_ns_ = 0;
    std::uint64_t rtt_ns_ = 0;

    // Per-session round counter driving `Payload::seq_num`. Matches cpu_verbs's `next_send_`:
    // the first kick sends seq_num=1, the second sends 2, etc. Reset on start().
    std::uint64_t next_send_ = 0;
};

} // namespace catalyst::transport::memcpy
