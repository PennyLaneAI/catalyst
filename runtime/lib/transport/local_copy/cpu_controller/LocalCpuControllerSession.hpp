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
#include <vector>

#include "LocalRegistry.hpp"
#include "Transport.hpp"

namespace catalyst::transport::local_copy {

class LocalCpuControllerSession : public ControllerSession {
  public:
    explicit LocalCpuControllerSession(std::string = {}) {}
    ~LocalCpuControllerSession() override = default;

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

  private:
    /// Process-local rendezvous object used to find the paired coprocessor.
    std::shared_ptr<EndpointPair> pair_;

    /// This controller's advertised reply region; the coprocessor writes replies here.
    MemRegion local_reply_{};
    /// The coprocessor's advertised request region; kick() copies requests here.
    PeerRef peer_request_{};

    /// Owned allocations returned from alloc_memory(); MemRegion is only a view into these.
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    /// Controller-owned request staging buffer filled by data_slot()/write_data_slot().
    std::vector<std::byte> request_staging_;

    /// Committed request size for the next round.
    std::uint64_t in_bytes_ = 0;
    /// Committed reply size expected for the next round.
    std::uint64_t out_bytes_ = 0;
    /// Number of bytes currently staged in request_staging_.
    std::uint64_t staged_bytes_ = 0;
    /// Decoder selector to attach to the next kicked request.
    std::uint32_t decoder_id_ = 0;

    /// Timestamp captured at kick() for round-trip timing.
    std::uint64_t kick_ns_ = 0;
    /// Last measured round-trip time in nanoseconds.
    std::uint64_t rtt_ns_ = 0;
};

} // namespace catalyst::transport::local_copy
