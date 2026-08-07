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

#include "Transport.hpp"
#include "../LocalRegistry.hpp"

namespace catalyst::transport::local_copy {

class LocalCoprocessorSession : public CoprocessorSession {
  public:
    explicit LocalCoprocessorSession(std::string = {}) {}
    ~LocalCoprocessorSession() override = default;

    // TransportSession
    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override;
    int collect(void *const *replies, const std::uint64_t *replies_bytes,
                std::size_t n) override;
    void stop() override;

    // CoprocessorSession
    void set_coprocessor_fn(CoprocessorFn fn, void *ctx) override;

    // CPU local peer-memory doorbell: consume the request in local_request_ and write the reply
    // into peer_reply_.
    int run_once();

  private:
    /// Process-local rendezvous object used to find the paired controller.
    std::shared_ptr<EndpointPair> pair_;

    /// This coprocessor's advertised request region; the controller writes requests here.
    MemRegion local_request_{};
    /// The controller's advertised reply region; run_once() writes replies here.
    PeerRef peer_reply_{};

    /// Owned allocations returned from alloc_memory(); MemRegion is only a view into these.
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    /// Bound per-message coprocessor function; nullptr means not bound yet.
    CoprocessorFn fn_ = nullptr;
    /// Opaque context passed back to fn_.
    void *ctx_ = nullptr;
};

} // namespace catalyst::transport::local_copy
