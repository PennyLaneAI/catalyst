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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "MemcpyLink.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::memcpy {

// A persistent worker jthread polls a
// request ring, runs the bound CoprocessorFn, and publishes to a reply ring. process_message
// is called inline from the controller's kick(), stages the request into the next request
// slot, spin-waits for the worker to publish the paired reply, and copies it out.
class CpuCoprocessorSession : public CoprocessorSession {
  public:
    explicit CpuCoprocessorSession(const std::string &config = {})
        : pair_key_(parse_pair_key(config)) {}
    ~CpuCoprocessorSession() override;

    // TransportSession
    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override;
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override;
    void stop() override;

    // CoprocessorSession
    void set_coprocessor_fn(CoprocessorFn fn, void *ctx) override;

    // Invoked inline from the controller's kick(). See MemcpyLink::ProcessMessage.
    int process_message(const void *in, std::size_t in_len, void *out, std::size_t out_cap);

  private:
    // The worker loop; runs on engine_.
    void run(std::stop_token st);

    std::string pair_key_;
    std::shared_ptr<MemcpyLink> link_;

    /// Owns the buffers backing MemRegions handed out by alloc_memory().
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    CoprocessorFn fn_ = nullptr;
    void *ctx_ = nullptr;

    // Rings backing the worker pipeline. Both sides index by `cursor & (K_RING_SLOTS - 1)` and
    // publish `seq = cursor + 1` after the payload writes, so the reader only observes a slot
    // once it is fully written.
    std::array<common::PayloadSlot, common::K_RING_SLOTS> request_ring_{};
    std::array<common::PayloadSlot, common::K_RING_SLOTS> reply_ring_{};
    // Monotonic cursor advanced by process_message (single-writer: controller's kick() is
    // synchronous, so no concurrent process_message on the same session).
    std::uint64_t process_cursor_ = 0;

    // Engine error plumbing: mirrors the cpu_verbs pattern. If run() throws, error_ is set and
    // failed_ published (release); process_message acquire-loads failed_ and rethrows.
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    std::jthread engine_;
};

} // namespace catalyst::transport::memcpy
