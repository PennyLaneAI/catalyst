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

#include "GpuRuntime.hpp"
#include "MemcpyLink.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::memcpy {

// Two roles in one process: the controller writes into the request ring, a persistent decode
// kernel drains it and publishes into the handoff ring, and an engine jthread reads the
// handoff and republishes into a reply ring the controller consumes. process_message runs on
// the controller thread; the engine thread bridges kernel completions to reply publication.
class GpuCoprocessorSession : public CoprocessorSession {
  public:
    explicit GpuCoprocessorSession(const std::string &config = {}, int gpu_device = 0);
    ~GpuCoprocessorSession() override;

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
    void set_coprocessor_launcher(CoprocessorLauncherFn fn, void *ctx) override;
    CoprocConvention coprocessor_fn_convention() const override {
        return CoprocConvention::LaunchOnce;
    }

    // Called on the controller thread from kick(). Publishes the request into the next request
    // slot, spin-waits for the engine thread to publish the paired reply slot (fed by the
    // persistent decode kernel's handoff), and copies the reply into `out`.
    //
    // Expects `in_len == sizeof(common::Payload)` (16, a wire-shaped frame) and
    // `out_cap >= sizeof(int64_t)`; the reply is always `sizeof(int64_t)` bytes.
    // Anything else throws.
    int process_message(const void *in, std::size_t in_len, void *out, std::size_t out_cap);

  private:
    void ensure_gpu_state();
    // Engine loop: handoff -> reply_ring.
    void run(std::stop_token st);

    std::string pair_key_;
    std::shared_ptr<MemcpyLink> link_;

    /// Owns the buffers backing MemRegions handed out by alloc_memory().
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    int gpu_device_ = 0;
    std::unique_ptr<coproc::GpuRuntime> gpu_;
    coproc::GpuRuntime::Handoff handoff_{};
    // Host-mapped request ring polled by the persistent decode kernel via `ring_dev_`.
    common::PayloadSlot *ring_host_ = nullptr;
    common::PayloadSlot *ring_dev_ = nullptr;
    // Reply ring: engine thread writes here after observing a kernel handoff, controller thread
    // reads here in process_message.
    std::array<common::PayloadSlot, common::K_RING_SLOTS> reply_ring_{};
    // Monotonic cursor advanced by process_message (single-writer on controller thread).
    std::uint64_t process_cursor_ = 0;
    bool kernel_running_ = false;

    // Engine error plumbing: if run() throws, error_ is set and failed_ published (release);
    // process_message acquire-loads failed_ and rethrows.
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    std::jthread engine_;

    CoprocessorLauncherFn launcher_ = nullptr;
    void *launcher_ctx_ = nullptr;
};

} // namespace catalyst::transport::memcpy
