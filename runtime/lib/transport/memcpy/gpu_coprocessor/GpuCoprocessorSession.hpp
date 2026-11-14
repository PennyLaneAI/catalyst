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

#include "Frame.hpp"
#include "GpuRuntime.hpp"
#include "MemcpyLink.hpp"
#include "MessageWorker.hpp"
#include "Transport.hpp"

namespace catalyst::transport::memcpy {

// The GPU coprocessor of the in-process memcpy transport, in one of two modes.
//
// Launch-once (the default): process_message, on the controller thread, frames each request in a
// host-mapped request ring, a persistent decode kernel answers it with a reply frame in the
// host-mapped handoff ring, and process_message copies the reply out. Both rings hold codec frames
// in the session's layout, so messages carry whatever that layout does.
//
// Per-message (`per_message` true): a MessageWorker runs a host CoprocessorFn once per message on a
// thread whose current GPU is `gpu_device`, and the function launches whatever GPU work it needs.
// This suits a model compiled ahead of time, which is launched from the host per call rather than
// resident as a persistent kernel. Messages carry any size each way.
class GpuCoprocessorSession : public CoprocessorSession {
  public:
    explicit GpuCoprocessorSession(const std::string &config = {}, int gpu_device = 0,
                                   bool per_message = false);
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
    void set_coprocessor_fn(CoprocessorFn fn, void *ctx) override;
    CoprocConvention coprocessor_fn_convention() const override {
        return per_message_ ? CoprocConvention::PerMessage : CoprocConvention::LaunchOnce;
    }

    // Called on the controller thread from kick() with the round's payload, its length and fn_id.
    // Frames the request in the next request slot, spin-waits for the kernel's reply frame in the
    // same handoff slot, and copies `out_cap` bytes of its payload, zero past what the decoder
    // wrote, into `out`. Returns `out_cap`; throws if either size exceeds the session's frames.
    // Per-message hands the message to the MessageWorker instead.
    std::size_t process_message(const void *payload, std::size_t len, std::uint32_t fn_id,
                                void *out, std::size_t out_cap);

  private:
    void ensure_gpu_state();

    std::string pair_key_;
    std::shared_ptr<MemcpyLink> link_;

    /// Owns the buffers backing MemRegions handed out by alloc_memory().
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    int gpu_device_ = 0;
    bool per_message_ = false;
    common::frame::RingGeometry ring_; // both rings' layout, from the config
    MessageWorker worker_;
    std::unique_ptr<coproc::GpuRuntime> gpu_;
    coproc::GpuRuntime::Handoff handoff_{};
    // Host-mapped request ring polled by the persistent decode kernel via `ring_dev_`.
    std::uint8_t *ring_host_ = nullptr;
    void *ring_dev_ = nullptr;
    // Monotonic cursor advanced by process_message (single-writer on controller thread).
    std::uint64_t process_cursor_ = 0;
    bool kernel_running_ = false;

    CoprocessorLauncherFn launcher_ = nullptr;
    void *launcher_ctx_ = nullptr;
};

} // namespace catalyst::transport::memcpy
