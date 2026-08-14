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

#include "GpuRuntime.hpp"
#include "MemcpyLink.hpp"
#include "Transport.hpp"

namespace catalyst::transport::common {
struct PayloadSlot;
}

namespace catalyst::transport::memcpy {

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

    // Called synchronously from the paired controller's kick(); launches one GPU decode and
    // writes the reply into `out`. Uses single slots (no ring) since total=1.
    //
    // Expects `in_len == sizeof(common::Payload)` (16, a wire-shaped frame) and
    // `out_cap >= sizeof(int64_t)`; the reply is always `sizeof(int64_t)` bytes.
    // Anything else throws.
    std::size_t process_message(const void *in, std::size_t in_len, void *out, std::size_t out_cap);

  private:
    void ensure_gpu_state();

    std::string pair_key_;
    std::shared_ptr<MemcpyLink> link_;

    /// Owns the buffers backing MemRegions handed out by alloc_memory().
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    int gpu_device_ = 0;
    std::unique_ptr<gpu_verbs::GpuRuntime> gpu_;
    gpu_verbs::GpuRuntime::Handoff handoff_{};
    common::PayloadSlot *request_slot_host_ = nullptr;
    common::PayloadSlot *request_slot_dev_ = nullptr;

    CoprocessorLauncherFn launcher_ = nullptr;
    void *launcher_ctx_ = nullptr;
};

} // namespace catalyst::transport::memcpy
