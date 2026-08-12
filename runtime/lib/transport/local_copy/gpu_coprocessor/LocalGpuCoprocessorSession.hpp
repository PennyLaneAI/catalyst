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

#include "GpuRuntime.hpp"
#include "LocalRegistry.hpp"
#include "Transport.hpp"

namespace catalyst::transport::common {
struct PayloadSlot;
}

namespace catalyst::transport::local_copy {

class LocalGpuCoprocessorSession : public CoprocessorSession {
  public:
    explicit LocalGpuCoprocessorSession(std::string = {}, int gpu_device = 0);
    ~LocalGpuCoprocessorSession() override;

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

    // Consume one request, launch one GPU decode, and write the reply into `reply`.
    // Invoked synchronously from the paired controller's kick(). Uses a single request slot and a
    // single handoff slot; total=1 never needs a ring.
    std::size_t run_once(const void *req, std::size_t req_bytes, std::uint32_t decoder_id,
                         void *reply, std::size_t reply_cap);

  private:
    void ensure_gpu_state();

    /// Process-local rendezvous with the paired controller.
    std::shared_ptr<EndpointPair> pair_;

    /// Owned allocations returned from alloc_memory(); MemRegion is only a view into these.
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    int gpu_device_ = 0;
    std::unique_ptr<gpu_verbs::GpuRuntime> gpu_;
    gpu_verbs::GpuRuntime::Handoff handoff_{};
    common::PayloadSlot *request_slot_host_ = nullptr;
    common::PayloadSlot *request_slot_dev_ = nullptr;

    CoprocessorLauncherFn launcher_ = nullptr;
    void *launcher_ctx_ = nullptr;
};

} // namespace catalyst::transport::local_copy
