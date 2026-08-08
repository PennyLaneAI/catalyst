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

class LocalGpuCoprocessorSession : public CoprocessorSession, public LocalCoprocessorEndpoint {
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

    // CPU local peer-memory doorbell: consume the request in local_request_, launch one GPU decode,
    // and write the reply into peer_reply_. Fixed-width GPU launchers currently accept one 8-byte
    // payload and return one 8-byte correction.
    int run_once() override;

  private:
    void ensure_gpu_state();

    /// Process-local rendezvous object used to find the paired controller.
    std::shared_ptr<EndpointPair> pair_;

    /// This coprocessor's advertised request region; the controller writes requests here.
    MemRegion local_request_{};
    /// The controller's advertised reply region; run_once() writes replies here.
    PeerRef peer_reply_{};

    /// Owned allocations returned from alloc_memory(); MemRegion is only a view into these.
    std::vector<std::unique_ptr<std::byte[]>> caller_memory_regions_;

    /// GPU device to run the launcher on.
    int gpu_device_ = 0;
    /// Lazily created HIP runtime wrapper.
    std::unique_ptr<gpu_verbs::GpuRuntime> gpu_;
    /// Host-visible reply ring the GPU writes into.
    gpu_verbs::GpuRuntime::Handoff handoff_{};
    /// Host-mapped request ring shared with the GPU launcher.
    common::PayloadSlot *request_ring_host_ = nullptr;
    /// Device alias of request_ring_host_.
    common::PayloadSlot *request_ring_dev_ = nullptr;

    /// Bound launch-once GPU coprocessor function; nullptr means built-in echo.
    CoprocessorLauncherFn launcher_ = nullptr;
    /// Opaque context passed back to launcher_.
    void *launcher_ctx_ = nullptr;
};

} // namespace catalyst::transport::local_copy
