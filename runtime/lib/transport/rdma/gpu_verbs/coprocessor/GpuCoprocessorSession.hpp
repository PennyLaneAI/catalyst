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
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

#include "Frame.hpp"
#include "GpuRuntime.hpp"
#include "MemoryRegion.hpp"
#include "QueuePair.hpp"
#include "Transport.hpp"
#include "WireLink.hpp"

namespace catalyst::transport::gpu_verbs {
using namespace catalyst::transport;
using namespace catalyst::transport::coproc; // GpuRuntime, launchers

// The controller writes each request into a ring in GPU memory, a persistent kernel decodes it
// and writes the reply frame into a ring the host maps, and a pinned CPU thread sends the reply.
// The link speaks the wire to the controller (WireLink.hpp).
class GpuCoprocessorSession final : public CoprocessorSession {
  public:
    // The coprocessor only receives on an RC link's fwd queue pair (the controller writes into its
    // ring), so that queue pair needs almost no send queue and no inline capacity.
    static constexpr common::QpShape FWD_QP_SHAPE{.sq_depth = 16, .cq_depth = 256, .inline_max = 0};

    /// On the RC wire, with `ring` the frames both ends agree on.
    GpuCoprocessorSession(std::string dev, int gid_idx, int gpu_device,
                          common::frame::RingGeometry ring = common::frame::BL16_RING);
    /// On the wire `link` speaks (common::make_coprocessor_link, with FWD_QP_SHAPE).
    GpuCoprocessorSession(int gpu_device, std::unique_ptr<common::WireLink> link);
    ~GpuCoprocessorSession() override { stop(); }

    GpuCoprocessorSession(const GpuCoprocessorSession &) = delete;
    GpuCoprocessorSession &operator=(const GpuCoprocessorSession &) = delete;
    GpuCoprocessorSession(GpuCoprocessorSession &&) = delete;
    GpuCoprocessorSession &operator=(GpuCoprocessorSession &&) = delete;

    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override;
    int collect(void *const *outputs, const std::uint64_t *output_bytes, std::size_t n) override;
    void stop() override;
    void set_coprocessor_launcher(CoprocessorLauncherFn fn, void *ctx) override;

    void set_thread_affinity(int cpu, bool realtime) {
        pin_cpu_ = cpu;
        pin_realtime_ = realtime;
    }
    CoprocConvention coprocessor_fn_convention() const override {
        return CoprocConvention::LaunchOnce;
    }

    MemKind preferred_mem_kind() const override { return MemKind::GpuHbm; }

  private:
    // The reply loop; runs on engine_.
    void run(std::stop_token st);
    template <common::frame::Mode M, class Link> void serve(Link &link, std::stop_token st);

    std::unique_ptr<common::WireLink> link_;
    common::frame::RingGeometry ring_; // both rings, the request ring and the handoff
    GpuRuntime gpu_;                   // HIP wrapper: owns GPU mem/stream/sync; the decode
                                       // kernel itself is launched by the bound CoprocessorFn
    GpuRuntime::HbmRing hbm_{};        // the HBM receive-ring buffer itself (device
                                       // ptr + size + dma-buf fd)
    GpuRuntime::Handoff handoff_{};    // host-mapped GPU->CPU ring of reply frames
    std::optional<common::MemoryRegion>
        hbm_ring_; // ibverbs dma-buf MR registering hbm_ (gives the rkey)

    // Launcher used by start() to launch the on-device decode kernel (one-shot,
    // not per-message); nullptr selects the built-in echo launcher.
    CoprocessorLauncherFn coproc_launcher_ = nullptr;
    void *coproc_ctx_ = nullptr;

    // failed_ (release) publishes error_; collect() acquire-loads failed_ and
    // rethrows.
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    int pin_cpu_ = -1; // -1 -> leave affinity alone
    bool pin_realtime_ = false;
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<std::int64_t> last_word_{0};
    std::jthread engine_;
};

} // namespace catalyst::transport::gpu_verbs
