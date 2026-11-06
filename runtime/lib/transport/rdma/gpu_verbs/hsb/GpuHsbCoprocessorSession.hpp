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

/**
 * @file
 * The GPU coprocessor on the HSB wire (HsbWire.hpp), selected by `wire=hsb` in the gpu_verbs
 * coprocessor's config. The data path is GpuCoprocessorSession's: HSB writes each request into a
 * ring in GPU memory, a persistent kernel decodes it and writes the reply frame into a ring the
 * host maps, and a pinned CPU thread posts the reply. Only the wire differs: a UC QP, 128 B slots
 * (CoprocLaunchDesc::slot_stride), and the reply as a SEND.
 */

#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

#include "Frame.hpp"
#include "GpuRuntime.hpp"
#include "HsbLink.hpp"
#include "MemoryRegion.hpp"
#include "Transport.hpp"

namespace catalyst::transport::gpu_verbs {
using namespace catalyst::transport::coproc; // GpuRuntime, launchers

class GpuHsbCoprocessorSession final : public CoprocessorSession {
  public:
    GpuHsbCoprocessorSession(std::string dev, int gid_idx, int gpu_device,
                             common::frame::RingGeometry ring)
        : link_(std::move(dev), gid_idx), ring_(ring), gpu_(gpu_device) {}
    ~GpuHsbCoprocessorSession() override { stop(); }

    GpuHsbCoprocessorSession(const GpuHsbCoprocessorSession &) = delete;
    GpuHsbCoprocessorSession &operator=(const GpuHsbCoprocessorSession &) = delete;

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
    void run(std::stop_token st);

    common::HsbLink link_;
    common::frame::RingGeometry ring_; // both rings, the request ring and the handoff
    GpuRuntime gpu_;
    GpuRuntime::HbmRing hbm_{};
    GpuRuntime::Handoff handoff_{};
    std::optional<common::MemoryRegion> ring_mr_; // the HBM ring, at IOVA 0, through dma-buf

    CoprocessorLauncherFn coproc_launcher_ = nullptr; // nullptr -> built-in echo launcher
    void *coproc_ctx_ = nullptr;

    // failed_ (release) publishes error_; collect() acquire-loads it and rethrows.
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    int pin_cpu_ = -1; // -1 -> leave affinity alone
    bool pin_realtime_ = false;
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<std::int64_t> last_word_{0};
    std::jthread engine_;
};

} // namespace catalyst::transport::gpu_verbs
