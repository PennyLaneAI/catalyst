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

#include "CompletionQueue.hpp"
#include "Context.hpp"
#include "GpuRuntime.hpp"
#include "MemoryRegion.hpp"
#include "OobSocket.hpp"
#include "ProtectionDomain.hpp"
#include "QueuePair.hpp"
#include "Transport.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::gpu_verbs {
using namespace catalyst::transport;
using namespace catalyst::transport::coproc; // GpuRuntime, HandoffSlot, launchers

class GpuCoprocessorSession : public CoprocessorSession {
  public:
    GpuCoprocessorSession(std::string dev, int gid_idx, int gpu_device);
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
    void run_coprocessor(std::stop_token st);
    // Fill the reply + post one inline RDMA_WRITE. Returns true if this send
    // was signaled (selective signalling: 1 in SIGNAL_EVERY), so the caller can
    // track the signaled-completion count.
    bool post_inline(std::uint64_t cursor);
    // Non-blocking batch reap of the bwd CQ; decrements `outstanding` per CQE.
    // With drain=true, keeps polling until `outstanding` hits 0 (teardown).
    void reap_bwd(int &outstanding, bool drain);

    std::string dev_name_;
    int gid_idx_;
    ibv_gid mygid_{};              // local GID, cached in connect() for exchange_keys()
    std::uint32_t active_mtu_ = 0; // local active_mtu enum, cached in connect()
    std::shared_ptr<common::Context> ctx_;
    std::shared_ptr<common::ProtectionDomain> pd_;
    std::shared_ptr<common::CompletionQueue> fwd_cq_, bwd_cq_;
    std::shared_ptr<common::QueuePair> fwd_qp_, bwd_qp_;
    common::FdGuard oob_fd_;

    GpuRuntime gpu_;                // HIP wrapper: owns GPU mem/stream/sync; the decode
                                    // kernel itself is launched by the bound CoprocessorFn
    GpuRuntime::HbmRing hbm_{};     // the HBM receive-ring buffer itself (device
                                    // ptr + size + dma-buf fd)
    GpuRuntime::Handoff handoff_{}; // host-mapped GPU->CPU ring: kernel writes
                                    // {correction,seq}, CPU reads
    std::optional<common::MemoryRegion>
        hbm_ring_; // ibverbs dma-buf MR registering hbm_ (gives the rkey)
    std::optional<common::MemoryRegion> reply_buf_; // host RAM inline reply source

    MemRegion local_{};
    PeerRef peer_{};
    ChannelDesc desc_{};
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
