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

// HwhsControllerSession - ControllerSession backed by the hardware-handshake engine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CompletionQueue.hpp"
#include "Context.hpp"
#include "Handshake.hpp"
#include "HwhsMem.hpp"
#include "OobSocket.hpp"
#include "ProtectionDomain.hpp"
#include "QueuePair.hpp"
#include "Transport.hpp"

namespace catalyst::transport::hwhs {

namespace common = catalyst::transport::common;

// A Region is a buffer on the FPGA (PL-DDR / PS-DDR / BRAM)
class Region {
  public:
    Region() = default;
    Region(HwhsMem *umm, std::shared_ptr<common::Context> ctx) : umm_(umm), ctx_(std::move(ctx)) {}
    ~Region();
    Region(Region &&other) noexcept;
    Region &operator=(Region &&other) noexcept;
    Region(const Region &) = delete;
    Region &operator=(const Region &) = delete;

    int chunk = -1;
    int mr_chunk = -1;
    std::uint64_t va = 0;
    std::uint64_t adv_va = 0;
    std::uint64_t pa = 0;
    volatile std::uint8_t *view = nullptr; // /dev/mem window onto pa
    void *map_base = nullptr;              // mmap base
    std::size_t map_span = 0;              // mmap length
    std::uint64_t size = 0;
    ibv_mr *mr = nullptr;

  private:
    void release() noexcept;
    HwhsMem *umm_ = nullptr;
    std::shared_ptr<common::Context> ctx_;
};

// Demo mode settings
struct DemoCfg {
    bool enable = false;
    std::uint64_t freq_num = 0;    // core-clock cycles between sends (200 MHz: 5 ms = 1'000'000)
    std::uint32_t cmd_cnt = 1;     // syndromes per run
    std::uint32_t syn_depth = 256; // syndrome table size in bytes (multiple of 64)
    std::uint32_t freq_span = 0;   // 0 = fixed interval
    std::uint32_t lfsr_seed = 0;   // 0 = engine default
    std::uint64_t bram_pa = 0x8000'0000ull; // BRAM base address
    std::uint64_t trace_pa = 0;             // trace memory base address; required when enable
    std::string trace_out;                  // where to write cycles,ns; empty = do not dump
    std::string table_path;                 // path to the BRAM table image
};

class HwhsControllerSession final : public ControllerSession {
  public:
    HwhsControllerSession(std::string dev, int gid_idx, std::uint32_t ring_slots,
                          std::uint32_t stride_log2,
                          std::optional<MemKind> data_kind = std::nullopt,
                          std::optional<MemKind> sq_kind = std::nullopt,
                          std::optional<MemKind> reply_kind = std::nullopt, bool sw_poll = true,
                          bool kick_ioctl = false, DemoCfg demo = {});
    ~HwhsControllerSession() override;

    HwhsControllerSession(const HwhsControllerSession &) = delete;
    HwhsControllerSession &operator=(const HwhsControllerSession &) = delete;

    // TransportSession
    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override {}
    int collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n) override;
    void stop() override;
    std::uint64_t last_rtt_ns() const override { return last_rtt_ns_; }

    // ControllerSession
    void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                          std::uint64_t out_bytes) override;
    void *data_slot() override;
    void write_data_slot(const void *src, std::uint64_t bytes, std::uint32_t decoder_id) override;
    void *reply_slot() override;
    int kick(std::uint32_t work_item_idx = 0) override;

  private:
    Region region_alloc(std::uint64_t size, int mem_type, int access);

    // Runtime-loaded UMM allocator
    HwhsMem umm_;

    // Config
    std::string dev_;
    int gid_idx_ = 1;
    std::uint32_t ring_slots_ = 8;
    std::uint32_t stride_log2_ = 6;
    std::uint32_t stride_ = 64;

    // Work item bookkeeping (WQE)
    static constexpr std::uint32_t kNumWorkItems = 16;
    std::uint32_t correction_bytes_[kNumWorkItems] = {};
    bool committed_[kNumWorkItems] = {};
    std::uint32_t active_item_ = ~0u; // invalid until first kick()
    std::optional<MemKind> data_kind_;
    std::optional<MemKind> sq_kind_;
    std::optional<MemKind> reply_kind_;
    bool sw_poll_ = true;

    // Engine device nodes (hwhs-specific).
    int hh_fd_ = -1;
    int mem_fd_ = -1;

    // Userspaced control registers (mapped from /dev/xib0)
    void *hh_reg_map_ = nullptr;
    volatile std::uint32_t *hh_doorbell_ = nullptr;

    // Currently, we only support memory-mapped kick() when kick_ioctl is set to false
    bool kick_ioctl_ = false;

    // Demo mode settings
    DemoCfg demo_;
    volatile std::uint32_t *hh_demo_doorbell_ = nullptr; // DEMO_CTRL in the mmap'd window
    void demo_program();                                 // push DemoCfg into the engine
    void demo_preload();                                 // fill the BRAM syndrome table
    void demo_report() const;                            // print the demo mode results
    void demo_dump_trace() const;                        // dump the trace memory
    void demo_diff_dump() const;           // dump the diff between the expected and actual results
    void demo_arm();                       // arm the demo mode
    void demo_dump(const char *why) const; // dump the demo mode results

    // RDMA setup reused from devices/common.
    common::FdGuard oob_;
    std::shared_ptr<common::Context> ctx_;
    std::shared_ptr<common::ProtectionDomain> pd_;
    std::shared_ptr<common::CompletionQueue> fwd_cq_, bwd_cq_;
    std::shared_ptr<common::QueuePair> fwd_qp_, bwd_qp_;
    std::uint8_t port_num_ = 1;
    std::uint32_t active_mtu_ = 0;
    std::uint8_t local_gid_[16] = {0};

    // Data-plane regions
    Region reply_;
    Region out_;
    Region sq_ring_;

    // Regions handed out by alloc_memory(), kept alive so their MRs stay valid.
    std::vector<Region> caller_regions_;

    // Peer
    std::uint32_t peer_rkey_ = 0;
    std::uint64_t peer_addr_ = 0;
    common::QpInfo peer_fwd_ = {};
    common::QpInfo peer_bwd_ = {};
    std::uint32_t peer_mtu_ = 0;

    // Round bookkeeping
    bool armed_ = false;
    bool stopped_ = false;
    std::uint64_t submitted_ = 0;   // rounds kicked
    std::uint64_t collected_ = 0;   // rounds drained
    std::uint64_t kick_ns_ = 0;     // host clock at doorbell (sw-poll RTT)
    std::uint64_t last_rtt_ns_ = 0; // RTT reported to the app

    // Demo mode bookkeeping
    std::uint32_t last_round_cnt_ = 0;
    std::uint64_t rounds_seen_ = 0;    // rounds the engine actually ran while we watched
    bool demo_armed_ = false;          // the pacer level has been raised for this run
    bool demo_finished_ = false;       // the engine hit cmd_cnt; no more rounds are coming
    std::uint32_t err_cnt_base_ = 0;   // since power-on error count

    // Report the collected per-round RTTs (min/percentiles/max/mean) on teardown.
    void report_rtt() const;
    std::vector<std::uint64_t> rtt_samples_;
};

} // namespace catalyst::transport::hwhs
