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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "CompletionQueue.hpp"
#include "Context.hpp"
#include "Handshake.hpp"
#include "OobSocket.hpp"
#include "ProtectionDomain.hpp"
#include "QueuePair.hpp"
#include "Transport.hpp"
#include "UmmLib.hpp"
#include "WireProtocol.hpp"

namespace rdma::devices::fpga_verbs {
using namespace catalyst::transport;
namespace common = catalyst::transport::common;

inline constexpr std::size_t GID_BYTES = 16; // RoCE GID width

// Region backed by the on-board allocator. `view` is the CPU-accessible device
// VA. The WQE/SGE lkey is the allocator chunk id, not an ibv_reg_mr lkey; `mr`
// is set only when the region is registered for the peer's remote writes.
struct Region {
    int chunk = -1;
    std::uint64_t va = 0;
    volatile std::uint8_t *view = nullptr;
    std::uint64_t size = 0;
    ibv_mr *mr = nullptr;
};

// FPGA controller with ibverbs software datapath
class FpgaControllerSession : public ControllerSession {
  public:
    FpgaControllerSession(
        std::string dev = "xib_0", int gid_idx = 1,
        std::uint32_t ring_slots = static_cast<std::uint32_t>(common::K_RING_SLOTS),
        std::uint32_t stride_log2 = 6, std::optional<MemKind> data_kind = std::nullopt,
        std::optional<MemKind> reply_kind = std::nullopt);
    ~FpgaControllerSession() override { stop(); }

    FpgaControllerSession(const FpgaControllerSession &) = delete;
    FpgaControllerSession &operator=(const FpgaControllerSession &) = delete;

    int connect(const ConnectInfo &info) override;
    MemRegion alloc_memory(std::size_t size, MemKind kind) override;
    PeerRef exchange_keys(const MemRegion &local) override;
    void establish_channel(const ChannelDesc &desc, const MemRegion &local,
                           const PeerRef &peer) override;
    void start() override;
    int collect(void *const *outputs, const std::size_t *output_bytes, std::size_t n) override;
    void stop() override;
    std::uint64_t last_rtt_ns() const override { return last_rtt_ns_; }

    void commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                          std::uint64_t out_bytes) override;
    int kick(std::uint32_t work_item_idx) override;
    void *data_slot() override;
    void write_data_slot(const void *src, std::uint64_t bytes, std::uint32_t decoder_id) override;
    void *reply_slot() override;

    void set_cpu_pin(int cpu) { cpu_pin_ = cpu; }

  private:
    // Allocates a 64-B-aligned chunk (NIC DMA requirement) and zeroes it.
    Region region_alloc(std::uint64_t size, int mem_type, int access);
    void region_free(Region &r);

    void reap(ibv_cq *cq, int &outstanding, bool drain);
    void post_write(ibv_qp *qp, std::uint32_t work_item_idx, std::uint64_t cursor, bool signaled);
    volatile common::PayloadSlot *poll_message_arrival(std::uint64_t cursor);

    UmmLib umm_;

    std::string dev_;
    int gid_idx_ = 1;
    std::uint32_t ring_slots_ = static_cast<std::uint32_t>(common::K_RING_SLOTS);
    std::uint32_t stride_log2_ = 6;
    std::uint32_t stride_ = 64;
    std::uint32_t pending_decoder_ = 0;
    MemKind mem_kind_ = MemKind::CpuRam;
    std::optional<MemKind> data_kind_;

    std::shared_ptr<common::Context> ctx_;
    std::shared_ptr<common::ProtectionDomain> pd_;
    std::shared_ptr<common::CompletionQueue> fwd_cq_, bwd_cq_;
    std::shared_ptr<common::QueuePair> fwd_qp_, bwd_qp_;
    common::FdGuard oob_;
    std::uint8_t port_num_ = 1;
    std::uint32_t active_mtu_ = 0;
    std::uint8_t local_gid_[GID_BYTES] = {0};

    Region reply_; // inbound (peer RDMA_WRITEs here)
    Region out_;   // outbound request ring

    std::uint32_t peer_rkey_ = 0;
    std::uint64_t peer_addr_ = 0;
    common::QpInfo peer_fwd_ = {};
    common::QpInfo peer_bwd_ = {};
    std::uint32_t peer_mtu_ = 0;

    static constexpr std::uint32_t K_NUM_WORK_ITEMS = 16;
    std::uint32_t correction_bytes_[K_NUM_WORK_ITEMS] = {};
    bool committed_[K_NUM_WORK_ITEMS] = {};
    std::uint32_t active_item_ = 0;
    ibv_send_wr wr_[K_NUM_WORK_ITEMS] = {};
    ibv_sge sge_[K_NUM_WORK_ITEMS] = {};

    bool armed_ = false;
    bool stopped_ = false;
    std::uint32_t max_in_flight_ = 1;
    std::uint64_t submitted_ = 0;
    std::uint64_t collected_ = 0;
    int signaled_outstanding_ = 0;
    int cpu_pin_ = -1;

    std::uint64_t kick_t0_ns_ = 0;
    std::uint64_t last_rtt_ns_ = 0;
};

} // namespace rdma::devices::fpga_verbs
