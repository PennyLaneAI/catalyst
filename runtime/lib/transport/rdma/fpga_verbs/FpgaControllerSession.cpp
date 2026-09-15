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

#include "FpgaControllerSession.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <sched.h>

#ifdef RDMA_VENDOR_IBVERBS
#include "infiniband/verbs.h"
#else
#include <infiniband/verbs.h>
#endif

#include "Error.hpp"

namespace rdma::devices::fpga_verbs {
using namespace catalyst::transport;
using namespace catalyst::transport::common;

namespace {
constexpr std::uint64_t SPIN_CHECK_EVERY = 1024; // CQ sampling interval
constexpr std::uint64_t NIC_DMA_ALIGN = 64;      // NIC DMA alignment
constexpr std::uint32_t PSN_MASK = 0xFFFFFF;     // RoCE 24-bit PSN
constexpr int REAP_BATCH = 16;

// On-board memory-type ids. PsDdr is the tested placement; PlDdr/Bram are
// documented for reference.
enum class XMem : int {
    PlDdr = 1,
    PsDdr = 2,
    Bram = 5,
};

int mem_type_of(MemKind kind) {
    switch (kind) {
    case MemKind::CpuRam:
        return static_cast<int>(XMem::PsDdr);
    case MemKind::Ddr:
        return static_cast<int>(XMem::PlDdr);
    case MemKind::Other:
    case MemKind::GpuHbm:
    default:
        TP_FAIL("FpgaControllerSession: Other/GpuHbm is not a valid FPGA-Verbs MemKind");
    }
}

// libc memset() SIGBUSes on the allocator's memory: glibc/aarch64 zeroes with
// DC ZVA above ~256 B, and cache maintenance faults on this Device mapping.
// Also, the allocator does not zero on alloc, so stale seq_num values could
// masquerade as replies.
void set_zero(volatile std::uint8_t *p, std::uint64_t n) {
    auto *q = reinterpret_cast<volatile std::uint64_t *>(p);
    for (std::uint64_t k = 0; k < n / sizeof(std::uint64_t); ++k) {
        q[k] = 0;
    }
}

void pin_to_cpu(int cpu) {
    if (cpu < 0) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    TP_CHECK(sched_setaffinity(0, sizeof(set), &set) == 0, "sched_setaffinity(cpu=%d) failed", cpu);
}

std::uint64_t mono_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}
} // namespace

FpgaControllerSession::FpgaControllerSession(std::string dev, int gid_idx, std::uint32_t ring_slots,
                                             std::uint32_t stride_log2,
                                             std::optional<MemKind> data_kind,
                                             std::optional<MemKind> reply_kind)
    : dev_(std::move(dev)), gid_idx_(gid_idx), ring_slots_(ring_slots ? ring_slots : 1),
      stride_log2_(stride_log2), stride_((stride_log2 < 32) ? (1u << stride_log2) : 1u),
      mem_kind_(reply_kind.value_or(MemKind::CpuRam)), data_kind_(data_kind) {
    TP_CHECK(stride_log2 < 32, "stride_log2 out of range (must be < 32)");
    TP_CHECK(stride_ >= sizeof(PayloadSlot),
             "stride too small: a ring slot must hold a PayloadSlot (%zu B)", sizeof(PayloadSlot));
}

Region FpgaControllerSession::region_alloc(std::uint64_t size, int mem_type, int access) {
    void *ctx = ctx_->get();
    Region r;
    r.size = size;
    r.chunk = umm_.alloc_chunk(ctx, mem_type, size, size, /*proc=*/true);
    TP_CHECK(r.chunk >= 0, "on-board alloc_chunk(size=%llu): %d",
             static_cast<unsigned long long>(size), r.chunk);
    r.va = umm_.alloc_mem(ctx, r.chunk, size);
    if (r.va == 0) {
        umm_.free_chunk(ctx, r.chunk);
        TP_FAIL("on-board alloc_mem(chunk=%d size=%llu) returned 0", r.chunk,
                static_cast<unsigned long long>(size));
    }
    TP_CHECK((r.va % NIC_DMA_ALIGN) == 0, "buffer not %llu-B aligned: 0x%llx",
             static_cast<unsigned long long>(NIC_DMA_ALIGN), static_cast<unsigned long long>(r.va));
    r.view = reinterpret_cast<volatile std::uint8_t *>(r.va);
    set_zero(r.view, size);
    if (access) {
        r.mr = umm_.reg_mr(pd_->get(), r.va, size, access);
        if (r.mr == nullptr) {
            umm_.free_mem(ctx, static_cast<unsigned>(r.chunk), r.va, size);
            umm_.free_chunk(ctx, r.chunk);
            TP_FAIL("ibv_reg_mr(_ex) failed");
        }
    }
    return r;
}

void FpgaControllerSession::region_free(Region &r) {
    void *ctx = ctx_ ? ctx_->get() : nullptr;
    if (r.mr != nullptr) {
        ibv_dereg_mr(r.mr);
        r.mr = nullptr;
    }
    if (r.chunk >= 0 && ctx != nullptr) {
        if (r.va != 0) {
            umm_.free_mem(ctx, static_cast<unsigned>(r.chunk), r.va, r.size);
        }
        umm_.free_chunk(ctx, r.chunk);
    }
    r = Region{};
}

void FpgaControllerSession::reap(ibv_cq *cq, int &outstanding, bool drain) {
    std::array<ibv_wc, REAP_BATCH> wc{};
    int empty = 0;
    constexpr int DRAIN_MAX_EMPTY = 1000000;
    do {
        int n = ibv_poll_cq(cq, static_cast<int>(wc.size()), wc.data());
        if (n == 0) {
            if (!drain) {
                return;
            }
            if (++empty >= DRAIN_MAX_EMPTY) {
                return;
            }
            continue;
        }
        empty = 0;
        for (int k = 0; k < n; ++k) {
            TP_CHECK(wc[k].status == IBV_WC_SUCCESS, "send CQE status=%d", wc[k].status);
            --outstanding;
        }
    } while (drain && outstanding > 0);
}

int FpgaControllerSession::connect(const ConnectInfo &info) {
    TP_CHECK(umm_.loaded(),
             "on-board memory allocator not available (FPGA controller requires it)");
    ctx_ = std::make_shared<Context>(dev_);
    pd_ = std::make_shared<ProtectionDomain>(ctx_);

    ibv_port_attr pa = ctx_->port_attr(port_num_);
    TP_CHECK(pa.state == IBV_PORT_ACTIVE, "NIC port not ACTIVE (state=%d)", pa.state);
    active_mtu_ = static_cast<std::uint32_t>(pa.active_mtu);
    ibv_gid gid = ctx_->gid(port_num_, gid_idx_);
    std::memcpy(local_gid_, &gid, sizeof(local_gid_));

    fwd_cq_ = std::make_shared<CompletionQueue>(ctx_, 64);
    bwd_cq_ = std::make_shared<CompletionQueue>(ctx_, 64);
    fwd_qp_ = std::make_shared<QueuePair>(pd_, fwd_cq_, fwd_cq_, 16);
    bwd_qp_ = std::make_shared<QueuePair>(pd_, bwd_cq_, bwd_cq_, 16);
    fwd_qp_->to_init(port_num_);
    bwd_qp_->to_init(port_num_);

    oob_ = tcp_connect(info.peer.c_str(), info.oob_port);

    reply_ =
        region_alloc(static_cast<std::uint64_t>(ring_slots_) * stride_, mem_type_of(mem_kind_),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    return 0;
}

MemRegion FpgaControllerSession::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(ctx_ != nullptr, "alloc_memory before connect()");

    constexpr int REPLY_ACCESS = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (reply_.mr == nullptr) {
        reply_ = region_alloc(size, mem_type_of(kind), REPLY_ACCESS);
        mem_kind_ = kind;
    }

    return MemRegion{
        .addr = reinterpret_cast<void *>(reply_.va),
        .size = size,
        .lkey = static_cast<std::uint32_t>(reply_.chunk),
        .rkey = reply_.mr ? reply_.mr->rkey : 0,
        .kind = kind,
    };
}

PeerRef FpgaControllerSession::exchange_keys(const MemRegion &local) {
    TP_CHECK(oob_.valid(), "exchange_keys before connect()");
    TP_CHECK(reply_.mr != nullptr, "exchange_keys: reply region has no MR (alloc remote-write)");

    // Seed each QP's PSN from its QPN masked to 24 bits so both sides agree
    // without a separate exchange.
    HandshakeMsg my{
        .fwd = {.qpn = fwd_qp_->qpn(), .psn = fwd_qp_->qpn() & PSN_MASK},
        .bwd = {.qpn = bwd_qp_->qpn(), .psn = bwd_qp_->qpn() & PSN_MASK},
        .mr_vaddr = local.addr ? reinterpret_cast<std::uint64_t>(local.addr) : reply_.va,
        .mr_rkey = local.rkey ? local.rkey : reply_.mr->rkey,
        .mtu_enum = active_mtu_,
    };
    std::memcpy(my.fwd.gid, local_gid_, sizeof(local_gid_));
    std::memcpy(my.bwd.gid, local_gid_, sizeof(local_gid_));

    // The controller recvs then sends.
    HandshakeMsg peer{};
    recv_exact(oob_.get(), &peer, sizeof(peer));
    send_exact(oob_.get(), &my, sizeof(my));
    oob_.reset();

    peer_rkey_ = peer.mr_rkey;
    peer_addr_ = peer.mr_vaddr;
    peer_fwd_ = peer.fwd;
    peer_bwd_ = peer.bwd;
    peer_mtu_ = peer.mtu_enum;

    return PeerRef{
        .rkey = peer.mr_rkey,
        .remote_addr = peer.mr_vaddr,
        .size = static_cast<std::uint64_t>(ring_slots_) * stride_,
    };
}

void FpgaControllerSession::establish_channel(const ChannelDesc &desc, const MemRegion & /*local*/,
                                              const PeerRef &peer) {
    (void)desc.transport;
    peer_rkey_ = peer.rkey;
    peer_addr_ = peer.remote_addr;

    std::uint32_t mtu = (peer_mtu_ && peer_mtu_ < active_mtu_) ? peer_mtu_ : active_mtu_;
    fwd_qp_->to_rtr(peer_fwd_.qpn, peer_fwd_.psn, peer_fwd_.gid, gid_idx_, port_num_, mtu);
    bwd_qp_->to_rtr(peer_bwd_.qpn, peer_bwd_.psn, peer_bwd_.gid, gid_idx_, port_num_, mtu);
    fwd_qp_->to_rts(fwd_qp_->qpn() & PSN_MASK);
    bwd_qp_->to_rts(bwd_qp_->qpn() & PSN_MASK);

    out_ = region_alloc(static_cast<std::uint64_t>(ring_slots_) * stride_,
                        mem_type_of(data_kind_.value_or(mem_kind_)), /*access=*/0);
    set_zero(reply_.view, reply_.size);
}

void FpgaControllerSession::start() { pin_to_cpu(cpu_pin_); }

void FpgaControllerSession::commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                             std::uint64_t out_bytes) {
    TP_CHECK(out_.va != 0 && reply_.va != 0, "commit_work_item before establish_channel()");
    TP_CHECK(work_item_idx < K_NUM_WORK_ITEMS,
             "commit_work_item: work_item_idx out of range (0..15)");
    TP_CHECK(in_bytes != 0, "commit_work_item: in_bytes is 0");
    TP_CHECK(in_bytes <= stride_, "commit_work_item: in_bytes exceeds ring-slot stride");
    TP_CHECK(out_bytes <= stride_, "commit_work_item: out_bytes exceeds ring-slot stride");

    // Always transfer at least a full Payload so the reader sees a complete
    // framed message (value + seq trailer), even if the decode input is smaller.
    std::uint32_t xfer = (in_bytes < sizeof(Payload)) ? static_cast<std::uint32_t>(sizeof(Payload))
                                                      : static_cast<std::uint32_t>(in_bytes);

    ibv_send_wr &wr = wr_[work_item_idx];
    ibv_sge &sge = sge_[work_item_idx];
    sge = {};
    sge.length = xfer;
    sge.lkey = static_cast<std::uint32_t>(out_.chunk);
    wr = {};
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.wr.rdma.rkey = peer_rkey_;

    correction_bytes_[work_item_idx] = static_cast<std::uint32_t>(out_bytes);
    committed_[work_item_idx] = true;

    if (!armed_) {
        submitted_ = 0;
        collected_ = 0;
    }
    armed_ = true;
}

void *FpgaControllerSession::data_slot() {
    TP_CHECK(out_.va != 0, "data_slot before establish_channel()");
    std::uint32_t slot = static_cast<std::uint32_t>(submitted_ % ring_slots_);
    return reinterpret_cast<void *>(out_.va + static_cast<std::uint64_t>(slot) * stride_);
}

void FpgaControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                            std::uint32_t decoder_id) {
    TP_CHECK(out_.va != 0, "write_data_slot before establish_channel()");
    TP_CHECK(bytes <= stride_, "write_data_slot: bytes exceeds ring-slot capacity");
    pending_decoder_ = decoder_id;
    std::uint32_t slot = static_cast<std::uint32_t>(submitted_ % ring_slots_);
    auto *dst = reinterpret_cast<volatile std::uint8_t *>(
        out_.va + static_cast<std::uint64_t>(slot) * stride_);
    const auto *s = static_cast<const std::uint8_t *>(src);
    for (std::uint64_t i = 0; i < bytes; ++i) {
        dst[i] = s[i];
    }
}

void *FpgaControllerSession::reply_slot() {
    TP_CHECK(reply_.va != 0, "reply_slot before connect()");
    std::uint32_t slot = static_cast<std::uint32_t>(collected_ % ring_slots_);
    return reinterpret_cast<void *>(reply_.va + static_cast<std::uint64_t>(slot) * stride_);
}

void FpgaControllerSession::post_write(ibv_qp *qp, std::uint32_t work_item_idx,
                                       std::uint64_t cursor, bool signaled) {
    std::uint32_t slot = static_cast<std::uint32_t>(cursor % ring_slots_);
    std::uint64_t off = static_cast<std::uint64_t>(slot) * stride_;

    auto *outp = reinterpret_cast<volatile Payload *>(out_.va + off);
    outp->seq_num = static_cast<std::uint32_t>(cursor + 1);
    outp->decoder_id = pending_decoder_;

    ibv_send_wr &wr = wr_[work_item_idx];
    ibv_sge &sge = sge_[work_item_idx];
    sge.addr = out_.va + off;
    wr.wr.rdma.remote_addr = peer_addr_ + off;
    wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;

    ibv_send_wr *bad = nullptr;
    TP_CHECK(ibv_post_send(qp, &wr, &bad) == 0, "ibv_post_send failed (cursor=%llu)",
             static_cast<unsigned long long>(cursor));
}

int FpgaControllerSession::kick(std::uint32_t work_item_idx) {
    kick_t0_ns_ = mono_ns();

    TP_CHECK(work_item_idx < K_NUM_WORK_ITEMS && committed_[work_item_idx],
             "kick: work_item_idx not committed");
    TP_CHECK(submitted_ - collected_ < max_in_flight_,
             "kick: max_in_flight reached, collect() first");

    std::uint64_t cursor = submitted_;
    bool signaled = (cursor % SIGNAL_EVERY) == 0;
    active_item_ = work_item_idx;

    post_write(fwd_qp_->get(), work_item_idx, cursor, signaled);

    if (signaled) {
        ++signaled_outstanding_;
        reap(fwd_cq_->get(), signaled_outstanding_, /*drain=*/false);
    }
    ++submitted_;
    return 0;
}

volatile PayloadSlot *FpgaControllerSession::poll_message_arrival(std::uint64_t cursor) {
    std::uint32_t slot = static_cast<std::uint32_t>(cursor % ring_slots_);
    std::uint32_t expected = static_cast<std::uint32_t>(cursor + 1);
    auto *rslot = reinterpret_cast<volatile PayloadSlot *>(
        reply_.view + static_cast<std::size_t>(slot) * stride_);

    for (std::uint64_t spins = 0; rslot->p.seq_num != expected; ++spins) {
        if ((spins % SPIN_CHECK_EVERY) != 0) {
            continue;
        }
        // Surface a failed fwd WRITE instead of spinning on a reply that
        // will never arrive.
        ibv_wc wc[REAP_BATCH] = {};
        int got = ibv_poll_cq(fwd_cq_->get(), REAP_BATCH, wc);
        for (int k = 0; k < got; ++k) {
            TP_CHECK(wc[k].status == IBV_WC_SUCCESS,
                     "collect: fwd WRITE failed at cursor=%llu: status=%d (%s)",
                     static_cast<unsigned long long>(cursor), wc[k].status,
                     ibv_wc_status_str(wc[k].status));
            --signaled_outstanding_;
        }
    }
    // seq_num is the data-ready flag; acquire so we observe `value` in the same slot.
    std::atomic_thread_fence(std::memory_order_acquire);
    return rslot;
}

int FpgaControllerSession::collect(void *const *outputs, const std::size_t *output_bytes,
                                   std::size_t n) {
    TP_CHECK(armed_, "collect before commit_work_item()");
    TP_CHECK(submitted_ > collected_, "collect: nothing in flight");
    TP_CHECK(n >= 1 && outputs != nullptr && output_bytes != nullptr, "collect: no output buffer");
    void *replies = outputs[0];
    std::uint64_t bytes = output_bytes[0];
    TP_CHECK(replies != nullptr, "collect: null output");

    std::uint64_t cursor = collected_;
    std::uint32_t corr = correction_bytes_[active_item_];

    TP_CHECK(corr == 0 || bytes >= corr,
             "collect: output buffer (%llu B) smaller than committed correction (%u B)",
             static_cast<unsigned long long>(bytes), corr);

    const std::uint64_t t0 = kick_t0_ns_;
    volatile PayloadSlot *rslot = poll_message_arrival(cursor);
    last_rtt_ns_ = mono_ns() - t0;

    std::uint64_t v = rslot->p.value;
    std::uint64_t nmean = (corr && corr < sizeof(v)) ? corr : sizeof(v);
    if (nmean > bytes) {
        nmean = bytes;
    }
    std::memcpy(replies, &v, static_cast<std::size_t>(nmean));
    if (bytes > nmean) {
        std::memset(static_cast<std::uint8_t *>(replies) + nmean, 0,
                    static_cast<std::size_t>(bytes - nmean));
    }
    ++collected_;
    return 0;
}

void FpgaControllerSession::stop() {
    if (stopped_) {
        return;
    }
    armed_ = false;
    // Drain signaled sends before releasing the CQ so the NIC is done with the SQ.
    if (fwd_cq_) {
        reap(fwd_cq_->get(), signaled_outstanding_, /*drain=*/true);
    }
    fwd_qp_.reset();
    bwd_qp_.reset();
    fwd_cq_.reset();
    bwd_cq_.reset();
    region_free(out_);
    region_free(reply_);
    oob_.reset();
    pd_.reset();
    ctx_.reset();
    stopped_ = true;
}

} // namespace rdma::devices::fpga_verbs
