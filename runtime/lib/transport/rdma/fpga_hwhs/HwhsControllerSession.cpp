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

#include "HwhsControllerSession.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>  // report_rtt()
#include <cstdlib> // getenv (HWHS_RTT_WARMUP)
#include <cstring>
#include <ctime>
#include <string>
#include <sys/mman.h>
#include <vector>

#include "Exception.hpp"
#include "HwhsAbi.h"
#include "TransportCAPI.h" // CATALYST_TRANSPORT_* status codes
#include "WireProtocol.hpp"

#include <fcntl.h>
#include <infiniband/verbs.h>
#include <unistd.h>

namespace catalyst::transport::hwhs {

namespace {

// Map a physical address to a virtual address and return a pointer to the mapped region with page
// alignment (4KB)
volatile std::uint8_t *map_pa(int memfd, std::uint64_t pa, std::size_t len, void **out_base,
                              std::size_t *out_span) {
    auto base = static_cast<off_t>(pa & ~0xfffULL);
    std::size_t span = static_cast<std::size_t>(pa - static_cast<std::uint64_t>(base)) + len;
    void *m = ::mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, base);
    RT_FAIL_IF(m == MAP_FAILED, "mmap(/dev/mem) failed");
    *out_base = m;
    *out_span = span;
    return static_cast<volatile std::uint8_t *>(m) + (pa - static_cast<std::uint64_t>(base));
}

void view_zero(volatile std::uint8_t *p, std::size_t len) {
    for (std::size_t i = 0; i + 4 <= len; i += 4) {
        *reinterpret_cast<volatile std::uint32_t *>(p + i) = 0;
    }
}

std::uint64_t now_ns() {
    timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

// Short names for the statuses the CAPI defines. Aliases rather than literals, so this file
// cannot drift from the values the runtime and the compiled code agree on.
constexpr int kOk = CATALYST_TRANSPORT_OK;
constexpr int kErr = CATALYST_TRANSPORT_ERR;
constexpr int kErrMemory = CATALYST_TRANSPORT_ERR_MEMORY;
constexpr int kErrTimeout = CATALYST_TRANSPORT_ERR_TIMEOUT;
constexpr int kErrStuck = CATALYST_TRANSPORT_ERR_STUCK;

// Demo mode only, and deliberately outside the CAPI set: the pacer reached demo_cnt and
// stopped, which ends a bounded demo run. Callers must not read it as an error -- and must not
// read it as success either, since nothing upstream translates it. Completion is answered by
// HH_DEMO_STATUS.done (see demo_report) and by the trace header's entries/full counters.
constexpr int kDemoComplete = -5;
constexpr std::uint64_t kCollectTimeoutNs = 5'000'000'000ULL;
constexpr std::size_t kRttSamplesReserve = 3'000'000; // 24 MB

constexpr std::uint64_t kReplyTimeoutCycles =
    1000ULL * 1000ULL * static_cast<std::uint64_t>(kClkMhz); // 1s @ 200MHz
static_assert(kReplyTimeoutCycles <= 0xFFFFFFFFULL, "reply_to is a 32-bit cycle count");

int mem_type_of(MemKind kind) {
    switch (kind) {
    case MemKind::CpuRam:
        return XMEM_PS_DDR;
    case MemKind::Ddr:
        return XMEM_PL_DDR;
    case MemKind::Other:
        return XMEM_BRAM;
    case MemKind::GpuHbm:
    default:
        RT_FAIL("HwhsControllerSession: unsupported MemKind for the FPGA controller");
    }
}

} // namespace

HwhsControllerSession::HwhsControllerSession(std::string dev, int gid_idx, std::uint32_t ring_slots,
                                             std::uint32_t stride_log2,
                                             std::optional<MemKind> data_kind,
                                             std::optional<MemKind> sq_kind,
                                             std::optional<MemKind> reply_kind, bool sw_poll,
                                             bool kick_ioctl, DemoCfg demo)
    : dev_(std::move(dev)), gid_idx_(gid_idx), ring_slots_(ring_slots ? ring_slots : 1),
      stride_log2_(stride_log2), stride_((stride_log2 < 32) ? (1u << stride_log2) : 1u),
      data_kind_(data_kind), sq_kind_(sq_kind), reply_kind_(reply_kind), sw_poll_(sw_poll),
      kick_ioctl_(kick_ioctl), demo_(demo) {
    if (demo_.enable) {
        RT_FAIL_IF(demo_.syn_depth == 0 || (demo_.syn_depth % 64u) != 0,
                   "demo_depth must be a non-zero multiple of 64");
        RT_FAIL_IF(demo_.cmd_cnt == 0, "demo_cnt must be non-zero");
        RT_FAIL_IF(demo_.freq_span & (demo_.freq_span + 1u),
                   "demo_span must be 2^N-1 (all ones), e.g. 0xFFFFF");
        RT_FAIL_IF(sw_poll_, "demo mode needs reply_poll=hw (the RTT comes from the engine)");
        RT_FAIL_IF(demo_.trace_pa == 0,
                   "demo mode needs demo_trace=<pa>: the trace RAM is the only thing that "
                   "records rounds");
    }

    rtt_samples_.reserve(kRttSamplesReserve);
    RT_FAIL_IF(stride_log2 >= 32, "stride_log2 out of range (must be < 32)");
    RT_FAIL_IF(stride_ < 16, "stride_log2 too small: a ring slot must be >= 16 bytes");
    RT_FAIL_IF(ring_slots_ != common::K_RING_SLOTS,
               "ring must equal the wire protocol's K_RING_SLOTS");
    RT_FAIL_IF(stride_ != sizeof(common::PayloadSlot),
               "stride_log2 must give the wire protocol's slot size (sizeof(PayloadSlot))");
}

HwhsControllerSession::~HwhsControllerSession() { stop(); }

// The unified region allocation interface
Region HwhsControllerSession::region_alloc(std::uint64_t size, int mem_type, int access) {
    void *ctx = ctx_->get();
    Region r(&umm_, ctx_);
    r.size = size;

    // For case we need to allocate a BRAM on MR
    // we need to allocate a placeholder chunk in PL-DDR
    // and then rebase the MR onto the BRAM PA
    bool bram_mr = access && (mem_type == XMEM_BRAM);
    if (bram_mr) {
        r.mr_chunk = umm_.alloc_chunk(ctx, XMEM_PL_DDR, size, size, false);
        RT_FAIL_IF(r.mr_chunk < 0, "xib_umem_alloc_chunk (MR placeholder) failed");
        r.adv_va = umm_.alloc_mem(ctx, r.mr_chunk, size);
        RT_FAIL_IF(XMEM_IS_INVALID_VADDR(r.adv_va), "xib_umem_alloc_mem (MR placeholder) failed");
    }

    r.chunk = umm_.alloc_chunk(ctx, mem_type, size, size, false);
    RT_FAIL_IF(r.chunk < 0, "xib_umem_alloc_chunk (region) failed");
    r.va = umm_.alloc_mem(ctx, r.chunk, size);
    RT_FAIL_IF(XMEM_IS_INVALID_VADDR(r.va), "xib_umem_alloc_mem (region) failed");
    RT_FAIL_IF(umm_.get_phy_addr(ctx, static_cast<unsigned>(r.chunk), r.va, &r.pa) != 0,
               "xib_umem_get_phy_addr failed");
    if (!bram_mr) {
        r.adv_va = r.va;
    }
    r.view = map_pa(mem_fd_, r.pa, size, &r.map_base, &r.map_span);
    if (access) {
        r.mr = umm_.reg_mr(pd_->get(), r.adv_va, size, access);
        RT_FAIL_IF(!r.mr, "ibv_reg_mr(_ex) failed");
        // Rebase the MR onto the BRAM PA
        if (bram_mr) {
            hh_mr_rebase rb = {};
            rb.rkey = r.mr->rkey;
            rb.bram_pa = r.pa;
            RT_FAIL_IF(ioctl(hh_fd_, HH_MR_REBASE, &rb) != 0, "HH_MR_REBASE failed");
        }
    }
    return r;
}

void Region::release() noexcept {
    void *ctx = ctx_ ? ctx_->get() : nullptr;
    if (mr) {
        ibv_dereg_mr(mr);
        mr = nullptr;
    }
    if (map_base != nullptr && map_span != 0) {
        ::munmap(map_base, map_span);
        map_base = nullptr;
        map_span = 0;
    }
    if (chunk >= 0 && umm_ && ctx) {
        if (!XMEM_IS_INVALID_VADDR(va)) {
            umm_->free_mem(ctx, static_cast<unsigned>(chunk), va, size);
        }
        umm_->free_chunk(ctx, chunk);
        chunk = -1;
    }
    if (mr_chunk >= 0 && umm_ && ctx) {
        if (!XMEM_IS_INVALID_VADDR(adv_va)) {
            umm_->free_mem(ctx, static_cast<unsigned>(mr_chunk), adv_va, size);
        }
        umm_->free_chunk(ctx, mr_chunk);
        mr_chunk = -1;
    }
}

Region::~Region() { release(); }

Region::Region(Region &&other) noexcept { *this = std::move(other); }

Region &Region::operator=(Region &&other) noexcept {
    if (this != &other) {
        release();
        umm_ = other.umm_;
        ctx_ = std::move(other.ctx_);
        chunk = other.chunk;
        mr_chunk = other.mr_chunk;
        va = other.va;
        adv_va = other.adv_va;
        pa = other.pa;
        view = other.view;
        map_base = other.map_base;
        map_span = other.map_span;
        size = other.size;
        mr = other.mr;
        other.umm_ = nullptr;
        other.chunk = -1;
        other.mr_chunk = -1;
        other.va = 0;
        other.adv_va = 0;
        other.pa = 0;
        other.view = nullptr;
        other.map_base = nullptr;
        other.map_span = 0;
        other.size = 0;
        other.mr = nullptr;
    }
    return *this;
}

int HwhsControllerSession::connect(const ConnectInfo &info) {
    RT_FAIL_IF(!umm_.loaded(), "UMM allocator not loaded");

    // Engine + physical-memory device nodes
    hh_fd_ = ::open(kHwhsDev, O_RDWR);
    RT_FAIL_IF(hh_fd_ < 0, "open(/dev/xib0) failed");
    mem_fd_ = ::open("/dev/mem", O_RDWR | O_SYNC);
    RT_FAIL_IF(mem_fd_ < 0, "open(/dev/mem) failed");

    // Map the HWHs register window
    if (!kick_ioctl_) {
        void *m = ::mmap(nullptr, HH_REG_WIN_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, hh_fd_, 0);
        if (m != MAP_FAILED) {
            hh_reg_map_ = m;
            hh_doorbell_ = reinterpret_cast<volatile std::uint32_t *>(
                static_cast<std::uint8_t *>(m) + HH_REG_CTRL);
            hh_demo_doorbell_ = reinterpret_cast<volatile std::uint32_t *>(
                static_cast<std::uint8_t *>(m) + HH_REG_DEMO_CTRL);
        }
    }

    // RDMA bring-up
    ctx_ = std::make_shared<common::Context>(dev_);
    pd_ = std::make_shared<common::ProtectionDomain>(ctx_);

    ibv_port_attr pa = ctx_->port_attr(port_num_);
    RT_FAIL_IF(pa.state != IBV_PORT_ACTIVE, "ERNIC port not ACTIVE");
    active_mtu_ = static_cast<std::uint32_t>(pa.active_mtu);
    ibv_gid gid = ctx_->gid(port_num_, gid_idx_);
    std::memcpy(local_gid_, &gid, 16);

    fwd_cq_ = std::make_shared<common::CompletionQueue>(ctx_, 64);
    bwd_cq_ = std::make_shared<common::CompletionQueue>(ctx_, 64);
    fwd_qp_ = std::make_shared<common::QueuePair>(pd_, fwd_cq_, fwd_cq_, 16);
    bwd_qp_ = std::make_shared<common::QueuePair>(pd_, bwd_cq_, bwd_cq_, 16);
    fwd_qp_->to_init(port_num_);
    bwd_qp_->to_init(port_num_);

    oob_ = common::tcp_connect(info.peer.c_str(), info.oob_port);

    // We need to allocate the reply buffer just before exchange_keys()
    // So that the MR is valid before the exchange_keys() call
    std::uint64_t reply_bytes = static_cast<std::uint64_t>(ring_slots_) * stride_;
    MemKind kind = reply_kind_.value_or(MemKind::Ddr);
    reply_ =
        region_alloc(reply_bytes, mem_type_of(kind),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    return kOk;
}

MemRegion HwhsControllerSession::alloc_memory(std::size_t size, MemKind kind) {
    RT_FAIL_IF(!ctx_, "alloc_memory before connect()");
    caller_regions_.push_back(
        region_alloc(size, mem_type_of(kind),
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
    const Region &r = caller_regions_.back();

    MemRegion out;
    out.addr = reinterpret_cast<void *>(r.adv_va);
    out.size = size;
    out.lkey = r.mr ? r.mr->lkey : 0;
    out.rkey = r.mr ? r.mr->rkey : 0;
    out.kind = kind;
    return out;
}

PeerRef HwhsControllerSession::exchange_keys(const MemRegion & /*local*/) {
    RT_FAIL_IF(!oob_.valid(), "exchange_keys before connect()");
    RT_FAIL_IF(!reply_.mr, "exchange_keys: local reply region has no MR");

    common::HandshakeMsg my = {}, peer = {};
    my.fwd.qpn = fwd_qp_->qpn();
    my.fwd.psn = 0;
    std::memcpy(my.fwd.gid, local_gid_, 16);
    my.bwd.qpn = bwd_qp_->qpn();
    my.bwd.psn = 0;
    std::memcpy(my.bwd.gid, local_gid_, 16);
    my.mr_vaddr = reply_.adv_va;
    my.mr_rkey = reply_.mr->rkey;
    my.mtu_enum = active_mtu_;

    common::recv_exact(oob_.get(), &peer, sizeof(peer));
    common::send_exact(oob_.get(), &my, sizeof(my));
    oob_.reset();

    peer_rkey_ = peer.mr_rkey;
    peer_addr_ = peer.mr_vaddr;
    peer_fwd_ = peer.fwd;
    peer_bwd_ = peer.bwd;
    peer_mtu_ = peer.mtu_enum;

    PeerRef ref;
    ref.rkey = peer.mr_rkey;
    ref.remote_addr = peer.mr_vaddr;
    ref.size = static_cast<std::uint64_t>(ring_slots_) * stride_;
    return ref;
}

void HwhsControllerSession::establish_channel(const ChannelDesc &desc, const MemRegion & /*local*/,
                                              const PeerRef &peer) {
    RT_FAIL_IF(desc.transport != "rdma",
               "HwhsControllerSession only implements the \"rdma\" transport");
    peer_rkey_ = peer.rkey;
    peer_addr_ = peer.remote_addr;

    // Bring both QPs to RTS
    std::uint32_t mtu_enum = (peer_mtu_ && peer_mtu_ < active_mtu_) ? peer_mtu_ : active_mtu_;
    fwd_qp_->to_rtr(peer_fwd_.qpn, peer_fwd_.psn, peer_fwd_.gid, gid_idx_, port_num_, mtu_enum);
    bwd_qp_->to_rtr(peer_bwd_.qpn, peer_bwd_.psn, peer_bwd_.gid, gid_idx_, port_num_, mtu_enum);
    fwd_qp_->to_rts(0);
    bwd_qp_->to_rts(0);

    int out_mem_type = mem_type_of(data_kind_.value_or(MemKind::Ddr));
    int sq_mem_type = mem_type_of(sq_kind_.value_or(MemKind::Ddr));

    // The engine advances the local offset over ring_slots_ slots, so the region has to span all
    // of them or it writes past the end once the cursor passes the allocation.
    std::uint64_t rbytes =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(ring_slots_) * stride_, 4096UL);

    // Allocate the output ring buffer
    out_ = region_alloc(rbytes, out_mem_type, 0);

    // Allocate the SQ ring buffer
    constexpr int sq_depth = 16;
    sq_ring_ = region_alloc(sq_depth * static_cast<std::uint64_t>(kWqeBytes), sq_mem_type, 0);

    RT_FAIL_IF(static_cast<std::uint64_t>(ring_slots_) * stride_ > reply_.size,
               "reply region is smaller than ring_slots * stride");

    view_zero(out_.view, out_.size);
    view_zero(sq_ring_.view, sq_ring_.size);
    view_zero(reply_.view, reply_.size);

    // The physical QP for ERNIC™ should be subtracted by 1
    std::uint32_t phys_qp = (fwd_qp_->qpn() > 0) ? fwd_qp_->qpn() - 1 : 0;
    hh_qpctx_cfg qc = {};
    qc.ctx_idx = 0;
    qc.qp_num = phys_qp;
    qc.sq_base = sq_ring_.pa;
    qc.sq_depth = sq_depth;
    qc.reply_buf = reply_.pa;
    qc.flags = HH_FLAG_REPOST_WQE | (sw_poll_ ? 0u : HH_FLAG_REPLY_POLL);
    qc.reply_to = static_cast<std::uint32_t>(kReplyTimeoutCycles);
    static_assert(offsetof(common::Payload, seq_num) % 4 == 0,
                  "reply-poll selects a dword: Payload::seq_num must be 4-byte aligned");
    static_assert(offsetof(common::Payload, seq_num) + sizeof(common::Payload::seq_num) <=
                      sizeof(common::PayloadSlot),
                  "reply-poll addresses within one slot: Payload::seq_num must fall inside it");
    qc.reply_seq_off = static_cast<std::uint32_t>(offsetof(common::Payload, seq_num));
    qc.reply_stride_log2 = stride_log2_;
    qc.rebind = 1;
    RT_FAIL_IF(ioctl(hh_fd_, HH_QPCTX_COMMIT, &qc) != 0, "HH_QPCTX_COMMIT failed");
}

void HwhsControllerSession::commit_work_item(std::uint32_t work_item_idx, std::uint64_t in_bytes,
                                             std::uint64_t out_bytes) {
    RT_FAIL_IF(!out_.view, "commit_work_item before establish_channel()");
    RT_FAIL_IF(work_item_idx != 0, "only a single work item (idx 0) is supported");
    std::uint32_t syndrome_bytes = static_cast<std::uint32_t>(in_bytes);
    std::uint32_t correction_bytes = static_cast<std::uint32_t>(out_bytes);
    RT_FAIL_IF(syndrome_bytes == 0, "commit_work_item: schema in_bytes is 0");
    RT_FAIL_IF(syndrome_bytes > common::PAYLOAD_DATA_BYTES,
               "commit_work_item: schema in_bytes exceeds Payload::value");
    RT_FAIL_IF(correction_bytes > common::PAYLOAD_DATA_BYTES,
               "commit_work_item: correction size exceeds Payload::value");

    hh_wqe_cfg w = {};
    w.idx = work_item_idx;
    w.opcode = HH_OP_WRITE;
    w.xfer_len = static_cast<std::uint32_t>(sizeof(common::Payload));
    w.rkey = peer_rkey_;
    w.va_lsb = static_cast<std::uint32_t>(peer_addr_);
    w.va_msb = static_cast<std::uint32_t>(peer_addr_ >> 32);
    w.local_offset = out_.pa;
    w.qp_ref = 0; // qp 0 as default
    w.wrid = 0;
    w.ring_en = 1;
    w.ring_len = ring_slots_;
    w.stride_local_log2 = stride_log2_;
    w.stride_remote_log2 = stride_log2_;
    RT_FAIL_IF(ioctl(hh_fd_, HH_WQE_COMMIT, &w) != 0, "HH_WQE_COMMIT failed");

    committed_[work_item_idx] = true;

    // Reset the reply ring
    view_zero(reply_.view, reply_.size);
    submitted_ = 0;
    collected_ = 0;
    armed_ = true;

    // Setup the demo mode
    demo_program();
}

// Push the DemoCfg into the engine and clear the demo counters.
void HwhsControllerSession::demo_program() {
    if (!demo_.enable) {
        return;
    }

    RT_FAIL_IF(hh_doorbell_ == nullptr,
               "demo mode needs the mmap'd register window (use kick=mmap)");
    *hh_doorbell_ = 0;
#if defined(__aarch64__)
    asm volatile("dsb st" ::: "memory");
#endif

    hh_demo_cfg dc = {};
    dc.freq_num = demo_.freq_num;
    dc.cmd_cnt = demo_.cmd_cnt;
    dc.syn_depth = demo_.syn_depth;
    dc.freq_span = demo_.freq_span;
    dc.lfsr_seed = demo_.lfsr_seed;
    RT_FAIL_IF(ioctl(hh_fd_, HH_DEMO_CFG, &dc) != 0, "HH_DEMO_CFG failed");
    demo_preload();

    demo_armed_ = false; // armed at the first kick(), see demo_arm()
}

// Start the pacer
void HwhsControllerSession::demo_arm() {
    // Baseline for the round counter
    hh_status_rd st = {};
    last_round_cnt_ = (ioctl(hh_fd_, HH_READ_STATUS, &st) == 0) ? st.round_cnt : 0u;
    rounds_seen_ = 0;
    demo_finished_ = false;

    {
        hh_demo_status ds0 = {};
        err_cnt_base_ = (ioctl(hh_fd_, HH_DEMO_STATUS, &ds0) == 0) ? ds0.err_cnt : 0u;
    }
    // Arm the pacer, once
    RT_FAIL_IF(ioctl(hh_fd_, HH_TRACE_ARM) != 0, "HH_TRACE_ARM failed");

    if (hh_demo_doorbell_ != nullptr) {
        *hh_demo_doorbell_ = HH_DEMO_START;
#if defined(__aarch64__)
        asm volatile("dsb st" ::: "memory");
#endif
    } else {
        RT_FAIL_IF(ioctl(hh_fd_, HH_DEMO_START_RUN) != 0, "HH_DEMO_START_RUN failed");
    }
}

// Fill the BRAM syndrome table transport_demo replays
void HwhsControllerSession::demo_preload() {
    const std::uint32_t slots = demo_.syn_depth / 64u;
    RT_FAIL_IF(slots == 0, "demo_depth must cover at least one 64 B slot");

    const std::size_t table_bytes = static_cast<std::size_t>(demo_.syn_depth) * 2;
    void *base = nullptr;
    std::size_t span = 0;
    volatile std::uint8_t *tbl = map_pa(mem_fd_, demo_.bram_pa, table_bytes, &base, &span);

    if (!demo_.table_path.empty()) {
        std::FILE *f = std::fopen(demo_.table_path.c_str(), "rb");
        if (f == nullptr) {
            ::munmap(base, span);
            RT_FAIL("demo_table: cannot open the table image");
        }
        std::vector<std::uint8_t> img(table_bytes);
        const std::size_t got = std::fread(img.data(), 1, table_bytes, f);
        const bool too_long = std::fgetc(f) != EOF;
        std::fclose(f);
        if (got != table_bytes || too_long) {
            std::fprintf(stderr,
                         "[hwhs demo] %s is %zu bytes%s; demo_depth=%u needs exactly %zu "
                         "(syndrome half then expected half)\n",
                         demo_.table_path.c_str(), got, too_long ? "+" : "", demo_.syn_depth,
                         table_bytes);
            ::munmap(base, span);
            RT_FAIL("demo_table: wrong size for demo_depth");
        }
        for (std::size_t b = 0; b < table_bytes; ++b) {
            tbl[b] = img[b];
        }
    } else {
        for (std::uint32_t i = 0; i < slots; ++i) {
            const std::uint64_t value = static_cast<std::uint64_t>(i) + 1;
            for (std::uint32_t half = 0; half < 2; ++half) {
                volatile std::uint8_t *p = tbl + half * demo_.syn_depth + i * 64u;
                for (std::size_t b = 0; b < 64; ++b) {
                    p[b] = 0;
                }
                for (std::size_t b = 0; b < sizeof(value); ++b) {
                    p[b] = static_cast<std::uint8_t>(value >> (8 * b));
                }
                *reinterpret_cast<volatile std::uint32_t *>(
                    p + offsetof(common::Payload, decoder_id)) = 0;
                *reinterpret_cast<volatile std::uint32_t *>(
                    p + offsetof(common::Payload, seq_num)) = static_cast<std::uint32_t>(value);
            }
        }
    }
#if defined(__aarch64__)
    asm volatile("dsb st" ::: "memory");
#endif
    ::munmap(base, span);
}

// Read the trace RAM out after the run and write it as cycles,ns
void HwhsControllerSession::demo_dump_trace() const {
    if (demo_.trace_pa == 0 || demo_.trace_out.empty() || mem_fd_ < 0) {
        return;
    }
    hh_trace_status ts = {};
    if (ioctl(hh_fd_, HH_TRACE_STATUS, &ts) != 0 || ts.cnt == 0) {
        std::fprintf(stderr, "[hwhs trace] nothing recorded\n");
        return;
    }

    void *base = nullptr;
    std::size_t span = 0;
    const std::size_t bytes = static_cast<std::size_t>(ts.cnt) * sizeof(std::uint16_t);
    volatile std::uint8_t *p = map_pa(mem_fd_, demo_.trace_pa, bytes, &base, &span);

    std::FILE *f = std::fopen(demo_.trace_out.c_str(), "w");
    if (f == nullptr) {
        std::fprintf(stderr, "[hwhs trace] cannot write %s\n", demo_.trace_out.c_str());
        ::munmap(base, span);
        return;
    }
    std::fprintf(f, "# rtt trace, one row per completed round, in order\n");
    std::fprintf(f, "# clk_hz=%llu entries=%u full=%u saturated=%u\n",
                 static_cast<unsigned long long>(kClkMhz * 1000000.0), ts.cnt, ts.full, ts.sat);
    std::fprintf(f, "cycles,ns\n");
    bool all_zero = true;
    for (std::uint32_t i = 0; i < ts.cnt; ++i) {
        const std::uint16_t cyc =
            *reinterpret_cast<volatile const std::uint16_t *>(p + i * sizeof(std::uint16_t));
        all_zero = all_zero && (cyc == 0);
        std::fprintf(f, "%u,%.0f\n", cyc, static_cast<double>(cyc) * 1000.0 / kClkMhz);
    }
    std::fclose(f);
    ::munmap(base, span);

    if (all_zero) {
        std::fprintf(stderr,
                     "[hwhs trace] every one of %u entries read back as 0 at 0x%llX: either "
                     "C_TRACE_EN=0 in the bitstream or demo_trace= names the wrong address\n",
                     ts.cnt, static_cast<unsigned long long>(demo_.trace_pa));
    }
    std::fprintf(stderr, "[hwhs trace] wrote %u round(s) to %s%s%s\n", ts.cnt,
                 demo_.trace_out.c_str(),
                 ts.full ? "   <-- buffer filled: rounds after this were not recorded" : "",
                 ts.sat ? "   <-- some entry clamped to 0xFFFF: it is a ceiling, not a value" : "");
}

void HwhsControllerSession::demo_diff_dump() const {
    if (!demo_.enable || mem_fd_ < 0 || reply_.view == nullptr) {
        return;
    }
    {
        hh_demo_status ds = {};
        if (ioctl(hh_fd_, HH_DEMO_STATUS, &ds) != 0 || ds.err_cnt == err_cnt_base_) {
            return;
        }
    }
    void *base = nullptr;
    std::size_t span = 0;
    volatile std::uint8_t *exp = map_pa(mem_fd_, demo_.bram_pa + demo_.syn_depth,
                                        static_cast<std::size_t>(demo_.syn_depth), &base, &span);

    std::fprintf(stderr, "[hwhs demo] expected table @0x%llx vs reply ring @0x%llx\n",
                 static_cast<unsigned long long>(demo_.bram_pa + demo_.syn_depth),
                 static_cast<unsigned long long>(reply_.pa));
    const std::uint32_t table_slots = demo_.syn_depth / 64u;
    for (std::uint32_t slot = 0; slot < 3 && slot < ring_slots_; ++slot) {
        const volatile std::uint8_t *r = reply_.view + static_cast<std::size_t>(slot) * stride_;
        const std::uint32_t rseq = *reinterpret_cast<volatile const std::uint32_t *>(
            r + offsetof(common::Payload, seq_num));
        // round n (1-based) read table entry (n-1) % table_slots
        const std::uint32_t tslot = rseq ? ((rseq - 1u) % table_slots) : 0u;
        const volatile std::uint8_t *e = exp + static_cast<std::size_t>(tslot) * 64u;
        char eb[200] = {}, rb[200] = {}, db[200] = {};
        int ep = 0, rp = 0, dp = 0;
        for (int i = 0; i < 32; ++i) { // the first 32 bytes carry everything meaningful
            ep += std::snprintf(eb + ep, sizeof(eb) - ep, "%02x", e[i]);
            rp += std::snprintf(rb + rp, sizeof(rb) - rp, "%02x", r[i]);
            dp += std::snprintf(db + dp, sizeof(db) - dp, "%s", e[i] == r[i] ? ".." : "XX");
        }
        std::fprintf(stderr, "  reply slot %u (seq=%u) vs table slot %u\n", slot, rseq, tslot);
        std::fprintf(stderr, "         exp %s\n", eb);
        std::fprintf(stderr, "         rpl %s\n", rb);
        std::fprintf(stderr, "         dif %s\n", db);
    }
    ::munmap(base, span);
}

void HwhsControllerSession::demo_dump(const char *why) const {
    static const char *kStates[16] = {"S_IDLE",    "S_WQE_ISS",  "S_WQE_WAIT",   "S_DAT_WAIT",
                                      "S_DAT_ISS", "S_RING_ISS", "S_RING_WAIT",  "S_WAIT_RPL",
                                      "S_NEXT",    "S_DAT_IDLE", "S_DAT_TX_ISS", "S_DAT_TX_WAIT",
                                      "resv12",    "S_POLL_ISS", "S_POLL_WAIT",  "S_JOIN"};
    hh_status_rd st = {};
    if (ioctl(hh_fd_, HH_READ_STATUS, &st) != 0) {
        std::fprintf(stderr, "[hwhs demo] %s: HH_READ_STATUS failed\n", why);
        return;
    }
    std::fprintf(stderr,
                 "[hwhs demo] %s: STATUS=0x%03x state=%s busy=%u done=%u reply_seen=%u "
                 "timeout=%u axi_err=%u round_cnt=%u last_rd=0x%08x%s\n",
                 why, st.status, kStates[st.status & HH_ST_STATE_MASK], (st.status >> 4) & 1u,
                 (st.status >> 5) & 1u, (st.status >> 6) & 1u, (st.status >> 7) & 1u,
                 (st.status >> 8) & 1u, st.round_cnt, st.last_rd,
                 (st.status == 0) ? "   <-- engine held in reset/abort? check REG_CTRL[2:1]" : "");

    hh_demo_status ds = {};
    if (ioctl(hh_fd_, HH_DEMO_STATUS, &ds) == 0) {
        std::fprintf(stderr,
                     "[hwhs demo]   readback CMD_CNT=%u DEMO_DEPTH=%u FREQ=%llu done=%u "
                     "cmp_err=%u   (programmed %u / %u / %llu)%s\n",
                     ds.cmd_cnt, ds.syn_depth, static_cast<unsigned long long>(ds.freq_num),
                     ds.done, ds.err_cnt, demo_.cmd_cnt, demo_.syn_depth,
                     static_cast<unsigned long long>(demo_.freq_num),
                     (ds.cmd_cnt != demo_.cmd_cnt || ds.syn_depth != demo_.syn_depth)
                         ? "   <-- readback disagrees: the engine did not take the config"
                         : "");
    }
}

void HwhsControllerSession::demo_report() const {
    if (!demo_.enable || hh_fd_ < 0) {
        return;
    }
    hh_demo_status ds = {};
    if (ioctl(hh_fd_, HH_DEMO_STATUS, &ds) != 0) {
        return;
    }
    // rounds_seen_ is what the host watched the engine's counter advance by; the authoritative
    // per-round record is the trace, and demo_dump_trace() reports its own entry count.
    std::fprintf(stderr, "[hwhs demo] engine ran %llu round(s)%s\n",
                 static_cast<unsigned long long>(rounds_seen_),
                 demo_finished_ ? "  [run complete]" : "");
    const std::uint32_t err_this_run = ds.err_cnt - err_cnt_base_;
    std::fprintf(
        stderr, "[hwhs demo] cmd_cnt=%u done=%u compare_errors=%u this run (%u since power-on)%s\n",
        ds.cmd_cnt, ds.done, err_this_run, ds.err_cnt, err_this_run ? "   <-- DATA MISMATCH" : "");
}

void *HwhsControllerSession::data_slot() {
    RT_FAIL_IF(!out_.view, "data_slot before establish_channel()");
    std::uint32_t slot = static_cast<std::uint32_t>(submitted_ % ring_slots_);
    return const_cast<std::uint8_t *>(out_.view + static_cast<std::size_t>(slot) * stride_);
}

void HwhsControllerSession::write_data_slot(const void *src, std::uint64_t bytes,
                                            std::uint32_t decoder_id) {
    RT_FAIL_IF(!out_.view, "write_data_slot before establish_channel()");
    RT_FAIL_IF(bytes > common::PAYLOAD_DATA_BYTES, "write_data_slot: bytes exceeds Payload::value");
    if (demo_.enable) {
        return;
    }
    std::uint32_t slot = static_cast<std::uint32_t>(submitted_ % ring_slots_);
    volatile std::uint8_t *dst = out_.view + static_cast<std::size_t>(slot) * stride_;

    const auto *s = static_cast<const std::uint8_t *>(src);
    for (std::uint64_t i = 0; i < bytes; ++i) {
        dst[i] = s[i];
    }

    // The header is the transport's to write, both fields of it: a peer notices a request by
    // polling seq_num for this round's number, and decoder_id says which of its decoders the
    // round is for. Neither can be left to the caller -- the bound above stops application data
    // before offset 8, so a caller cannot reach either field even if it wanted to.
    *reinterpret_cast<volatile std::uint32_t *>(dst + offsetof(common::Payload, decoder_id)) =
        decoder_id;
    *reinterpret_cast<volatile std::uint32_t *>(dst + offsetof(common::Payload, seq_num)) =
        static_cast<std::uint32_t>(submitted_ + 1);
}

void *HwhsControllerSession::reply_slot() {
    RT_FAIL_IF(!reply_.view, "reply_slot before connect()");
    std::uint32_t slot = static_cast<std::uint32_t>(collected_ % ring_slots_);
    return const_cast<std::uint8_t *>(reply_.view + static_cast<std::size_t>(slot) * stride_);
}

int HwhsControllerSession::kick(std::uint32_t work_item_idx) {
    RT_FAIL_IF(work_item_idx != 0, "only a single work item (idx 0) is supported");
    RT_FAIL_IF(!committed_[work_item_idx], "kick: work_item not committed");
    RT_FAIL_IF(submitted_ - collected_ >= 1,
               "kick: previous round still in flight, collect() first");

    if (active_item_ != work_item_idx) {
        // update the WQE index only if it's different from the active item
        std::uint32_t idx = work_item_idx;
        RT_FAIL_IF(ioctl(hh_fd_, HH_SET_WQE_IDX, &idx) != 0, "HH_SET_WQE_IDX failed");
        active_item_ = work_item_idx;
    }

    // Starts the round trip the app is charged for: everything from here to the reply being
    // visible.
    kick_ns_ = now_ns();

    if (demo_.enable) {
        if (!demo_armed_) {
            demo_arm(); // both sides are up by now; see demo_arm() for why not earlier
            demo_armed_ = true;
        }
        ++submitted_;
        return kOk;
    }

    if (hh_doorbell_ != nullptr) {
#if defined(__aarch64__)
        // Order the slot stores ahead of the START, so the engine cannot transmit a stale slot.
        asm volatile("dsb st" ::: "memory");
#endif
        // Userspace doorbell
        *hh_doorbell_ = HH_CTRL_START;
#if defined(__aarch64__)
        asm volatile("dsb st" ::: "memory");
#else
#error "HwhsControllerSession START doorbell needs an aarch64 Device-MMIO store barrier (dsb st)"
#endif
    } else {
        RT_FAIL_IF(ioctl(hh_fd_, HH_START) != 0, "HH_START failed");
    }
    ++submitted_;
    return kOk;
}

int HwhsControllerSession::collect(void *const *replies, const std::uint64_t * /*replies_bytes*/,
                                   std::size_t n) {
    RT_FAIL_IF(!armed_, "collect before commit_work_item()");
    RT_FAIL_IF(submitted_ == collected_, "collect: nothing in flight");

    if (demo_.enable) { // the trace is the only demo recording path
        hh_demo_status ds = {};
        hh_status_rd st = {};
        std::uint32_t last_seen = last_round_cnt_;
        std::uint64_t stall_deadline = now_ns() + kCollectTimeoutNs;
        for (;;) {
            if (ioctl(hh_fd_, HH_READ_STATUS, &st) == 0) {
                if (st.status & (HH_ST_TIMEOUT | HH_ST_AXI_ERR)) {
                    ++collected_;
                    return (st.status & HH_ST_AXI_ERR) ? kErrMemory : kErrTimeout;
                }
                if (st.round_cnt != last_seen) {
                    rounds_seen_ += static_cast<std::uint32_t>(st.round_cnt - last_seen);
                    last_seen = st.round_cnt;
                    stall_deadline = now_ns() + kCollectTimeoutNs;
                }
            }
            if (ioctl(hh_fd_, HH_DEMO_STATUS, &ds) == 0 && ds.done) {
                demo_finished_ = true;
                last_round_cnt_ = last_seen;
                return kDemoComplete;
            }
            if (now_ns() >= stall_deadline) {
                demo_dump("trace run stalled");
                ++collected_;
                return kErrStuck;
            }
            struct timespec ts = {0, 200'000}; // 0.2 ms; nothing here is latency-critical
            nanosleep(&ts, nullptr);
        }
    }

    RT_FAIL_IF(demo_.enable, "demo mode must be handled by the trace branch above");

    RT_FAIL_IF(n == 0 || replies == nullptr || replies[0] == nullptr, "collect: no reply buffer");
    volatile std::uint8_t *slot = static_cast<volatile std::uint8_t *>(replies[0]);

    if (sw_poll_) {
        volatile std::uint32_t *seq =
            reinterpret_cast<volatile std::uint32_t *>(slot + offsetof(common::Payload, seq_num));
        const std::uint32_t expected = static_cast<std::uint32_t>(collected_ + 1);
        const std::uint64_t deadline = now_ns() + kCollectTimeoutNs;
        for (;;) {
            if (*seq == expected) {
                break;
            }
            if (now_ns() >= deadline) {
                ++collected_;
                return kErrStuck;
            }
        }
        last_rtt_ns_ = now_ns() - kick_ns_;
    } else {
        // HW reply-poll
        hh_status_rd st = {};
        const std::uint64_t deadline = now_ns() + kCollectTimeoutNs;
        for (;;) {
            RT_FAIL_IF(ioctl(hh_fd_, HH_READ_STATUS, &st) != 0, "HH_READ_STATUS failed");
            if (st.status & (HH_ST_TIMEOUT | HH_ST_AXI_ERR)) {
                ++collected_;
                return (st.status & HH_ST_AXI_ERR) ? kErrMemory : kErrTimeout;
            }
            if (st.status & HH_ST_DONE) {
                break;
            }
            if (now_ns() >= deadline) {
                ++collected_;
                return kErrStuck;
            }
        }
        last_rtt_ns_ = static_cast<std::uint64_t>(static_cast<double>(st.rtt) * 1000.0 / kClkMhz);
    }

    rtt_samples_.push_back(last_rtt_ns_);
    ++collected_;
    return kOk;
}

void HwhsControllerSession::report_rtt() const {
    if (rtt_samples_.empty()) {
        return;
    }
    std::size_t warmup = 0;
    if (const char *w = std::getenv("HWHS_RTT_WARMUP")) {
        warmup = static_cast<std::size_t>(std::strtoull(w, nullptr, 10));
    }
    if (warmup >= rtt_samples_.size()) {
        warmup = 0;
    }
    std::vector<std::uint64_t> v(rtt_samples_.begin() + static_cast<std::ptrdiff_t>(warmup),
                                 rtt_samples_.end());
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    long double sum = 0.0L;
    for (std::uint64_t s : v) {
        sum += static_cast<long double>(s);
    }
    const auto pct = [&](double p) { return v[static_cast<std::size_t>(p * (n - 1))]; };
    std::fprintf(stderr,
                 "\n=== engine RTT (n=%zu, %zu warmup dropped, hardware handshake) ===\n"
                 "  min      %8llu ns\n  p50      %8llu ns\n  p95      %8llu ns\n"
                 "  p99      %8llu ns\n  p99.9    %8llu ns\n  max      %8llu ns\n"
                 "  mean     %8llu ns\n",
                 n, warmup, (unsigned long long)v.front(), (unsigned long long)pct(0.50),
                 (unsigned long long)pct(0.95), (unsigned long long)pct(0.99),
                 (unsigned long long)pct(0.999), (unsigned long long)v.back(),
                 (unsigned long long)(sum / n));
}

void HwhsControllerSession::stop() {
    if (stopped_) {
        return;
    }
    report_rtt();

    if (demo_.enable) {
        demo_report();
        demo_diff_dump();
        demo_dump_trace();
    }
    if (hh_fd_ >= 0 && armed_) {
        if (demo_.enable) {
            ioctl(hh_fd_, HH_DEMO_STOP_RUN);
        }
        ioctl(hh_fd_, HH_ABORT);
        ioctl(hh_fd_, HH_RESET);
    }
    armed_ = false;

    out_ = Region{};
    sq_ring_ = Region{};
    reply_ = Region{};

    fwd_qp_.reset();
    bwd_qp_.reset();
    fwd_cq_.reset();
    bwd_cq_.reset();
    pd_.reset();
    ctx_.reset();
    oob_.reset();

    if (hh_reg_map_ != nullptr) {
        ::munmap(hh_reg_map_, HH_REG_WIN_SIZE);
        hh_reg_map_ = nullptr;
        hh_doorbell_ = nullptr;
    }
    if (hh_fd_ >= 0) {
        ::close(hh_fd_);
        hh_fd_ = -1;
    }
    if (mem_fd_ >= 0) {
        ::close(mem_fd_);
        mem_fd_ = -1;
    }
    stopped_ = true;
}

} // namespace catalyst::transport::hwhs
