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

#include "VerbsSessionBase.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <utility>

#include "Error.hpp"
#include "Handshake.hpp"
#include "RealtimeThread.hpp"
#include "WireProtocol.hpp"

#include <arpa/inet.h>

namespace catalyst::transport::common {
using namespace catalyst::transport;

namespace {
// RDMA device port; rxe0 is single-port -> 1.
constexpr std::uint8_t PORT = 1;
// Max CQEs taken per non-blocking batch reap.
constexpr int REAP_BATCH = 16;
// Page alignment for registered host buffers.
constexpr std::size_t PAGE_ALIGN = 4096;
} // namespace

template <class Role>
VerbsSessionBase<Role>::VerbsSessionBase(std::string dev, int gid_idx, const char *backend,
                                         QpShape fwd_shape, frame::RingGeometry ring,
                                         bool cuqr_write_imm)
    : dev_name_(std::move(dev)), gid_idx_(gid_idx), backend_(backend), fwd_shape_(fwd_shape),
      ring_(ring), cuqr_write_imm_(cuqr_write_imm) {}

template <class Role> int VerbsSessionBase<Role>::connect(const ConnectInfo &info) {
    ctx_ = std::make_shared<Context>(dev_name_);
    pd_ = std::make_shared<ProtectionDomain>(ctx_);
    constexpr QpShape bwd = DEFAULT_QP_SHAPE;
    fwd_cq_ = std::make_shared<CompletionQueue>(ctx_, fwd_shape_.cq_depth);
    bwd_cq_ = std::make_shared<CompletionQueue>(ctx_, bwd.cq_depth);
    // A CUQR receiver holds one receive request per slot, and its CQ must hold their completions.
    const std::uint32_t recv_wr = cuqr() ? ring_.slots : 4;
    TP_CHECK(!cuqr() || static_cast<std::uint32_t>(fwd_shape_.cq_depth) >= ring_.slots,
             "%s: a %d-entry CQ cannot hold a %u-slot ring's receive completions", backend_,
             fwd_shape_.cq_depth, ring_.slots);
    fwd_qp_ = std::make_shared<QueuePair>(pd_, fwd_cq_, fwd_cq_, fwd_shape_.sq_depth,
                                          fwd_shape_.inline_max, recv_wr);
    bwd_qp_ =
        std::make_shared<QueuePair>(pd_, bwd_cq_, bwd_cq_, bwd.sq_depth, bwd.inline_max, recv_wr);
    fwd_qp_->to_init(PORT);
    bwd_qp_->to_init(PORT);
    active_mtu_ = static_cast<std::uint32_t>(ctx_->port_attr(PORT).active_mtu);
    mygid_ = ctx_->gid(PORT, gid_idx_);
    const bool listen = oob_listens();
    oob_fd_ =
        listen ? tcp_listen_accept(info.oob_port) : tcp_connect(info.peer.c_str(), info.oob_port);
    return 0;
}

template <class Role>
MemRegion VerbsSessionBase<Role>::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(kind == MemKind::CpuRam, "cpu_libibverbs: only MemKind::CpuRam supported");
    caller_memory_regions_.push_back(MemoryRegion::alloc_host(
        pd_, size, PAGE_ALIGN, MemAccess::LOCAL_WRITE | MemAccess::REMOTE_WRITE));
    const MemoryRegion &mr = caller_memory_regions_.back();
    return MemRegion{
        .addr = mr.addr(),
        .size = size,
        .lkey = mr.lkey(),
        .rkey = mr.rkey(),
        .kind = MemKind::CpuRam,
    };
}

template <class Role> PeerRef VerbsSessionBase<Role>::exchange_keys(const MemRegion &local) {
    HandshakeMsg my{
        .fwd = {.qpn = fwd_qp_->qpn(), .psn = 0},
        .bwd = {.qpn = bwd_qp_->qpn(), .psn = 0},
        .mr_vaddr = reinterpret_cast<std::uint64_t>(local.addr),
        .mr_rkey = local.rkey,
        .mtu_enum = active_mtu_,
        .frame_mode = static_cast<std::uint32_t>(ring_.mode),
        .frame_stride = ring_.stride,
        .cuqr_tx = cuqr_write_imm_ ? 1U : 0U,
    };
    // gid is a 16-byte array copied from the local GID after the aggregate
    // init.
    std::memcpy(my.fwd.gid, &mygid_, sizeof(mygid_));
    std::memcpy(my.bwd.gid, &mygid_, sizeof(mygid_));
    HandshakeMsg peer{};
    const int fd = oob_fd_.get();
    const bool listen = oob_listens();
    if (listen) {
        send_exact(fd, &my, sizeof(my));
        recv_exact(fd, &peer, sizeof(peer));
    } else {
        recv_exact(fd, &peer, sizeof(peer));
        send_exact(fd, &my, sizeof(my));
    }
    TP_CHECK(peer.frame_mode == my.frame_mode && peer.frame_stride == my.frame_stride,
             "%s: the peer's frames (mode %u, %u B slots) differ from this end's (mode %u, %u B "
             "slots); both ends need the same frame config",
             backend_, peer.frame_mode, peer.frame_stride, my.frame_mode, my.frame_stride);
    TP_CHECK(!cuqr() || peer.cuqr_tx == my.cuqr_tx,
             "%s: the peer completes CUQR frames with %s and this end with %s; set the same "
             "cuqr_tx on both",
             backend_, peer.cuqr_tx != 0 ? "write_imm" : "send",
             my.cuqr_tx != 0 ? "write_imm" : "send");
    const std::uint32_t mtu = std::min(my.mtu_enum, peer.mtu_enum);
    fwd_qp_->to_rtr(peer.fwd.qpn, peer.fwd.psn, peer.fwd.gid, gid_idx_, PORT, mtu);
    bwd_qp_->to_rtr(peer.bwd.qpn, peer.bwd.psn, peer.bwd.gid, gid_idx_, PORT, mtu);
    if (cuqr()) { // receive requests are in place before the peer can send
        local_ = local;
        for (std::uint64_t s = 0; s < ring_.slots; ++s) {
            post_recv(s);
        }
    }
    fwd_qp_->to_rts(my.fwd.psn);
    bwd_qp_->to_rts(my.bwd.psn);
    return PeerRef{
        .rkey = peer.mr_rkey,
        .remote_addr = peer.mr_vaddr,
        .size = ring_.region_bytes(),
    };
}

template <class Role>
void VerbsSessionBase<Role>::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                               const PeerRef &peer) {
    TP_CHECK(desc.transport == "rdma", "%s: only transport \"rdma\" supported", backend_);
    TP_CHECK(local.size >= ring_.region_bytes(), "region too small for ring: %zu < %zu",
             static_cast<std::size_t>(local.size), ring_.region_bytes());
    desc_ = desc;
    local_ = local;
    peer_ = peer;
    send_buf_ = common::MemoryRegion::alloc_host(pd_, frame::frame_bytes(ring_), ring_.stride,
                                                 common::MemAccess::LOCAL_WRITE);
}

template <class Role>
void VerbsSessionBase<Role>::post_write(ibv_qp *qp, std::uint64_t cursor, bool inline_data,
                                        bool signaled) {
    void *send = send_frame(); // staged by the caller
    frame::seal(ring_, send, static_cast<std::uint32_t>(cursor + 1));
    const std::size_t bytes = frame::wire_bytes(ring_, send);
    // A frame larger than the queue pair's inline capacity goes from the registered send buffer.
    const int inline_max =
        qp == fwd_qp_->get() ? fwd_shape_.inline_max : DEFAULT_QP_SHAPE.inline_max;
    inline_data = inline_data && bytes <= static_cast<std::size_t>(inline_max);
    ibv_sge sge{
        .addr = reinterpret_cast<std::uint64_t>(send),
        .length = static_cast<std::uint32_t>(bytes),
        .lkey = send_buf_->lkey(),
    };
    ibv_send_wr wr{
        .sg_list = &sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = static_cast<unsigned>((inline_data ? IBV_SEND_INLINE : 0) |
                                            (signaled ? IBV_SEND_SIGNALED : 0)),
    };
    wr.wr.rdma.remote_addr = peer_.remote_addr + ring_.slot_offset(cursor);
    wr.wr.rdma.rkey = peer_.rkey;
    if (cuqr()) {
        if (cuqr_write_imm_) { // the immediate names the slot written
            wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wr.imm_data = htonl(static_cast<std::uint32_t>(cursor & (ring_.slots - 1)));
        } else { // lands in the receive buffer the peer posted for this slot
            wr.opcode = IBV_WR_SEND;
        }
    }
    ibv_send_wr *bad = nullptr;
    const int rc = ibv_post_send(qp, &wr, &bad);
    TP_CHECK(rc == 0, "ibv_post_send rc=%d (%s)", rc, std::strerror(rc));
}

// Non-blocking batch reap of `cq`: take whatever completions are ready (up to
// REAP_BATCH) and decrement `outstanding`. With drain=true, keep polling until
// every signaled send has completed (teardown), guarded so a lost CQE can't
// hang the join.
template <class Role> void VerbsSessionBase<Role>::reap(ibv_cq *cq, int &outstanding, bool drain) {
    std::array<ibv_wc, REAP_BATCH> wc;
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
        for (int k = 0; k < n; k++) {
            TP_CHECK(wc[k].status == IBV_WC_SUCCESS, "CQE status=%d", wc[k].status);
            --outstanding;
        }
    } while (drain && outstanding > 0);
}

template <class Role>
bool VerbsSessionBase<Role>::poll_message_arrival(std::uint64_t cursor, std::stop_token st,
                                                  frame::FrameView &out) {
    if (cuqr()) {
        if (!wait_recv(cursor, st)) {
            return false;
        }
        frame::poll(ring_, ring_slot(cursor), static_cast<std::uint32_t>(cursor + 1), out);
        return true;
    }
    // Slots are reused (the slot count is a power of two). The peer writes each frame into the
    // head of its slot in a single RDMA_WRITE, seq last, so once frame::poll sees the expected seq
    // the payload before it is present.
    void *slot = ring_slot(cursor);
    const auto expected = static_cast<std::uint32_t>(cursor + 1);
    std::uint32_t spins = 0;
    while (!frame::poll(ring_, slot, expected, out)) {
        if (++spins == STOP_CHECK_SPINS) {
            spins = 0;
            if (st.stop_requested()) {
                return false;
            }
        }
        cpu_relax();
    }
    return true;
}

template <class Role> void VerbsSessionBase<Role>::post_recv(std::uint64_t slot) {
    // A SEND lands in the buffer its receive request names; a write with immediate needs none.
    ibv_sge sge{
        .addr = reinterpret_cast<std::uint64_t>(local_.addr) + ring_.slot_offset(slot),
        .length = ring_.stride,
        .lkey = local_.lkey,
    };
    ibv_recv_wr wr{
        .wr_id = slot,
        .sg_list = cuqr_write_imm_ ? nullptr : &sge,
        .num_sge = cuqr_write_imm_ ? 0 : 1,
    };
    ibv_recv_wr *bad = nullptr;
    const int rc = ibv_post_recv(recv_qp(), &wr, &bad);
    TP_CHECK(rc == 0, "ibv_post_recv rc=%d (%s)", rc, std::strerror(rc));
}

template <class Role>
bool VerbsSessionBase<Role>::wait_recv(std::uint64_t cursor, std::stop_token st) {
    const std::uint64_t slot = cursor & (ring_.slots - 1);
    ibv_wc wc{};
    std::uint32_t spins = 0;
    while (ibv_poll_cq(recv_cq(), 1, &wc) == 0) {
        if (++spins == STOP_CHECK_SPINS) {
            spins = 0;
            if (st.stop_requested()) {
                return false;
            }
        }
        cpu_relax();
    }
    TP_CHECK(wc.status == IBV_WC_SUCCESS, "%s: receive completion status=%d", backend_, wc.status);
    // Receive requests complete in the order they were posted, so a SEND filled `slot`; a write
    // names its slot in the immediate.
    const std::uint64_t got = cuqr_write_imm_ ? ntohl(wc.imm_data) : wc.wr_id;
    TP_CHECK(got == slot, "%s: a CUQR message arrived for slot %llu, expected slot %llu", backend_,
             static_cast<unsigned long long>(got), static_cast<unsigned long long>(slot));
    // Repost now: the peer reaches this slot again only after wrapping the whole ring, which it
    // cannot do before this message is answered.
    post_recv(slot);
    return true;
}

template class VerbsSessionBase<ControllerSession>;
template class VerbsSessionBase<CoprocessorSession>;

} // namespace catalyst::transport::common
