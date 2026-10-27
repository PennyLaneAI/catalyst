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
#include <chrono>
#include <cstring>
#include <memory>
#include <utility>

#include "Error.hpp"
#include "Handshake.hpp"
#include "RealtimeThread.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::common {
using namespace catalyst::transport;

namespace {
// RDMA device port; rxe0 is single-port -> 1.
constexpr std::uint8_t PORT = 1;
// Max CQEs taken per non-blocking batch reap.
constexpr int REAP_BATCH = 16;
// Page alignment for registered host buffers.
constexpr std::size_t PAGE_ALIGN = 4096;
// How often a CUQR receive wait checks for a stop. Each check of the completion queue costs far
// more than the memory poll STOP_CHECK_SPINS is sized for, so this is much smaller.
constexpr std::uint32_t RECV_STOP_CHECK_POLLS = 4096;
} // namespace

template <class Role>
VerbsSessionBase<Role>::VerbsSessionBase(std::string dev, int gid_idx, const char *backend,
                                         QpShape fwd_shape, frame::RingGeometry ring,
                                         CuqrConfig cuqr)
    : dev_name_(std::move(dev)), gid_idx_(gid_idx), backend_(backend), fwd_shape_(fwd_shape),
      ring_(ring), cuqr_(cuqr) {}

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
    // A cudaq-realtime peer uses one UC queue pair both ways: this end's sending one (the
    // controller's fwd, the coprocessor's bwd) also receives, into the CQ the other would use.
    const bool cudaq = cuqr_.cudaq_peer;
    TP_CHECK(!cudaq || cuqr(), "%s: peer=cudaq needs CUQR frames (frame=cuqr)", backend_);
    const bool listen = oob_listens();
    fwd_qp_ = std::make_shared<QueuePair>(pd_, fwd_cq_, cudaq && !listen ? bwd_cq_ : fwd_cq_,
                                          fwd_shape_.sq_depth, fwd_shape_.inline_max, recv_wr,
                                          /*uc=*/cudaq && !listen);
    bwd_qp_ = std::make_shared<QueuePair>(pd_, bwd_cq_, cudaq && listen ? fwd_cq_ : bwd_cq_,
                                          bwd.sq_depth, bwd.inline_max, recv_wr,
                                          /*uc=*/cudaq && listen);
    fwd_qp_->to_init(PORT);
    bwd_qp_->to_init(PORT);
    active_mtu_ = static_cast<std::uint32_t>(ctx_->port_attr(PORT).active_mtu);
    mygid_ = ctx_->gid(PORT, gid_idx_);
    oob_fd_ =
        listen ? tcp_listen_accept(info.oob_port) : tcp_connect(info.peer.c_str(), info.oob_port);
    return 0;
}

template <class Role>
MemRegion VerbsSessionBase<Role>::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(kind == MemKind::CpuRam, "cpu_libibverbs: only MemKind::CpuRam supported");
    // The peer writes a ring of frames into the region, which may be larger than asked for.
    size = std::max(size, ring_.region_bytes());
    caller_memory_regions_.push_back(MemoryRegion::alloc_host(
        pd_, size, PAGE_ALIGN, MemAccess::LOCAL_WRITE | MemAccess::REMOTE_WRITE,
        /*zero_iova=*/cuqr_.cudaq_peer));
    const MemoryRegion &mr = caller_memory_regions_.back();
    MemRegion region{
        .addr = mr.addr(),
        .size = size,
        .lkey = mr.lkey(),
        .rkey = mr.rkey(),
        .kind = MemKind::CpuRam,
    };
    if (cuqr_.cudaq_peer) {
        // The peer addresses the ring from 0, but the driver may copy a small message into a
        // receive buffer with the CPU (mlx5 delivers it through the completion entry), so receive
        // requests name the ring by its address, through a second, local registration.
        caller_memory_regions_.emplace_back(pd_, region.addr, size, MemAccess::LOCAL_WRITE);
        region.lkey = caller_memory_regions_.back().lkey();
    }
    return region;
}

template <class Role> PeerRef VerbsSessionBase<Role>::exchange_keys(const MemRegion &local) {
    if (cuqr_.cudaq_peer) {
        return exchange_keys_with_cudaq(local);
    }
    HandshakeMsg my{
        .fwd = {.qpn = fwd_qp_->qpn(), .psn = 0},
        .bwd = {.qpn = bwd_qp_->qpn(), .psn = 0},
        .mr_vaddr = reinterpret_cast<std::uint64_t>(local.addr),
        .mr_rkey = local.rkey,
        .mtu_enum = active_mtu_,
        .frame_mode = static_cast<std::uint32_t>(ring_.mode),
        .frame_stride = ring_.stride,
        .frame_slots = ring_.slots,
        .cuqr_tx = cuqr_.write_imm ? 1U : 0U,
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
    TP_CHECK(peer.frame_mode == my.frame_mode && peer.frame_stride == my.frame_stride &&
                 peer.frame_slots == my.frame_slots,
             "%s: the peer's frames (mode %u, %u slots of %u B) differ from this end's (mode %u, "
             "%u slots of %u B); both ends need the same frame config",
             backend_, peer.frame_mode, peer.frame_slots, peer.frame_stride, my.frame_mode,
             my.frame_slots, my.frame_stride);
    TP_CHECK(!cuqr() || peer.cuqr_tx == my.cuqr_tx,
             "%s: the peer completes CUQR frames with %s and this end with %s; set the same "
             "cuqr_tx on both",
             backend_, peer.cuqr_tx != 0 ? "write_imm" : "send",
             my.cuqr_tx != 0 ? "write_imm" : "send");
    const std::uint32_t mtu = std::min(my.mtu_enum, peer.mtu_enum);
    fwd_qp_->to_rtr(peer.fwd.qpn, peer.fwd.psn, peer.fwd.gid, gid_idx_, PORT, mtu);
    bwd_qp_->to_rtr(peer.bwd.qpn, peer.bwd.psn, peer.bwd.gid, gid_idx_, PORT, mtu);
    if (cuqr()) { // receive requests are in place before the peer can send
        post_recvs(local);
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
template <frame::Mode M>
void VerbsSessionBase<Role>::post_write(ibv_qp *qp, std::uint64_t cursor, bool inline_data,
                                        bool signaled) {
    // A BL16 frame's size and stride are fixed, so they are constants here; the slot count is the
    // session's.
    const frame::RingGeometry &ring = M == frame::Mode::BL16 ? frame::BL16_RING : ring_;
    const std::size_t offset = (cursor & (ring_.slots - 1)) * ring.stride;
    void *send = send_frame(); // staged by the caller
    frame::seal<M>(ring, send, static_cast<std::uint32_t>(cursor + 1));
    const std::size_t bytes =
        M == frame::Mode::CUQR ? frame::wire_bytes(ring, send) : frame::frame_bytes<M>(ring);
    if constexpr (M != frame::Mode::BL16) {
        // A frame larger than the queue pair's inline capacity goes from the registered buffer.
        const int inline_max =
            qp == fwd_qp_->get() ? fwd_shape_.inline_max : DEFAULT_QP_SHAPE.inline_max;
        inline_data = inline_data && bytes <= static_cast<std::size_t>(inline_max);
    }
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
    wr.wr.rdma.remote_addr = peer_.remote_addr + offset;
    wr.wr.rdma.rkey = peer_.rkey;
    if constexpr (M == frame::Mode::CUQR) {
        // A coprocessor answers a cudaq-realtime peer by SEND, as cudaq-realtime's own servers do:
        // its clients take a reply's slot from the order replies arrive in.
        if (cuqr_.write_imm && !(cuqr_.cudaq_peer && oob_listens())) {
            // the immediate names the slot written
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
template <frame::Mode M>
bool VerbsSessionBase<Role>::poll_message_arrival(std::uint64_t cursor, std::stop_token st,
                                                  frame::FrameView &out) {
    if constexpr (M == frame::Mode::CUQR) {
        if (!wait_recv(cursor, st)) {
            return false;
        }
        if (!frame::poll<M>(ring_, ring_slot(cursor), static_cast<std::uint32_t>(cursor + 1),
                            out)) {
            out = frame::FrameView{}; // not a frame: the caller drops it
        }
        return true;
    }
    // Slots are reused (the slot count is a power of two). The peer writes each frame into the
    // head of its slot in a single RDMA_WRITE, seq last, so once frame::poll sees the expected seq
    // the payload before it is present.
    const frame::RingGeometry &ring = M == frame::Mode::BL16 ? frame::BL16_RING : ring_;
    void *slot =
        static_cast<std::uint8_t *>(local_.addr) + (cursor & (ring_.slots - 1)) * ring.stride;
    const auto expected = static_cast<std::uint32_t>(cursor + 1);
    std::uint32_t spins = 0;
    while (!frame::poll<M>(ring, slot, expected, out)) {
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

template <class Role>
void VerbsSessionBase<Role>::post_frame(ibv_qp *qp, std::uint64_t cursor, bool inline_data,
                                        bool signaled) {
    CATALYST_FRAME_DISPATCH(ring_.mode, post_write, qp, cursor, inline_data, signaled);
}

// cudaq-realtime's rendezvous over the out-of-band socket, the caller (controller) first; then the
// one UC queue pair goes to RTS at MTU 4096, which cudaq-realtime fixes. Its ring is addressed from
// 0, like ours in this mode.
template <class Role>
PeerRef VerbsSessionBase<Role>::exchange_keys_with_cudaq(const MemRegion &local) {
    TP_CHECK(active_mtu_ >= IBV_MTU_4096,
             "%s: peer=cudaq needs an active MTU of 4096, which cudaq-realtime fixes; raise the "
             "port's netdev MTU to 4200 or more",
             backend_);
    QueuePair &qp = oob_listens() ? *bwd_qp_ : *fwd_qp_;
    const CudaqRendezvous my = cudaq_rendezvous(qp.qpn(), local.rkey, mygid_.raw);
    CudaqRendezvous peer{};
    const int fd = oob_fd_.get();
    // UC drops anything that reaches a queue pair before it can receive, so the listener (our
    // coprocessor) is ready to receive before it answers, and the caller only sends after that.
    if (!oob_listens()) {
        send_exact(fd, &my, sizeof(my));
    }
    recv_exact(fd, &peer, sizeof(peer));
    std::uint8_t gid[16];
    cudaq_peer_gid(peer, gid);
    qp.to_rtr(ntohl(peer.qpn), /*dest_psn=*/0, gid, gid_idx_, PORT, IBV_MTU_4096);
    post_recvs(local);
    qp.to_rts(/*sq_psn=*/0);
    if (oob_listens()) {
        send_exact(fd, &my, sizeof(my));
    }
    return PeerRef{
        .rkey = ntohl(peer.rkey),
        .remote_addr = 0,
        .size = ring_.region_bytes(),
    };
}

template <class Role> void VerbsSessionBase<Role>::post_recvs(const MemRegion &local) {
    local_ = local;
    for (std::uint64_t s = 0; s < ring_.slots; ++s) {
        post_recv(s);
    }
}

template <class Role> void VerbsSessionBase<Role>::post_recv(std::uint64_t slot) {
    // A SEND lands in the buffer its receive request names; a write with immediate ignores it. Both
    // can arrive whichever verb this end sends by: a cudaq-realtime server always answers by SEND.
    // A GPU ring takes writes with immediate only, as the driver may copy a small SEND into its
    // buffer with the CPU; with no buffer named, a SEND fails visibly instead.
    const bool gpu = local_.kind == MemKind::GpuHbm;
    ibv_sge sge{
        .addr = reinterpret_cast<std::uint64_t>(local_.addr) + ring_.slot_offset(slot),
        .length = ring_.stride,
        .lkey = local_.lkey,
    };
    ibv_recv_wr wr{
        .wr_id = slot,
        .sg_list = gpu ? nullptr : &sge,
        .num_sge = gpu ? 0 : 1,
    };
    ibv_recv_wr *bad = nullptr;
    const int rc = ibv_post_recv(recv_qp(), &wr, &bad);
    TP_CHECK(rc == 0, "ibv_post_recv rc=%d (%s)", rc, std::strerror(rc));
}

template <class Role>
bool VerbsSessionBase<Role>::wait_recv(std::uint64_t cursor, std::stop_token st) {
    const std::chrono::nanoseconds timeout =
        cuqr_.cudaq_peer && !oob_listens() ? std::chrono::seconds(5) : std::chrono::seconds(0);
    const std::uint64_t slot = cursor & (ring_.slots - 1);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    ibv_wc wc{};
    std::uint32_t spins = 0;
    while (ibv_poll_cq(recv_cq(), 1, &wc) == 0) {
        if (++spins == RECV_STOP_CHECK_POLLS) {
            spins = 0;
            if (st.stop_requested()) {
                return false;
            }
            TP_CHECK(timeout.count() == 0 || std::chrono::steady_clock::now() < deadline,
                     "%s: no reply within %lld ms; the peer's UC queue pair may have dropped the "
                     "message",
                     backend_,
                     static_cast<long long>(
                         std::chrono::duration_cast<std::chrono::milliseconds>(timeout).count()));
        }
        cpu_relax();
    }
    TP_CHECK(wc.status == IBV_WC_SUCCESS, "%s: receive completion status=%d", backend_, wc.status);
    // Receive requests complete in the order they were posted, so a SEND filled `slot`; a write
    // names its slot in the immediate.
    const std::uint64_t got =
        wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM ? ntohl(wc.imm_data) : wc.wr_id;
    TP_CHECK(got == slot, "%s: a CUQR message arrived for slot %llu, expected slot %llu", backend_,
             static_cast<unsigned long long>(got), static_cast<unsigned long long>(slot));
    // Repost now: the peer reaches this slot again only after wrapping the whole ring, which it
    // cannot do before this message is answered.
    post_recv(slot);
    return true;
}

template class VerbsSessionBase<ControllerSession>;
template class VerbsSessionBase<CoprocessorSession>;
#define INSTANTIATE_FOR_MODE(Role, M)                                                              \
    template void VerbsSessionBase<Role>::post_write<M>(ibv_qp *, std::uint64_t, bool, bool);      \
    template bool VerbsSessionBase<Role>::poll_message_arrival<M>(std::uint64_t, std::stop_token,  \
                                                                  frame::FrameView &);
#define INSTANTIATE_FOR_ROLE(Role)                                                                 \
    INSTANTIATE_FOR_MODE(Role, frame::Mode::BL16)                                                  \
    INSTANTIATE_FOR_MODE(Role, frame::Mode::BLV)                                                   \
    INSTANTIATE_FOR_MODE(Role, frame::Mode::CUQR)
INSTANTIATE_FOR_ROLE(ControllerSession)
INSTANTIATE_FOR_ROLE(CoprocessorSession)
#undef INSTANTIATE_FOR_ROLE
#undef INSTANTIATE_FOR_MODE

} // namespace catalyst::transport::common
