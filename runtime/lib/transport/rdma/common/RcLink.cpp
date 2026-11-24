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

#include "RcLink.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>

#include "Error.hpp"
#include "Handshake.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::common {

namespace {
// RDMA device port; rxe0 is single-port -> 1.
constexpr std::uint8_t PORT = 1;
// Max CQEs taken per non-blocking batch reap.
constexpr int REAP_BATCH = 16;
} // namespace

RcLink::RcLink(std::string dev, int gid_idx, const char *backend, Side side,
               frame::RingGeometry ring, QpShape fwd_shape)
    : PeerLink(Peer::CATALYST, std::move(dev), gid_idx, ring), backend_(backend), side_(side),
      fwd_shape_(fwd_shape) {}

void RcLink::connect(const ConnectInfo &info) {
    ctx_ = std::make_shared<Context>(dev_name_);
    pd_ = std::make_shared<ProtectionDomain>(ctx_);
    constexpr QpShape bwd = DEFAULT_QP_SHAPE;
    fwd_cq_ = std::make_shared<CompletionQueue>(ctx_, fwd_shape_.cq_depth);
    bwd_cq_ = std::make_shared<CompletionQueue>(ctx_, bwd.cq_depth);
    fwd_qp_ = std::make_shared<QueuePair>(pd_, fwd_cq_, fwd_cq_, fwd_shape_.sq_depth,
                                          fwd_shape_.inline_max);
    bwd_qp_ = std::make_shared<QueuePair>(pd_, bwd_cq_, bwd_cq_, bwd.sq_depth, bwd.inline_max);
    fwd_qp_->to_init(PORT);
    bwd_qp_->to_init(PORT);
    active_mtu_ = static_cast<std::uint32_t>(ctx_->port_attr(PORT).active_mtu);
    mygid_ = ctx_->gid(PORT, gid_idx_);
    oob_fd_ = side_ == Side::Coprocessor ? tcp_listen_accept(info.oob_port)
                                         : tcp_connect(info.peer.c_str(), info.oob_port);
}

PeerRef RcLink::exchange(const MemRegion &local) {
    HandshakeMsg my{
        .fwd = {.qpn = fwd_qp_->qpn(), .psn = 0},
        .bwd = {.qpn = bwd_qp_->qpn(), .psn = 0},
        .mr_vaddr = reinterpret_cast<std::uint64_t>(local.addr),
        .mr_rkey = local.rkey,
        .mtu_enum = active_mtu_,
        .frame_mode = static_cast<std::uint32_t>(ring_.mode),
        .frame_stride = ring_.stride,
        .frame_slots = ring_.slots,
    };
    // gid is a 16-byte array copied from the local GID after the aggregate
    // init.
    std::memcpy(my.fwd.gid, &mygid_, sizeof(mygid_));
    std::memcpy(my.bwd.gid, &mygid_, sizeof(mygid_));
    HandshakeMsg peer{};
    const int fd = oob_fd_.get();
    if (side_ == Side::Coprocessor) {
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
    const std::uint32_t mtu = std::min(my.mtu_enum, peer.mtu_enum);
    fwd_qp_->to_rtr(peer.fwd.qpn, peer.fwd.psn, peer.fwd.gid, gid_idx_, PORT, mtu);
    bwd_qp_->to_rtr(peer.bwd.qpn, peer.bwd.psn, peer.bwd.gid, gid_idx_, PORT, mtu);
    fwd_qp_->to_rts(my.fwd.psn);
    bwd_qp_->to_rts(my.bwd.psn);
    return PeerRef{
        .rkey = peer.mr_rkey,
        .remote_addr = peer.mr_vaddr,
        .size = ring_.region_bytes(),
    };
}

void RcLink::establish(const MemRegion &local, const PeerRef &peer) {
    TP_CHECK(local.size >= ring_.region_bytes(), "region too small for ring: %zu < %zu",
             static_cast<std::size_t>(local.size), ring_.region_bytes());
    peer_ring_ = peer;
    send_buf_ = MemoryRegion::alloc_host(pd_, frame::frame_bytes(ring_), ring_.stride,
                                         MemAccess::LOCAL_WRITE);
}

template <frame::Mode M> void RcLink::post(std::uint32_t seq) {
    const std::uint32_t cursor = seq - 1;
    // A BL16 frame's size and stride are fixed, so they are constants here; the slot count is the
    // session's.
    const frame::RingGeometry &ring = M == frame::Mode::BL16 ? frame::BL16_RING : ring_;
    const std::size_t offset = static_cast<std::size_t>(cursor & (ring_.slots - 1)) * ring.stride;
    void *frame = send_frame(); // staged by the caller
    frame::seal<M>(ring, frame, seq);
    const std::size_t bytes = frame::frame_bytes<M>(ring);
    const bool from_fwd = side_ == Side::Controller;
    // A frame larger than the queue pair's inline capacity goes from the registered buffer.
    const int inline_max = from_fwd ? fwd_shape_.inline_max : DEFAULT_QP_SHAPE.inline_max;
    const bool inline_data =
        M == frame::Mode::BL16 || bytes <= static_cast<std::size_t>(inline_max);
    const bool signaled = cursor % SIGNAL_EVERY == 0;
    ibv_sge sge{
        .addr = reinterpret_cast<std::uint64_t>(frame),
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
    wr.wr.rdma.remote_addr = peer_ring_.remote_addr + offset;
    wr.wr.rdma.rkey = peer_ring_.rkey;
    ibv_send_wr *bad = nullptr;
    const int rc = ibv_post_send(from_fwd ? fwd_qp_->get() : bwd_qp_->get(), &wr, &bad);
    TP_CHECK(rc == 0, "ibv_post_send rc=%d (%s)", rc, std::strerror(rc));
    if (signaled) {
        ++outstanding_;
        reap(/*drain=*/false); // lazy: free the send queue
    }
}

void RcLink::drain() {
    if (fwd_cq_) {
        reap(/*drain=*/true);
    }
    outstanding_ = 0;
}

// Non-blocking batch reap of the send queue's CQ: take whatever completions are ready (up to
// REAP_BATCH) and decrement `outstanding_`. With drain=true, keep polling until every signaled
// send has completed (teardown), guarded so a lost CQE can't hang the join.
void RcLink::reap(bool drain) {
    ibv_cq *cq = side_ == Side::Controller ? fwd_cq_->get() : bwd_cq_->get();
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
            --outstanding_;
        }
    } while (drain && outstanding_ > 0);
}

template void RcLink::post<frame::Mode::BL16>(std::uint32_t);
template void RcLink::post<frame::Mode::BLV>(std::uint32_t);

} // namespace catalyst::transport::common
