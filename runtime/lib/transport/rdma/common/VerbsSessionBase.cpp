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
                                         QpShape fwd_shape, frame::RingGeometry ring)
    : dev_name_(std::move(dev)), gid_idx_(gid_idx), backend_(backend), fwd_shape_(fwd_shape),
      ring_(ring) {}

template <class Role> int VerbsSessionBase<Role>::connect(const ConnectInfo &info) {
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
    const bool listen = oob_listens();
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
        .frame_slots = ring_.slots,
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
    const std::size_t bytes = frame::frame_bytes<M>(ring);
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
    ring_.mode == frame::Mode::BL16
        ? post_write<frame::Mode::BL16>(qp, cursor, inline_data, signaled)
        : post_write<frame::Mode::BLV>(qp, cursor, inline_data, signaled);
}

template class VerbsSessionBase<ControllerSession>;
template class VerbsSessionBase<CoprocessorSession>;

template void VerbsSessionBase<CoprocessorSession>::post_write<frame::Mode::BL16>(ibv_qp *,
                                                                                  std::uint64_t,
                                                                                  bool, bool);
template void VerbsSessionBase<CoprocessorSession>::post_write<frame::Mode::BLV>(ibv_qp *,
                                                                                 std::uint64_t,
                                                                                 bool, bool);
template bool VerbsSessionBase<CoprocessorSession>::poll_message_arrival<frame::Mode::BL16>(
    std::uint64_t, std::stop_token, frame::FrameView &);
template bool VerbsSessionBase<CoprocessorSession>::poll_message_arrival<frame::Mode::BLV>(
    std::uint64_t, std::stop_token, frame::FrameView &);
template bool VerbsSessionBase<ControllerSession>::poll_message_arrival<frame::Mode::BL16>(
    std::uint64_t, std::stop_token, frame::FrameView &);
template bool VerbsSessionBase<ControllerSession>::poll_message_arrival<frame::Mode::BLV>(
    std::uint64_t, std::stop_token, frame::FrameView &);

} // namespace catalyst::transport::common
