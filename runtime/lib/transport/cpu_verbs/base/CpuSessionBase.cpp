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

#include "CpuSessionBase.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

#include "Error.hpp"
#include "Handshake.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::cpu_verbs {
using namespace catalyst::transport;
using namespace catalyst::transport::common;

namespace {
// RDMA device port; rxe0 is single-port -> 1.
constexpr std::uint8_t PORT = 1;
// Max CQEs taken per non-blocking batch reap.
constexpr int REAP_BATCH = 16;
// QP inline capacity for inline sends (>= sizeof(Payload), 16 B).
constexpr int INLINE_MAX = 256;
constexpr int SQ_DEPTH = 4096;
constexpr int CQ_DEPTH = 4096;
// Page alignment for registered host buffers.
constexpr std::size_t PAGE_ALIGN = 4096;
} // namespace

CpuSessionBase::CpuSessionBase(std::string dev, int gid_idx)
    : dev_name_(std::move(dev)), gid_idx_(gid_idx)
{
}

int CpuSessionBase::connect(const ConnectInfo &info)
{
    ctx_ = std::make_shared<Context>(dev_name_);
    pd_ = std::make_shared<ProtectionDomain>(ctx_);
    fwd_cq_ = std::make_shared<CompletionQueue>(ctx_, CQ_DEPTH);
    bwd_cq_ = std::make_shared<CompletionQueue>(ctx_, CQ_DEPTH);
    fwd_qp_ = std::make_shared<QueuePair>(pd_, fwd_cq_, fwd_cq_, SQ_DEPTH, INLINE_MAX);
    bwd_qp_ = std::make_shared<QueuePair>(pd_, bwd_cq_, bwd_cq_, SQ_DEPTH, INLINE_MAX);
    fwd_qp_->to_init(PORT);
    bwd_qp_->to_init(PORT);
    active_mtu_ = static_cast<std::uint32_t>(ctx_->port_attr(PORT).active_mtu);
    mygid_ = ctx_->gid(PORT, gid_idx_);
    const bool listen = oob_listens();
    oob_fd_ =
        listen ? tcp_listen_accept(info.oob_port) : tcp_connect(info.peer.c_str(), info.oob_port);
    return 0;
}

MemRegion CpuSessionBase::alloc_memory(std::size_t size, MemKind kind)
{
    RDMA_CHECK(kind == MemKind::CpuRam, "cpu_libibverbs: only MemKind::CpuRam supported");
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

PeerRef CpuSessionBase::exchange_keys(const MemRegion &local)
{
    HandshakeMsg my{
        .fwd = {.qpn = fwd_qp_->qpn(), .psn = 0},
        .bwd = {.qpn = bwd_qp_->qpn(), .psn = 0},
        .mr_vaddr = reinterpret_cast<std::uint64_t>(local.addr),
        .mr_rkey = local.rkey,
        .mtu_enum = active_mtu_,
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
    }
    else {
        recv_exact(fd, &peer, sizeof(peer));
        send_exact(fd, &my, sizeof(my));
    }
    const std::uint32_t mtu = std::min(my.mtu_enum, peer.mtu_enum);
    fwd_qp_->to_rtr(peer.fwd.qpn, peer.fwd.psn, peer.fwd.gid, gid_idx_, PORT, mtu);
    bwd_qp_->to_rtr(peer.bwd.qpn, peer.bwd.psn, peer.bwd.gid, gid_idx_, PORT, mtu);
    fwd_qp_->to_rts(my.fwd.psn);
    bwd_qp_->to_rts(my.bwd.psn);
    return PeerRef{
        .rkey = peer.mr_rkey,
        .remote_addr = peer.mr_vaddr,
        .size = REGION_BYTES,
    };
}

void CpuSessionBase::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                       const PeerRef &peer)
{
    RDMA_CHECK(local.size >= REGION_BYTES, "region too small for ring: %zu < %zu", local.size,
               REGION_BYTES);
    desc_ = desc;
    local_ = local;
    peer_ = peer;
    send_buf_ = common::MemoryRegion::alloc_host(pd_, sizeof(Payload), alignof(PayloadSlot),
                                                 common::MemAccess::LOCAL_WRITE);
}

void CpuSessionBase::post_write(ibv_qp *qp, std::uint64_t cursor, bool inline_data, bool signaled)
{
    auto *send = send_payload(); // value already written by the caller
    send->seq_num = static_cast<std::uint32_t>(cursor + 1);
    send->pad = 0;
    ibv_sge sge{
        .addr = reinterpret_cast<std::uint64_t>(send),
        .length = sizeof(Payload), // 16 B on the wire, into slot's first 16 B
        .lkey = send_buf_->lkey(),
    };
    ibv_send_wr wr{
        .sg_list = &sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = static_cast<unsigned>((inline_data ? IBV_SEND_INLINE : 0) |
                                            (signaled ? IBV_SEND_SIGNALED : 0)),
    };
    wr.wr.rdma.remote_addr =
        peer_.remote_addr + (cursor & (K_RING_SLOTS - 1)) * sizeof(PayloadSlot);
    wr.wr.rdma.rkey = peer_.rkey;
    ibv_send_wr *bad = nullptr;
    RDMA_CHECK(ibv_post_send(qp, &wr, &bad) == 0, "ibv_post_send");
}

// Non-blocking batch reap of `cq`: take whatever completions are ready (up to
// REAP_BATCH) and decrement `outstanding`. With drain=true, keep polling until
// every signaled send has completed (teardown), guarded so a lost CQE can't
// hang the join.
void CpuSessionBase::reap(ibv_cq *cq, int &outstanding, bool drain)
{
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
            RDMA_CHECK(wc[k].status == IBV_WC_SUCCESS, "CQE status=%d", wc[k].status);
            --outstanding;
        }
    } while (drain && outstanding > 0);
}

Payload *CpuSessionBase::poll_message_arrival(std::uint64_t cursor, std::stop_token st)
{
    // Slots are reused (K_RING_SLOTS is a power of two). The ring contains
    // 64 B PayloadSlots; the peer writes the 16 B Payload into each slot's
    // head.
    auto *ring = reinterpret_cast<PayloadSlot *>(local_.addr);
    Payload *slot = &ring[cursor & (K_RING_SLOTS - 1)].p;
    // Poll seq_num with acquire ordering: once it updates, value (written
    // before it in the single RDMA_WRITE) is present, and the acquire keeps its
    // read from being hoisted ahead.
    std::atomic_ref<std::uint32_t> seq_ref(slot->seq_num);
    const auto expected = static_cast<std::uint32_t>(cursor + 1);
    while (seq_ref.load(std::memory_order_acquire) != expected) {
        if (st.stop_requested()) {
            return nullptr;
        }
        std::this_thread::yield();
    }
    return slot;
}

void CpuSessionBase::start()
{
    stop();
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    completed_.store(0, std::memory_order_relaxed);
    last_word_.store(0, std::memory_order_relaxed);
    // jthread injects the stop_token. A data-path RDMA_CHECK throws RdmaError;
    // it must not escape the thread function (that would std::terminate).
    // Capture it into error_ and publish via failed_ (release) so collect()
    // can rethrow the real exception.
    auto body = [this](std::stop_token st) {
        try {
            run(st);
        }
        catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    };
    engine_ = std::jthread(body);
}

int CpuSessionBase::collect(void *const *replies, const std::uint64_t *replies_bytes, std::size_t n)
{
    while (completed_.load(std::memory_order_acquire) == 0) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_); // surface the engine's real error
        }
        if (!engine_.joinable() || engine_.get_stop_token().stop_requested()) {
            break;
        }
        std::this_thread::yield();
    }
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }
    // Stopped before any round completed -> no data (non-exceptional).
    if (completed_.load(std::memory_order_acquire) == 0) {
        return -1;
    }
    if (n > 0 && replies && replies[0]) {
        const std::uint64_t w = last_word_.load(std::memory_order_relaxed);
        const std::size_t nb =
            replies_bytes ? std::min<std::size_t>(replies_bytes[0], sizeof(w)) : sizeof(w);
        std::memcpy(replies[0], &w, nb);
    }
    return 0;
}

void CpuSessionBase::stop()
{
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
}

} // namespace catalyst::transport::cpu_verbs
