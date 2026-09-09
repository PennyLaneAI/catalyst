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

#include "GpuCoprocessorSession.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

#include "Error.hpp"
#include "GpuLaunchers.hpp"
#include "Handshake.hpp"
#include "RealtimeThread.hpp"

#include <unistd.h>

namespace catalyst::transport::gpu_verbs {
using namespace catalyst::transport;
using namespace catalyst::transport::common;
using namespace catalyst::transport::coproc; // GpuRuntime, HandoffSlot, launchers

namespace {
// RDMA device port (single-port NIC -> 1).
constexpr std::uint8_t PORT = 1;
// Max CQEs taken per non-blocking batch reap.
constexpr int REAP_BATCH = 16;
constexpr int SQ_DEPTH = 4096;
constexpr int CQ_DEPTH = 4096;
// QP inline capacity for inline sends (>= sizeof(Payload), 16 B).
constexpr int INLINE_MAX = 256;

// fwd qp is for one-sided RDMA_WRITE from controller to coprocessor,
// so the CQ/SQ depth does not matter here for the coprocessor session.
constexpr int FWD_SQ_DEPTH = 16;
constexpr int FWD_CQ_DEPTH = 256;
constexpr int FWD_INLINE_MAX = 0;
} // namespace

GpuCoprocessorSession::GpuCoprocessorSession(std::string dev, int gid_idx, int gpu_device)
    : dev_name_(std::move(dev)), gid_idx_(gid_idx), gpu_(gpu_device) {}

int GpuCoprocessorSession::connect(const ConnectInfo &info) {
    ctx_ = std::make_shared<Context>(dev_name_);
    pd_ = std::make_shared<ProtectionDomain>(ctx_);
    fwd_cq_ = std::make_shared<CompletionQueue>(ctx_, FWD_CQ_DEPTH);
    bwd_cq_ = std::make_shared<CompletionQueue>(ctx_, CQ_DEPTH);
    fwd_qp_ = std::make_shared<QueuePair>(pd_, fwd_cq_, fwd_cq_, FWD_SQ_DEPTH, FWD_INLINE_MAX);
    bwd_qp_ = std::make_shared<QueuePair>(pd_, bwd_cq_, bwd_cq_, SQ_DEPTH, INLINE_MAX);
    fwd_qp_->to_init(PORT);
    bwd_qp_->to_init(PORT);
    // Cache local device attrs; QP+MR handshake is deferred to exchange_keys.
    active_mtu_ = static_cast<std::uint32_t>(ctx_->port_attr(PORT).active_mtu);
    mygid_ = ctx_->gid(PORT, gid_idx_);
    oob_fd_ = tcp_listen_accept(info.oob_port); // coprocessor listens
    return 0;
}

MemRegion GpuCoprocessorSession::alloc_memory(std::size_t size, MemKind kind) {
    TP_CHECK(kind == MemKind::GpuHbm, "gpu_verbs: only MemKind::GpuHbm supported");
    hbm_ = gpu_.alloc_hbm_ring(size);
#ifdef TRANSPORT_GPU_STUB
    // The host-memory GpuRuntime reports no dma-buf fd, so register the pages directly. Compiled
    // only in the stub build; on a real GPU the export always yields an fd and a silent fallback
    // here would hide a failed one.
    TP_CHECK(hbm_.dmabuf_fd < 0, "gpu stub: unexpected dma-buf fd %d", hbm_.dmabuf_fd);
    hbm_ring_.emplace(pd_, hbm_.ptr, hbm_.size, MemAccess::REMOTE_WRITE);
#else
    hbm_ring_.emplace(pd_, /*offset=*/static_cast<std::uint64_t>(0), hbm_.size,
                      /*iova=*/reinterpret_cast<std::uint64_t>(hbm_.ptr), hbm_.dmabuf_fd,
                      MemAccess::REMOTE_WRITE);
    ::close(hbm_.dmabuf_fd); // MR holds its own reference
    hbm_.dmabuf_fd = -1;
#endif
    return MemRegion{
        .addr = hbm_.ptr,
        .size = hbm_.size,
        .lkey = hbm_ring_->lkey(),
        .rkey = hbm_ring_->rkey(),
        .kind = MemKind::GpuHbm,
    };
}

PeerRef GpuCoprocessorSession::exchange_keys(const MemRegion &local) {
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
    send_exact(fd, &my, sizeof(my)); // coprocessor listens -> sends first
    recv_exact(fd, &peer, sizeof(peer));
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

void GpuCoprocessorSession::establish_channel(const ChannelDesc &desc, const MemRegion &local,
                                              const PeerRef &peer) {
    TP_CHECK(desc.transport == "rdma", "gpu_verbs: only transport \"rdma\" supported");
    TP_CHECK(local.size >= REGION_BYTES, "region too small for ring: %zu < %zu",
             static_cast<std::size_t>(local.size), REGION_BYTES);
    desc_ = desc;
    local_ = local;
    peer_ = peer;
    handoff_ = gpu_.alloc_handoff(K_RING_SLOTS);
    reply_buf_ = common::MemoryRegion::alloc_host(pd_, sizeof(Payload), alignof(PayloadSlot),
                                                  common::MemAccess::LOCAL_WRITE);
}

void GpuCoprocessorSession::set_coprocessor_launcher(CoprocessorLauncherFn fn, void *ctx) {
    // Store the launcher; start() invokes it once to launch the on-device decode
    // kernel. nullptr selects the built-in echo launcher.
    coproc_launcher_ = fn;
    coproc_ctx_ = ctx;
}

bool GpuCoprocessorSession::post_inline(std::uint64_t cursor) {
    auto *reply = reinterpret_cast<Payload *>(reply_buf_->addr());
    reply->value = static_cast<std::uint64_t>(last_word_.load(std::memory_order_relaxed));
    // TODO: echo the request's id by forwarding it through.
    // Currently not sent to reduce Handoff size.
    reply->decoder_id = 0;
    reply->seq_num = static_cast<std::uint32_t>(cursor + 1);
    ibv_sge sge{
        .addr = reinterpret_cast<std::uint64_t>(reply),
        .length = sizeof(Payload),
        .lkey = reply_buf_->lkey(),
    };
    const bool signaled = (cursor % SIGNAL_EVERY == 0);
    ibv_send_wr wr{
        .sg_list = &sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = static_cast<unsigned>(IBV_SEND_INLINE | (signaled ? IBV_SEND_SIGNALED : 0)),
    };
    wr.wr.rdma.remote_addr =
        peer_.remote_addr + (cursor & (K_RING_SLOTS - 1)) * sizeof(PayloadSlot);
    wr.wr.rdma.rkey = peer_.rkey;
    ibv_send_wr *bad = nullptr;
    const int rc = ibv_post_send(bwd_qp_->get(), &wr, &bad);
    TP_CHECK(rc == 0, "ibv_post_send rc=%d (%s)", rc, std::strerror(rc));
    return signaled;
}

// Non-blocking batch reap of the bwd CQ: take whatever completions are ready
// (up to REAP_BATCH at a time) and decrement `outstanding`. With drain=true,
// keep polling until every signaled send has completed (teardown), guarded so a
// lost completion can't hang the join forever.
void GpuCoprocessorSession::reap_bwd(int &outstanding, bool drain) {
    std::array<ibv_wc, REAP_BATCH> wc;
    int empty = 0;
    constexpr int DRAIN_MAX_EMPTY = 1000000;
    do {
        int n = ibv_poll_cq(bwd_cq_->get(), static_cast<int>(wc.size()), wc.data());
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
            TP_CHECK(wc[k].status == IBV_WC_SUCCESS, "bwd CQE status=%d", wc[k].status);
            --outstanding;
        }
    } while (drain && outstanding > 0);
}

// Coprocessor: wait for the fused kernel to publish each shot's correction into
// the host-mapped handoff ring, then post it inline on the bwd QP. Sends are
// selectively signaled (1 in SIGNAL_EVERY) and the bwd CQ is reaped in batches
// at those points, so completions stay off the per-shot critical path.
void GpuCoprocessorSession::run_coprocessor(std::stop_token st) {
    pin_thread(pin_cpu_, pin_realtime_);
    int signaled_outstanding = 0;
    for (std::uint64_t c = 0; !st.stop_requested(); ++c) {
        const std::uint32_t expect = static_cast<std::uint32_t>(c + 1);
        std::uint32_t index = c & (K_RING_SLOTS - 1);
        volatile std::uint32_t *sptr = &handoff_.host[index].seq;
        std::uint32_t spins = 0;
        while (*sptr != expect) {
            if (++spins == STOP_CHECK_SPINS) {
                spins = 0;
                if (st.stop_requested()) {
                    reap_bwd(signaled_outstanding, /*drain=*/true);
                    return;
                }
            }
            cpu_relax();
        }
        // seq observed -> the kernel published correction+seq together in one
        // 16 B store followed by a single system fence (no fence between the
        // two fields); larger replies instead use write, fence, seq, fence.
        std::atomic_thread_fence(std::memory_order_acquire);
        last_word_.store(handoff_.host[index].correction, std::memory_order_relaxed);
        if (post_inline(c)) {
            ++signaled_outstanding;
            reap_bwd(signaled_outstanding, /*drain=*/false); // lazy batch reap
        }
        completed_.fetch_add(1, std::memory_order_release);
    }
    reap_bwd(signaled_outstanding, /*drain=*/true);
}

void GpuCoprocessorSession::start() {
    stop();
    failed_.store(false, std::memory_order_relaxed);
    error_ = nullptr;
    completed_.store(0, std::memory_order_relaxed);
    last_word_.store(0, std::memory_order_relaxed);
    if (handoff_.stop_host) {
        *handoff_.stop_host = 0;
    }
    // Launch the persistent kernel once via the bound launcher (nullptr selects
    // the built-in echo launcher). The kernel runs until the stop flag is set
    // (total = 0); the session keeps owning the stream/stop/sync + reply thread.
    // CoprocLaunchDesc is the HIP-free bridge to the launcher.
    CoprocLaunchDesc L{
        .ring = hbm_.ptr,
        .ring_slots = K_RING_SLOTS,
        .handoff = handoff_.dev,
        .stop = handoff_.stop_dev,
        .total = 0,
        .stream = gpu_.stream(),
    };
    CoprocessorLauncherFn launch = coproc_launcher_ ? coproc_launcher_ : &default_echo_launcher;
    const int lrc = launch(&L, coproc_ctx_);
    TP_CHECK(lrc == 0, "gpu_verbs: coprocessor kernel launch failed");
    // A data-path TP_CHECK throws TransportError; capture it here so it doesn't
    // escape the thread (which would std::terminate), and publish via failed_.
    auto body = [this](std::stop_token st) {
        try {
            run_coprocessor(st);
        } catch (...) {
            error_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
        }
    };
    engine_ = std::jthread(body);
}

int GpuCoprocessorSession::collect(void *const *outputs, const std::uint64_t *output_bytes,
                                   std::size_t n) {
    while (completed_.load(std::memory_order_acquire) == 0) {
        if (failed_.load(std::memory_order_acquire)) {
            std::rethrow_exception(error_);
        }
        if (!engine_.joinable() || engine_.get_stop_token().stop_requested()) {
            break;
        }
        std::this_thread::yield();
    }
    if (failed_.load(std::memory_order_acquire)) {
        std::rethrow_exception(error_);
    }
    if (completed_.load(std::memory_order_acquire) == 0) {
        return -1;
    }
    if (n > 0 && outputs && outputs[0]) {
        const std::int64_t w = last_word_.load(std::memory_order_relaxed);
        const std::size_t cap = output_bytes ? output_bytes[0] : sizeof(w);
        TP_CHECK(cap <= sizeof(w), "output capacity (%zu) exceeds the %zu B payload", cap,
                 sizeof(w));
        std::memcpy(outputs[0], &w, cap);
    }
    return 0;
}

void GpuCoprocessorSession::stop() {
    if (handoff_.stop_host) {
        *handoff_.stop_host = 1; // let the fused kernel exit its loop
    }
    if (engine_.joinable()) {
        engine_.request_stop();
        engine_.join();
    }
    gpu_.sync(); // wait for the kernel to observe stop and finish
}

} // namespace catalyst::transport::gpu_verbs
