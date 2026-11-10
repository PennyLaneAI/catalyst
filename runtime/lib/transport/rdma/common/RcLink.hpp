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

/**
 * @file
 * One end of the RC wire, which every catalyst controller speaks: two RC queue pairs, fwd
 * (controller to coprocessor) and bwd (coprocessor to controller), connected over the out-of-band
 * socket with a HandshakeMsg. Each end owns a ring of frames the other RDMA WRITEs into, round n
 * into slot n with seq n + 1. A coprocessor on another wire uses WireLink.hpp instead.
 */

#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "CompletionQueue.hpp"
#include "MemoryRegion.hpp"
#include "OobSocket.hpp"
#include "WireLink.hpp"

namespace catalyst::transport::common {

class RcLink final : public WireLink {
  public:
    /// The coprocessor listens on the out-of-band socket, sends its HandshakeMsg first and
    /// WRITEs on bwd; the controller connects and WRITEs on fwd.
    enum class Side { Controller, Coprocessor };

    /// @param backend Name used in error messages, e.g. "cpu_verbs".
    /// @param ring Frame layout of both rings, from the session's config.
    /// @param fwd_shape Shape of the fwd queue pair.
    RcLink(std::string dev, int gid_idx, const char *backend, Side side, frame::RingGeometry ring,
           QpShape fwd_shape = DEFAULT_QP_SHAPE);

    /// Open the device, bring both queue pairs to INIT, and reach the peer on the out-of-band
    /// socket.
    void connect(const ConnectInfo &info) override;
    /// Where a local ring is registered: at its own address, which exchange() hands the peer.
    std::uint64_t ring_iova(const void *addr) const override {
        return reinterpret_cast<std::uint64_t>(addr);
    }
    /// Swap HandshakeMsgs for `local`, the ring the peer writes into, and bring both queue pairs
    /// to RTS. Returns the peer's ring.
    PeerRef exchange(const MemRegion &local) override;
    /// Keep `peer`, the ring this end writes into, and register the frame send() sends from.
    void establish(const MemRegion &local, const PeerRef &peer) override;

    void *send_frame() override { return send_buf_->addr(); }
    /// WRITE the staged frame into slot seq - 1 of the peer's ring, inline when it fits. One send
    /// in SIGNAL_EVERY is signaled, and the completions ready then are reaped.
    void send(std::uint32_t seq) override {
        ring_.mode == frame::Mode::BL16 ? post<frame::Mode::BL16>(seq)
                                        : post<frame::Mode::BLV>(seq);
    }
    /// Nothing: send() reaps as it goes.
    void idle() override {}
    /// Wait for the signaled sends still outstanding.
    void drain() override;

  private:
    template <frame::Mode M> void post(std::uint32_t seq);
    void reap(bool drain);

    const char *backend_;
    Side side_;
    QpShape fwd_shape_;
    ibv_gid mygid_{};              // local GID, cached in connect() for exchange()
    std::uint32_t active_mtu_ = 0; // local active_mtu enum, cached in connect()
    std::shared_ptr<CompletionQueue> fwd_cq_, bwd_cq_;
    std::shared_ptr<QueuePair> fwd_qp_, bwd_qp_;
    FdGuard oob_fd_;
    std::optional<MemoryRegion> send_buf_; // local send source (one frame)
    PeerRef peer_{};
    int outstanding_ = 0; // signaled sends not yet reaped
};

} // namespace catalyst::transport::common
