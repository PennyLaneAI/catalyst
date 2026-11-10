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
 * A coprocessor's end of the BALBOA wire (BalboaWire.hpp), `wire=balboa`: one RC QP for both
 * directions, BALBOA's out-of-band handshake, and the reply as an inline RDMA WRITE to the address
 * BALBOA names. BALBOA writes each request into a ring of BL16 frames at IOVA 0.
 */

#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "BalboaWire.hpp"
#include "CompletionQueue.hpp"
#include "OobSocket.hpp"
#include "WireLink.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::common {

class BalboaLink final : public WireLink {
  public:
    static constexpr unsigned MAX_SEND = 64;
    // Every SIGNAL_EVERY-th reply is signaled; its completion retires that many.
    static constexpr unsigned SIGNAL_EVERY = 32;

    BalboaLink(std::string dev, int gid_idx, frame::RingGeometry ring);

    void connect(const ConnectInfo &info) override;
    /// BALBOA writes round n at n * stride from IOVA 0.
    std::uint64_t ring_iova(const void * /*addr*/) const override { return 0; }
    /// Send HostInfo for `local` and take FpgaInfo back. BALBOA takes the reply at an address of
    /// its own, so there is no peer ring to return.
    PeerRef exchange(const MemRegion &local) override;
    /// Bring the QP to RTS toward BALBOA, then tell the controller it may start.
    void establish(const MemRegion &local, const PeerRef &peer) override;

    void *send_frame() override { return &send_frame_; }
    void send(std::uint32_t seq) override;
    /// Retire send completions.
    void idle() override {
        if (inflight_ >= SIGNAL_EVERY) {
            reap_send();
        }
    }
    void drain() override {}

  private:
    void reap_send();

    std::uint8_t port_ = 1;
    std::uint32_t active_mtu_ = 0;
    Ipv4Gid local_{};       // the RoCE address BALBOA sends to
    std::uint32_t psn_ = 0; // of both directions; ours, and BALBOA echoes it
    std::shared_ptr<CompletionQueue> cq_;
    std::shared_ptr<QueuePair> qp_;
    FdGuard oob_;
    balboa::FpgaInfo fpga_{};

    Payload send_frame_{};
    std::uint64_t sent_ = 0;
    unsigned inflight_ = 0; // sends posted and not yet known complete
};

} // namespace catalyst::transport::common
