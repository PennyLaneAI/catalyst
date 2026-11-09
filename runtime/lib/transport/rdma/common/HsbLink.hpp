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
 * A coprocessor's end of the HSB wire (HsbWire.hpp): one UC QP, the out-of-band handshake, and
 * the queue bookkeeping the CPU and GPU HSB sessions share. The ring HSB writes into, host or GPU
 * memory, belongs to the session, which registers it at IOVA 0 on pd().
 */

#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "CompletionQueue.hpp"
#include "Context.hpp"
#include "Frame.hpp"
#include "HsbWire.hpp"
#include "OobSocket.hpp"
#include "ProtectionDomain.hpp"
#include "QueuePair.hpp"

namespace catalyst::transport::common {

class HsbLink {
  public:
    // With imm, HSB's WRITE_WITH_IMM consumes a receive per request, so this many stay posted.
    static constexpr unsigned MAX_RECV = 512;
    static constexpr unsigned MAX_SEND = 64;
    // Every SIGNAL_EVERY-th reply is signaled; its completion retires that many.
    static constexpr unsigned SIGNAL_EVERY = 32;
    // At most this many receive completions are taken per reap_recv().
    static constexpr int RECV_BATCH = 16;

    /// `imm`: HSB sends WRITE_WITH_IMM (`imm=1`) or plain WRITE (`imm=0`)
    HsbLink(std::string dev, int gid_idx, bool imm);
    HsbLink(const HsbLink &) = delete;
    HsbLink &operator=(const HsbLink &) = delete;

    /// Open the device, bring a UC QP to INIT (with its receives posted if imm), and accept the
    /// controller on `oob_port`.
    void connect(std::uint16_t oob_port);
    std::shared_ptr<ProtectionDomain> pd() const { return pd_; }

    /// Send HostInfo for `ring`, registered at IOVA 0 under `rkey`, and take FpgaInfo back.
    void exchange(std::uint32_t rkey, const frame::RingGeometry &ring);
    /// Bring the QP to RTS toward the FPGA, then tell the controller it may start.
    void establish();

    /// Hot path: post `reply` (a Payload) as an inline SEND.
    void send(const void *reply);
    /// Idle path: retire one signaled send completion if one is ready.
    void reap_send();
    unsigned sends_inflight() const { return inflight_; }
    bool imm() const { return imm_; }
    /// Idle path: take up to RECV_BATCH receive completions, each one request HSB delivered,
    /// repost their receives, and put the ring slot each request landed in into `slots`.
    int reap_recv(std::uint32_t slots[RECV_BATCH]);

  private:
    void post_recvs(unsigned n);

    std::string dev_name_;
    int gid_idx_;
    bool imm_;
    std::uint8_t port_ = 1;
    std::uint32_t active_mtu_ = 0;
    std::uint32_t ip_ = 0; // RoCE IPv4 of the GID, host byte order
    std::uint8_t mac_[6] = {};
    std::shared_ptr<Context> ctx_;
    std::shared_ptr<ProtectionDomain> pd_;
    std::shared_ptr<CompletionQueue> scq_, rcq_;
    std::shared_ptr<QueuePair> qp_;
    FdGuard oob_;
    hsb::FpgaInfo fpga_{};
    std::uint64_t sent_ = 0;
    unsigned inflight_ = 0; // sends posted and not yet known complete
};

} // namespace catalyst::transport::common
