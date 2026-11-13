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
 * One end of a session's connection to its peer. A coprocessor's config names the peer with
 * `peer=` (BackendConfig.hpp), and the peer decides the protocol the link speaks:
 *
 * | peer     | link    | the other end                                       |
 * |----------|---------|-----------------------------------------------------|
 * | catalyst | RcLink  | another catalyst session, such as a controller      |
 * | hsb      | HsbLink | a controller behind NVIDIA's Holoscan Sensor Bridge |
 *
 * The peer writes each request into a ring of frames that the session owns. The links are final,
 * so call one through its concrete type on the critical path to skip the virtual call.
 */

#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "BackendConfig.hpp"
#include "Context.hpp"
#include "Frame.hpp"
#include "ProtectionDomain.hpp"
#include "QueuePair.hpp"
#include "Transport.hpp"

namespace catalyst::transport::common {

class PeerLink {
  public:
    virtual ~PeerLink() = default;
    PeerLink(const PeerLink &) = delete;
    PeerLink &operator=(const PeerLink &) = delete;

    Peer peer() const { return peer_; }
    /// Frame layout of the ring the peer writes into.
    const frame::RingGeometry &ring() const { return ring_; }
    std::shared_ptr<ProtectionDomain> pd() const { return pd_; }

    /// Open the device, bring the link's queue pairs to INIT, and connect to the peer over the
    /// out-of-band socket.
    virtual void connect(const ConnectInfo &info) = 0;
    /// IOVA at which to register the local ring that starts at `addr`.
    virtual std::uint64_t ring_iova(const void *addr) const = 0;
    /// Trade handshakes with the peer, advertising `local` as the ring the peer writes into.
    /// Returns the peer's ring if the peer has one.
    virtual PeerRef exchange(const MemRegion &local) = 0;
    /// Finish bring-up. Once this returns, the peer may start sending.
    virtual void establish(const MemRegion &local, const PeerRef &peer) = 0;

    /// The frame that the next send() sends. Stage it with frame::stage first.
    virtual void *send_frame() = 0;
    /// Hot path: seal the staged frame with the request's `seq` and send it.
    virtual void send(std::uint32_t seq) = 0;
    /// Idle path: service the send queue while no request is due.
    virtual void idle() = 0;
    /// Teardown: wait for outstanding sends to complete.
    virtual void drain() = 0;

  protected:
    PeerLink(Peer peer, std::string dev, int gid_idx, frame::RingGeometry ring)
        : peer_(peer), dev_name_(std::move(dev)), gid_idx_(gid_idx), ring_(ring) {}

    Peer peer_;
    std::string dev_name_;
    int gid_idx_;
    frame::RingGeometry ring_;
    std::shared_ptr<Context> ctx_;
    std::shared_ptr<ProtectionDomain> pd_;
};

/// Build the link that a coprocessor's `config` names with `peer=`. `backend` names it in
/// messages, and `fwd_shape` sizes the fwd queue pair of an RC link.
std::unique_ptr<PeerLink> make_coprocessor_link(const std::string &config, const char *backend,
                                                QpShape fwd_shape = DEFAULT_QP_SHAPE);

} // namespace catalyst::transport::common
