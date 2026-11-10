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
 * One end of a wire, the protocol a session speaks to its peer, which a coprocessor's config names
 * with `wire=` (BackendConfig.hpp):
 *
 * | wire   | link       | peer                                                   |
 * |--------|------------|--------------------------------------------------------|
 * | rc     | RcLink     | another catalyst session, e.g. any catalyst controller |
 * | hsb    | HsbLink    | a controller behind NVIDIA's Holoscan Sensor Bridge    |
 * | balboa | BalboaLink | a controller behind ETH Zurich's BALBOA                |
 *
 * The peer writes each request into a ring of frames the session owns. The links are final, so a
 * session that calls a link through its own type in the hot loop pays no virtual call.
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

class WireLink {
  public:
    virtual ~WireLink() = default;
    WireLink(const WireLink &) = delete;
    WireLink &operator=(const WireLink &) = delete;

    Wire wire() const { return wire_; }
    /// The frames of the ring the peer writes into.
    const frame::RingGeometry &ring() const { return ring_; }
    std::shared_ptr<ProtectionDomain> pd() const { return pd_; }

    /// Open the device, bring the wire's queue pairs to INIT, and reach the peer on the
    /// out-of-band socket.
    virtual void connect(const ConnectInfo &info) = 0;
    /// Where to register a local ring.
    virtual std::uint64_t ring_iova(const void *addr) const = 0;
    /// Swap the wire's handshake for `local`, the ring the peer writes into. Returns the peer's
    /// ring, for a wire that has one.
    virtual PeerRef exchange(const MemRegion &local) = 0;
    /// Finish bring-up toward the peer, which may then start sending.
    virtual void establish(const MemRegion &local, const PeerRef &peer) = 0;

    /// The frame the next send() sends; stage it with frame::stage first.
    virtual void *send_frame() = 0;
    /// Hot path: seal the staged frame with `seq`, the request's, and send it.
    virtual void send(std::uint32_t seq) = 0;
    /// Idle path: service the send queue while no request is due.
    virtual void idle() = 0;
    /// Teardown: wait for the sends still outstanding.
    virtual void drain() = 0;

  protected:
    WireLink(Wire wire, std::string dev, int gid_idx, frame::RingGeometry ring)
        : wire_(wire), dev_name_(std::move(dev)), gid_idx_(gid_idx), ring_(ring) {}

    Wire wire_;
    std::string dev_name_;
    int gid_idx_;
    frame::RingGeometry ring_;
    std::shared_ptr<Context> ctx_;
    std::shared_ptr<ProtectionDomain> pd_;
};

/// The link a coprocessor's `config` asks for with `wire=`. `backend` names it in messages and
/// `fwd_shape` sizes an RC link's fwd queue pair.
std::unique_ptr<WireLink> make_coprocessor_link(const std::string &config, const char *backend,
                                                QpShape fwd_shape = DEFAULT_QP_SHAPE);

} // namespace catalyst::transport::common
