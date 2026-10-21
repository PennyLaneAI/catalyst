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

#pragma once
#include <cstdint>
#include <cstring>

#include <arpa/inet.h>

namespace catalyst::transport::common {

/**
 * @struct QpInfo
 * @brief Connection metadata exchanged out-of-band to connect remote Queue
 * Pairs.
 */
struct QpInfo {
    std::uint32_t qpn;    // Unique Queue Pair Number hardware identifier.
    std::uint32_t psn;    // Starting Packet Sequence Number for flow validation.
    std::uint8_t gid[16]; // 128-bit Global Identifier routing address.
};

/**
 * @struct HandshakeMsg
 * @brief Message exchanged once over the OOB TCP socket, after the MR exists,
 * so QP identity and MR handle are swapped together.
 */
struct HandshakeMsg {
    QpInfo fwd;             // forward QP (controller -> coprocessor)
    QpInfo bwd;             // backward QP (coprocessor -> controller)
    std::uint64_t mr_vaddr; // where the peer writes into us
    std::uint32_t mr_rkey;
    std::uint32_t mtu_enum; // ibv_mtu enum
    // The frame ring this end expects the peer to write into (frame::Mode and slot stride); both
    // ends must agree.
    std::uint32_t frame_mode;
    std::uint32_t frame_stride;
    std::uint32_t cuqr_tx; // CUQR completion verb: 0 SEND, 1 RDMA write with immediate
};

/**
 * @struct CudaqRendezvous
 * @brief cudaq-realtime's handshake (its "rendezvous"): each end sends its queue pair number, ring
 * rkey and RoCE IPv4 address, in network byte order, and the caller (our controller) writes first.
 * No address is exchanged: both rings are registered at iova 0.
 */
struct CudaqRendezvous {
    std::uint32_t qpn;
    std::uint32_t rkey;
    std::uint8_t ipv4[4];
};
static_assert(sizeof(CudaqRendezvous) == 12, "cudaq-realtime's rendezvous is 12 B");

/// This end's rendezvous; the IPv4 address is the tail of its IPv4-mapped RoCEv2 GID.
inline CudaqRendezvous cudaq_rendezvous(std::uint32_t qpn, std::uint32_t rkey,
                                        const std::uint8_t gid[16]) {
    CudaqRendezvous r{.qpn = htonl(qpn), .rkey = htonl(rkey), .ipv4 = {}};
    std::memcpy(r.ipv4, gid + 12, sizeof(r.ipv4));
    return r;
}

/// The peer's RoCEv2 GID from its rendezvous: `::ffff:<ipv4>`.
inline void cudaq_peer_gid(const CudaqRendezvous &r, std::uint8_t gid[16]) {
    std::memset(gid, 0, 16);
    gid[10] = 0xff;
    gid[11] = 0xff;
    std::memcpy(gid + 12, r.ipv4, sizeof(r.ipv4));
}

} // namespace catalyst::transport::common
