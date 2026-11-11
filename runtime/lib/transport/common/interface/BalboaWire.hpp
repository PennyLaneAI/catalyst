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
 * The BALBOA wire: a controller on an FPGA running ETH Zurich's BALBOA RoCE v2 stack talking to a
 * coprocessor's ConnectX NIC over one RC connection.
 *
 * - Request: BALBOA writes each request into the coprocessor's ring with an RC RDMA WRITE, round
 *   n into slot n mod the ring's slots, from IOVA 0. It starts with a BL16 frame (Frame.hpp); the
 *   controller stamps its seq, counting from 1.
 * - Reply: the coprocessor RDMA WRITEs a Payload back over the same QP, echoing the request's
 *   seq_num, which is how the controller matches it.
 * - Session setup: the coprocessor listens on its out-of-band port; once the controller has
 *   connected, the coprocessor sends HostInfo, the controller answers with FpgaInfo, the
 *   coprocessor moves its QP to RTS and sends Ready. One PSN serves both directions.
 *
 * The three structs are byte for byte backline-roce's common/include/balboa_handshake.h, so its
 * balboactl and balboa-echo interoperate with these backends.
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "Error.hpp"
#include "Frame.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::common::balboa {

inline constexpr std::uint32_t HS_MAGIC = 0x31424C42u; // "BLB1"
inline constexpr std::uint32_t HS_VERSION = 1;

// The engine's slot and reply RAMs hold this many entries, so the ring can have no more.
inline constexpr std::uint32_t MAX_SLOTS = 1024;
// One 512-bit beat per request.
inline constexpr std::uint32_t FRAME_BYTES = 64;
inline constexpr std::uint32_t REPLY_BYTES = sizeof(Payload);

/// Coprocessor -> controller: its QP and the ring BALBOA writes.
struct HostInfo {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t qpn;         // RC QP that takes BALBOA's WRITEs
    std::uint32_t psn;         // the PSN of both directions (24 bit)
    std::uint32_t rkey;        // of the ring
    std::uint32_t ring_slots;  // <= MAX_SLOTS
    std::uint64_t ring_iova;   // IOVA of slot 0
    std::uint32_t ring_stride; // bytes between slots, a power of two >= 64
    std::uint32_t ip;          // RoCE IPv4, host byte order
    std::uint8_t mac[6];       // RoCE netdev MAC; BALBOA resolves it with ARP
    std::uint16_t pad;
};
static_assert(sizeof(HostInfo) == 48, "HostInfo layout");

/// Controller -> coprocessor: BALBOA's QP and where the reply goes.
struct FpgaInfo {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t qpn;        // BALBOA's QP
    std::uint32_t psn;        // the same PSN back
    std::uint32_t ip;         // BALBOA's IPv4, host byte order
    std::uint32_t reply_rkey; // of the reply WRITE (BALBOA does not check it)
    std::uint64_t reply_addr; // of the reply WRITE (the engine takes every WRITE)
    std::uint8_t mac[6];
    std::uint16_t pad0;
    std::uint32_t req_bytes;   // bytes of each request WRITE
    std::uint32_t reply_bytes; // bytes of the reply WRITE
    std::uint32_t seq_off;     // byte offset of the u32 seq in request and reply
    std::uint32_t pad1;
};
static_assert(sizeof(FpgaInfo) == 56, "FpgaInfo layout");

/// Coprocessor -> controller: the QP is at RTS, BALBOA may start writing.
struct Ready {
    std::uint32_t magic;
    std::uint32_t qp_state; // IBV_QPS_RTS
};
static_assert(sizeof(Ready) == 8, "Ready layout");

/// The ring BALBOA writes into: BL16 frames, `frame_slots` of them (256 when not given).
inline frame::RingGeometry ring_from_config(std::string_view config) {
    frame::RingGeometry ring = frame::geometry_from_config(config);
    TP_CHECK(ring.mode == frame::Mode::BL16,
             "peer=balboa: the engine carries only BL16 frames (8 B messages)");
    TP_CHECK(ring.slots <= MAX_SLOTS, "peer=balboa: frame_slots is %u, the engine holds at most %u",
             ring.slots, MAX_SLOTS);
    TP_CHECK(ring.stride >= FRAME_BYTES, "peer=balboa: a %u B slot cannot hold a %u B request",
             ring.stride, FRAME_BYTES);
    return ring;
}

} // namespace catalyst::transport::common::balboa
