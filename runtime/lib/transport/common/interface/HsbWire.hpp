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
 * The HSB wire: a controller on an FPGA running NVIDIA's Holoscan Sensor Bridge (HSB) IP talking
 * to a coprocessor's NIC.
 *
 * - Request: HSB writes each 64 B request into the coprocessor's ring, slot n at n * SLOT_BYTES
 *   from IOVA 0, as a plain UC WRITE. It starts with a BL16 frame (Frame.hpp); the
 *   controller stamps its seq, counting from 1.
 * - Reply: the coprocessor sends a Payload back as an inline UC SEND to FPGA_RX_QPN, echoing the
 *   request's seq_num, which is how the controller matches it.
 * - Session setup: the coprocessor listens on its out-of-band port; once the controller has
 *   connected, the coprocessor sends HostInfo, the controller answers with FpgaInfo, the
 *   coprocessor moves its QP to RTS and sends Ready.
 *
 * The three structs are byte for byte backline-hsb's common/include/hsb_handshake.h, so its
 * hsbctl and hsb-echo interoperate with these backends.
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "ConfigParser.hpp"
#include "Error.hpp"
#include "Frame.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::common::hsb {

inline constexpr std::uint32_t HS_MAGIC = 0x31425348u; // "HSB1"
inline constexpr std::uint32_t HS_VERSION = 1;

// HSB's page is 128 B, the smallest slot pitch it can express.
inline constexpr std::uint32_t SLOT_BYTES = 128;
// The immediate carries a 12-bit slot index.
inline constexpr std::uint32_t MAX_SLOTS = 4096;
// One 512-bit beat: HSB's data path cannot carry less per message.
inline constexpr std::uint32_t FRAME_BYTES = 64;
inline constexpr std::uint32_t REPLY_BYTES = sizeof(Payload);
// Sensor-TX port 0, which hands the reply to the controller.
inline constexpr std::uint32_t FPGA_RX_QPN = 2;

/// Coprocessor -> controller: where HSB writes.
struct HostInfo {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t qpn;        // UC QP receiving HSB's WRITEs
    std::uint32_t rkey;       // of the ring
    std::uint64_t ring_iova;  // IOVA of slot 0
    std::uint32_t ring_pitch; // bytes between slots, a multiple of 128
    std::uint32_t ring_slots; // <= 4096
    std::uint32_t ip;         // RoCE IPv4, host byte order
    std::uint8_t mac[6];      // RoCE netdev MAC
    std::uint16_t imm;        // 1: WRITE_WITH_IMM (stock HSB), 0: plain WRITE
    std::uint32_t reserved;
};
static_assert(sizeof(HostInfo) == 48, "HostInfo layout");

/// Controller -> coprocessor: where the reply goes.
struct FpgaInfo {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t ip; // HSB host port IPv4, host byte order
    std::uint8_t mac[6];
    std::uint16_t pad0;
    std::uint32_t qpn;         // FPGA_RX_QPN
    std::uint32_t frame_bytes; // request bytes HSB writes per message
    std::uint32_t reply_bytes; // reply SEND payload the controller expects
    std::uint32_t seq_off;     // byte offset of the u32 seq in request and reply
};
static_assert(sizeof(FpgaInfo) == 36, "FpgaInfo layout");

/// Coprocessor -> controller: the QP is at RTS, HSB may start writing.
struct Ready {
    std::uint32_t magic;
    std::uint32_t qp_state; // IBV_QPS_RTS
};
static_assert(sizeof(Ready) == 8, "Ready layout");

static_assert(offsetof(Payload, seq_num) + sizeof(Payload::seq_num) <= FRAME_BYTES,
              "the seq a request carries must fall inside the frame HSB writes");

/// The ring HSB writes into: BL16 frames, SLOT_BYTES apart, `frame_slots` of them
inline frame::RingGeometry ring_from_config(std::string_view config) {
    frame::RingGeometry ring = frame::geometry_from_config(config);
    TP_CHECK(ring.mode == frame::Mode::BL16,
             "peer=hsb: HSB carries only BL16 frames (8 B messages)");
    TP_CHECK(ring.slots <= MAX_SLOTS, "peer=hsb: frame_slots is %u, HSB addresses at most %u",
             ring.slots, MAX_SLOTS);
    ring.stride = SLOT_BYTES;
    return ring;
}

/// Whether a coprocessor's config asks HSB for WRITE_WITH_IMM
inline bool imm_from_config(const std::string &config) {
    return configparser::parse_optional_index(config, "imm", 0) != 0;
}

} // namespace catalyst::transport::common::hsb
