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
#include <cstddef>
#include <cstdint>

namespace catalyst::transport::common {

// Ring size, identical on both sides; power of two so index is a mask.
inline constexpr std::size_t K_RING_SLOTS = 256;

// Selective-signalling stride (flow control on the pipelined send paths).
inline constexpr std::uint32_t SIGNAL_EVERY = 64;

// Validation salt: packed into Payload.value low bits in the RTT self-test.
inline constexpr std::uint32_t SALT = 0xC0DE1515u;

// Demo/loopback payload the controller ships each shot (stand-in for a real
// measurement outcome). Its (echo) decode is what the self-test checks.
inline constexpr std::uint64_t DEMO_SYNDROME = 0x0123456789ABCDEFull;

// 16 B wire frame.
inline constexpr std::size_t PAYLOAD_DATA_BYTES = 8;

// Note: this application payload size is unrelated to the network MTU (the QP's
// max packet size, negotiated at RTR). The size here is far below any MTU, so
// each transfer is always a single packet.
// Payload layout:
// byte  0                     dec_off   seq_off (4B-aligned)
// v                              v       v
// ┌──────────────────────────────┬───────┬──────────┐
// │      value                   │ dec   │  seq_num │
// └──────────────────────────────┴───────┴──────────┘
#pragma pack(push, 1)
struct Payload {
    std::uint64_t value;      // 8B data payload
    std::uint32_t decoder_id; // selects which decoder handles this message
    std::uint32_t seq_num;    // arrival flag
};
#pragma pack(pop)
static_assert(sizeof(Payload) == 16, "Payload must be exactly 16 B");
static_assert(offsetof(Payload, seq_num) + sizeof(Payload::seq_num) == sizeof(Payload),
              "seq_num must be the last field in Payload");
static_assert(offsetof(Payload, value) == 0,
              "value must be first so a decoder can read the data from the frame's start");
static_assert(offsetof(Payload, decoder_id) == PAYLOAD_DATA_BYTES,
              "the header follows the data area, so decoder_id sits at PAYLOAD_DATA_BYTES");

// Some controller DMA engine requires 64-B aligned. Rings are 64-B-strided slots; only the leading
// Payload is transferred per slot, and the padding is what is left of the slot after it.
struct alignas(64) PayloadSlot {
    Payload p;
    std::uint8_t pad_[64 - sizeof(Payload)];
};
static_assert(sizeof(PayloadSlot) == 64, "PayloadSlot must be exactly 64 B");
static_assert(alignof(PayloadSlot) == 64, "PayloadSlot must be 64-B aligned");

// Receive ring is K_RING_SLOTS PayloadSlots; the peer writes slot[cursor %
// K_RING_SLOTS]. K_RING_SLOTS must be a power of two.
inline constexpr std::size_t REGION_BYTES = K_RING_SLOTS * sizeof(PayloadSlot);

// Messages larger than PAYLOAD_DATA_BYTES.
//
// A backend that is not bound to the fixed 16 B frame (the memcpy backends) carries a message of up
// to MAX_MESSAGE_BYTES in each direction. Its frame has the same shape as Payload with a wider data
// area: `frame_data_bytes(in_bytes)` data bytes at offset 0, then the u32 decoder_id, then the u32
// seq_num. For in_bytes <= PAYLOAD_DATA_BYTES the data area is PAYLOAD_DATA_BYTES and the frame is
// byte-for-byte a Payload, so a coprocessor function written against Payload reads it unchanged.
// Backends bound to the 16 B frame reject a committed size above PAYLOAD_DATA_BYTES.
inline constexpr std::size_t MAX_MESSAGE_BYTES = 4096;

// Bytes in the data area of a frame carrying `in_bytes` of payload: at least PAYLOAD_DATA_BYTES,
// and rounded up to a multiple of 8 so decoder_id and seq_num stay aligned.
inline constexpr std::size_t frame_data_bytes(std::size_t in_bytes) {
    const std::size_t rounded = (in_bytes + 7) & ~std::size_t{7};
    return rounded < PAYLOAD_DATA_BYTES ? PAYLOAD_DATA_BYTES : rounded;
}

// Total bytes of a frame whose data area is `data_bytes`: the data, decoder_id and seq_num.
inline constexpr std::size_t frame_bytes(std::size_t data_bytes) {
    return data_bytes + sizeof(Payload::decoder_id) + sizeof(Payload::seq_num);
}

// The largest frame a memcpy backend carries.
inline constexpr std::size_t MAX_FRAME_BYTES = frame_bytes(frame_data_bytes(MAX_MESSAGE_BYTES));

static_assert(frame_bytes(frame_data_bytes(PAYLOAD_DATA_BYTES)) == sizeof(Payload),
              "a frame of PAYLOAD_DATA_BYTES must be exactly a Payload");

} // namespace catalyst::transport::common
