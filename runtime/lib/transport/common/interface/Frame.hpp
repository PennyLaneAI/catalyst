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

// The frame codec: how a message is laid out in a ring slot, and how a ring is laid out.
//
// Every backend builds and reads frames through these functions, so the layouts are defined here
// once. Two layouts are implemented, chosen per session by message size (see geometry_for):
//
//   BL16  16 B: | payload 8 | fn_id 4 | seq 4 |            (Payload in WireProtocol.hpp)
//   BLV   one slot: | header 32 | payload ... | completion 4 |
//         header = magic "BLV1", version, flags, fn_id, seq, round, len, t_ptp_ns
//
// In both, the word a receiver polls (BL16's seq, BLV's completion word) sits at a fixed offset
// and is written last, so a frame is complete once it holds the expected sequence number. A BLV
// frame always fills its slot, so the completion word's offset never depends on the payload.

#pragma once
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "ConfigParser.hpp"
#include "Error.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::common::frame {

/// A frame layout. Matches the CATALYST_TRANSPORT_FRAME_* bit a backend's caps report.
enum class Mode : std::uint8_t {
    BL16 = 0, // 16 B: 8 B of payload, the fn_id, then seq (Payload)
    BLV = 1,  // variable length: a 32 B header, the payload, then a completion word
    CUQR = 2, // TODO: cudaq-realtime compatible codec, not implemented yet
};

#pragma pack(push, 1)
/// The header that starts every BLV frame.
struct BlvHeader {
    std::uint32_t magic;    // BLV_MAGIC
    std::uint16_t version;  // BLV_VERSION
    std::uint16_t flags;    // 0; reserved for heralds, soft input and convergence flags
    std::uint32_t fn_id;    // which function on the coprocessor handles the message
    std::uint32_t seq;      // the round's sequence number, also written to the completion word
    std::uint32_t round;    // syndrome round or window index; 0 if unused
    std::uint32_t len;      // payload bytes
    std::uint64_t t_ptp_ns; // sender timestamp; 0 if unavailable
};
#pragma pack(pop)
static_assert(sizeof(BlvHeader) == 32, "a BLV header is 32 B");

inline constexpr std::uint32_t BLV_MAGIC = 0x31564C42; // "BLV1", little-endian
inline constexpr std::uint16_t BLV_VERSION = 1;
inline constexpr std::size_t BLV_COMPLETION_BYTES = sizeof(std::uint32_t);
/// The smallest slot a ring uses, for the DMA engines that need 64 B alignment.
inline constexpr std::size_t MIN_STRIDE = 64;

/// How a ring of frames is laid out: `slots` frames, each starting `stride` bytes after the last.
struct RingGeometry {
    Mode mode = Mode::BL16;
    std::uint32_t slots = static_cast<std::uint32_t>(K_RING_SLOTS); // a power of two
    std::uint32_t stride = sizeof(PayloadSlot);                     // a power of two, >= 64

    std::size_t region_bytes() const { return std::size_t{slots} * stride; }
    /// Byte offset of the slot round `cursor` uses.
    std::size_t slot_offset(std::uint64_t cursor) const {
        return static_cast<std::size_t>(cursor & (slots - 1)) * stride;
    }
};

/// The ring every session uses unless its config asks for another: BL16 frames in 256 slots of
/// 64 B.
inline constexpr RingGeometry BL16_RING{};

/// The ring for messages of up to `max_payload` bytes each way: BL16 when they fit in 8 B unless
/// `mode` asks for BLV, else BLV with the smallest power-of-two slot that holds a frame.
inline RingGeometry geometry_for(Mode mode, std::size_t max_payload) {
    if (mode == Mode::BL16) {
        TP_CHECK(max_payload <= PAYLOAD_DATA_BYTES, "a BL16 frame carries at most %zu B, not %zu B",
                 PAYLOAD_DATA_BYTES, max_payload);
        return BL16_RING;
    }
    TP_CHECK(mode == Mode::BLV, "frame mode %d is not implemented", static_cast<int>(mode));
    const std::size_t frame = sizeof(BlvHeader) + max_payload + BLV_COMPLETION_BYTES;
    const std::size_t stride = std::bit_ceil(frame < MIN_STRIDE ? MIN_STRIDE : frame);
    TP_CHECK(stride <= UINT32_MAX, "a %zu B payload is too large for a frame", max_payload);
    return RingGeometry{.mode = Mode::BLV, .stride = static_cast<std::uint32_t>(stride)};
}

/// The ring a session's backend config asks for: `frame=bl16|blv` with `frame_payload=<bytes>`,
/// the largest payload either way. A config without `frame` gets BL16_RING.
inline RingGeometry geometry_from_config(std::string_view config) {
    const auto mode = configparser::find(config, "frame");
    if (!mode) {
        return BL16_RING;
    }
    const auto payload = configparser::find(config, "frame_payload");
    const std::size_t max_payload =
        payload ? configparser::parse_u64(*payload, "frame_payload") : PAYLOAD_DATA_BYTES;
    if (*mode == "bl16") {
        return geometry_for(Mode::BL16, max_payload);
    }
    TP_CHECK(*mode == "blv", "config: 'frame' must be 'bl16' or 'blv', got '%.*s'",
             static_cast<int>(mode->size()), mode->data());
    return geometry_for(Mode::BLV, max_payload);
}

/// A received frame, decoded.
struct FrameView {
    std::uint8_t *data = nullptr; // the payload
    std::uint32_t len = 0;        // payload bytes the frame carries (BL16: always 8, zero-padded)
    std::uint32_t fn_id = 0;      // which function on the coprocessor handles it
    std::uint32_t seq = 0;        // the round's sequence number, cursor + 1
};

namespace detail {
inline void require_mode(Mode mode) {
    TP_CHECK(mode == Mode::BL16 || mode == Mode::BLV, "frame mode %d is not implemented",
             static_cast<int>(mode));
}
inline BlvHeader *blv(void *frame) { return static_cast<BlvHeader *>(frame); }
inline std::uint32_t *completion(const RingGeometry &ring, void *frame) {
    return reinterpret_cast<std::uint32_t *>(static_cast<std::uint8_t *>(frame) + ring.stride -
                                             BLV_COMPLETION_BYTES);
}
} // namespace detail

// Each operation takes the mode at compile time; the plain forms at the end read it from the ring.

/// Bytes on the wire for one frame: 16 for BL16, the whole slot for BLV.
template <Mode M> inline std::size_t frame_bytes(const RingGeometry &ring) {
    return M == Mode::BL16 ? sizeof(Payload) : ring.stride;
}

/// Payload bytes one frame carries.
template <Mode M> inline std::size_t data_capacity(const RingGeometry &ring) {
    return M == Mode::BL16 ? PAYLOAD_DATA_BYTES
                           : ring.stride - sizeof(BlvHeader) - BLV_COMPLETION_BYTES;
}

/// Where a frame's payload starts, for a caller that writes or reads it in place.
template <Mode M> inline std::uint8_t *data(const RingGeometry &, void *frame) {
    return static_cast<std::uint8_t *>(frame) + (M == Mode::BL16 ? 0 : sizeof(BlvHeader));
}

/// Fill a frame's payload, its length and fn_id. `src` may be the frame's own payload area, for a
/// payload already written in place. BL16 zero-pads the payload to 8 B. seq is left for seal().
template <Mode M>
inline void stage(const RingGeometry &ring, void *frame, std::uint32_t fn_id, const void *src,
                  std::size_t len) {
    const std::size_t cap = data_capacity<M>(ring);
    TP_CHECK(len <= cap, "payload (%zu B) exceeds the %zu B frame", len, cap);
    if constexpr (M == Mode::BL16) {
        Payload *p = static_cast<Payload *>(frame);
        std::uint64_t value = 0;
        if (src == &p->value) {
            value =
                len < sizeof(value) ? p->value & ((std::uint64_t{1} << (8 * len)) - 1) : p->value;
        } else if (len != 0) {
            std::memcpy(&value, src, len);
        }
        p->value = value;
        p->decoder_id = fn_id;
        return;
    }
    std::uint8_t *payload = data<M>(ring, frame);
    if (len != 0 && src != payload) {
        std::memmove(payload, src, len);
    }
    *detail::blv(frame) = BlvHeader{
        .magic = BLV_MAGIC,
        .version = BLV_VERSION,
        .flags = 0,
        .fn_id = fn_id,
        .seq = 0,
        .round = 0,
        .len = static_cast<std::uint32_t>(len),
        .t_ptp_ns = 0,
    };
}

/// Set a staged frame's seq, writing last the word a receiver polls, which makes the frame
/// complete.
template <Mode M> inline void seal(const RingGeometry &ring, void *frame, std::uint32_t seq) {
    if constexpr (M == Mode::BL16) {
        std::atomic_ref<std::uint32_t>(static_cast<Payload *>(frame)->seq_num)
            .store(seq, std::memory_order_release);
        return;
    }
    detail::blv(frame)->seq = seq;
    std::atomic_ref<std::uint32_t>(*detail::completion(ring, frame))
        .store(seq, std::memory_order_release);
}

/// Whether the frame for sequence number `expect` has arrived in `frame`; if so, decode it into
/// `out`. The polled word is read with acquire ordering, so everything written before it is
/// visible.
template <Mode M>
inline bool poll(const RingGeometry &ring, void *frame, std::uint32_t expect, FrameView &out) {
    if constexpr (M == Mode::BL16) {
        Payload *p = static_cast<Payload *>(frame);
        if (std::atomic_ref<std::uint32_t>(p->seq_num).load(std::memory_order_acquire) != expect) {
            return false;
        }
        out = FrameView{
            .data = reinterpret_cast<std::uint8_t *>(&p->value),
            .len = static_cast<std::uint32_t>(PAYLOAD_DATA_BYTES),
            .fn_id = p->decoder_id,
            .seq = expect,
        };
        return true;
    }
    if (std::atomic_ref<std::uint32_t>(*detail::completion(ring, frame))
            .load(std::memory_order_acquire) != expect) {
        return false;
    }
    const BlvHeader *h = detail::blv(frame);
    TP_CHECK(h->magic == BLV_MAGIC && h->version == BLV_VERSION,
             "slot holds no BLV frame (magic 0x%08x, version %u)", h->magic, h->version);
    TP_CHECK(h->len <= data_capacity<M>(ring),
             "BLV frame claims %u B, more than its %zu B slot holds", h->len,
             data_capacity<M>(ring));
    out = FrameView{
        .data = data<M>(ring, frame),
        .len = h->len,
        .fn_id = h->fn_id,
        .seq = expect,
    };
    return true;
}

inline std::size_t frame_bytes(const RingGeometry &ring) {
    detail::require_mode(ring.mode);
    return ring.mode == Mode::BL16 ? frame_bytes<Mode::BL16>(ring) : frame_bytes<Mode::BLV>(ring);
}

inline std::size_t data_capacity(const RingGeometry &ring) {
    detail::require_mode(ring.mode);
    return ring.mode == Mode::BL16 ? data_capacity<Mode::BL16>(ring)
                                   : data_capacity<Mode::BLV>(ring);
}

inline std::uint8_t *data(const RingGeometry &ring, void *frame) {
    detail::require_mode(ring.mode);
    return ring.mode == Mode::BL16 ? data<Mode::BL16>(ring, frame) : data<Mode::BLV>(ring, frame);
}

inline void stage(const RingGeometry &ring, void *frame, std::uint32_t fn_id, const void *src,
                  std::size_t len) {
    detail::require_mode(ring.mode);
    ring.mode == Mode::BL16 ? stage<Mode::BL16>(ring, frame, fn_id, src, len)
                            : stage<Mode::BLV>(ring, frame, fn_id, src, len);
}

inline void seal(const RingGeometry &ring, void *frame, std::uint32_t seq) {
    detail::require_mode(ring.mode);
    ring.mode == Mode::BL16 ? seal<Mode::BL16>(ring, frame, seq)
                            : seal<Mode::BLV>(ring, frame, seq);
}

inline bool poll(const RingGeometry &ring, void *frame, std::uint32_t expect, FrameView &out) {
    if (ring.mode == Mode::BL16) {
        return poll<Mode::BL16>(ring, frame, expect, out);
    }
    detail::require_mode(ring.mode);
    return poll<Mode::BLV>(ring, frame, expect, out);
}

} // namespace catalyst::transport::common::frame
