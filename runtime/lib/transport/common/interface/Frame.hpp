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

// Layout helpers a GPU kernel calls too. The rest of the codec is host-only.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define FRAME_HOST_DEVICE __host__ __device__
#else
#define FRAME_HOST_DEVICE
#endif

namespace catalyst::transport::common::frame {

/// A frame layout. Matches the CATALYST_TRANSPORT_FRAME_* bit a backend's caps report.
enum class Mode : std::uint8_t {
    BL16 = 0, // 16 B: 8 B of payload, the fn_id, then seq (Payload)
    BLV = 1,  // variable length: a 32 B header, the payload, then a completion word
    CUQR = 2, // cudaq-realtime's RPCHeader/RPCResponse: a 24 B header, then the payload
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
#pragma pack(push, 1)
/// cudaq-realtime's request (RPCHeader) and reply (RPCResponse) header: the same 24 B, with
/// `word` the function_id in a request and the status in a reply.
struct CuqrHeader {
    std::uint32_t magic;      // CUQR_MAGIC (request) or CUQS_MAGIC (reply)
    std::uint32_t word;       // request: function_id (our fn_id); reply: int32 status
    std::uint32_t len;        // payload bytes (arg_len / result_len)
    std::uint32_t request_id; // opaque: set by the sender of a request, echoed in its reply
    std::uint64_t t_ptp_ns;   // opaque, echoed
};
#pragma pack(pop)
static_assert(sizeof(CuqrHeader) == 24, "a CUQR header is 24 B");
inline constexpr std::uint32_t CUQR_MAGIC = 0x43555152; // 'CUQR'
inline constexpr std::uint32_t CUQS_MAGIC = 0x43555153; // 'CUQS'

/// What a CUQR reply echoes from its request, and its status (0 ok, >0 the function failed).
struct ReplyTo {
    std::uint32_t request_id;
    std::uint64_t t_ptp_ns;
    std::int32_t status;
};

/// The smallest slot a ring uses, for the DMA engines that need 64 B alignment.
inline constexpr std::size_t MIN_STRIDE = 64;

/// How a ring of frames is laid out: `slots` frames, each starting `stride` bytes after the last.
struct RingGeometry {
    Mode mode = Mode::BL16;
    std::uint32_t slots = static_cast<std::uint32_t>(K_RING_SLOTS); // a power of two
    std::uint32_t stride = sizeof(PayloadSlot);                     // a power of two, >= 64

    FRAME_HOST_DEVICE std::size_t region_bytes() const { return std::size_t{slots} * stride; }
    /// Byte offset of the slot round `cursor` uses.
    FRAME_HOST_DEVICE std::size_t slot_offset(std::uint64_t cursor) const {
        return static_cast<std::size_t>(cursor & (slots - 1)) * stride;
    }
    /// Byte offset of the payload within a frame.
    FRAME_HOST_DEVICE std::size_t data_offset() const {
        return mode == Mode::BL16 ? 0 : mode == Mode::CUQR ? sizeof(CuqrHeader) : sizeof(BlvHeader);
    }
    /// Byte offset of the word a receiver polls: BL16's seq, or BLV's completion word.
    FRAME_HOST_DEVICE std::size_t poll_offset() const {
        return mode == Mode::BL16 ? offsetof(Payload, seq_num) : stride - BLV_COMPLETION_BYTES;
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
    TP_CHECK(mode == Mode::BLV || mode == Mode::CUQR, "frame mode %d is not implemented",
             static_cast<int>(mode));
    // A CUQR frame has no completion word: the transport signals its arrival.
    const std::size_t frame = mode == Mode::CUQR
                                  ? sizeof(CuqrHeader) + max_payload
                                  : sizeof(BlvHeader) + max_payload + BLV_COMPLETION_BYTES;
    const std::size_t stride = std::bit_ceil(frame < MIN_STRIDE ? MIN_STRIDE : frame);
    TP_CHECK(stride <= UINT32_MAX, "a %zu B payload is too large for a frame", max_payload);
    return RingGeometry{.mode = mode, .stride = static_cast<std::uint32_t>(stride)};
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
    if (*mode == "cuqr") {
        RingGeometry ring = geometry_for(Mode::CUQR, max_payload);
        // A cudaq-realtime server sets its slot count (cpu_roce: 8 by default).
        if (const auto slots = configparser::find(config, "frame_slots")) {
            const std::uint64_t n = configparser::parse_u64(*slots, "frame_slots");
            TP_CHECK(n > 0 && n <= UINT32_MAX && std::has_single_bit(n),
                     "config: 'frame_slots' must be a power of two, got %llu",
                     static_cast<unsigned long long>(n));
            ring.slots = static_cast<std::uint32_t>(n);
        }
        return ring;
    }
    TP_CHECK(*mode == "blv", "config: 'frame' must be 'bl16', 'blv' or 'cuqr', got '%.*s'",
             static_cast<int>(mode->size()), mode->data());
    return geometry_for(Mode::BLV, max_payload);
}

/// A received frame, decoded.
struct FrameView {
    std::uint8_t *data = nullptr; // the payload
    std::uint32_t len = 0;        // payload bytes the frame carries (BL16: always 8, zero-padded)
    std::uint32_t fn_id = 0;      // which function on the coprocessor handles it
    std::uint32_t seq = 0;        // the round's sequence number, cursor + 1
    std::int32_t status = 0;      // CUQR reply: 0 ok, >0 the coprocessor function failed
    std::uint32_t request_id = 0; // CUQR: the sender's opaque request id
    std::uint64_t t_ptp_ns = 0;   // CUQR: the sender's opaque timestamp
};

namespace detail {
inline void require_mode(Mode mode) {
    TP_CHECK(mode == Mode::BL16 || mode == Mode::BLV || mode == Mode::CUQR,
             "frame mode %d is not implemented", static_cast<int>(mode));
}
inline BlvHeader *blv(void *frame) { return static_cast<BlvHeader *>(frame); }
inline CuqrHeader *cuqr(void *frame) { return static_cast<CuqrHeader *>(frame); }
inline std::uint32_t *completion(const RingGeometry &ring, void *frame) {
    return reinterpret_cast<std::uint32_t *>(static_cast<std::uint8_t *>(frame) + ring.stride -
                                             BLV_COMPLETION_BYTES);
}
} // namespace detail

/// Bytes one frame takes in its slot: 16 for BL16, the whole slot for BLV and CUQR.
inline std::size_t frame_bytes(const RingGeometry &ring) {
    detail::require_mode(ring.mode);
    return ring.mode == Mode::BL16 ? sizeof(Payload) : ring.stride;
}

/// Payload bytes one frame carries.
inline std::size_t data_capacity(const RingGeometry &ring) {
    detail::require_mode(ring.mode);
    return ring.mode == Mode::BL16   ? PAYLOAD_DATA_BYTES
           : ring.mode == Mode::CUQR ? ring.stride - sizeof(CuqrHeader)
                                     : ring.stride - sizeof(BlvHeader) - BLV_COMPLETION_BYTES;
}

/// Bytes to send for a staged and sealed frame: CUQR sends only its header and payload.
inline std::size_t wire_bytes(const RingGeometry &ring, const void *frame) {
    if (ring.mode == Mode::CUQR) {
        return sizeof(CuqrHeader) + static_cast<const CuqrHeader *>(frame)->len;
    }
    return frame_bytes(ring);
}

/// Where a frame's payload starts, for a caller that writes or reads it in place.
inline std::uint8_t *data(const RingGeometry &ring, void *frame) {
    detail::require_mode(ring.mode);
    return static_cast<std::uint8_t *>(frame) + ring.data_offset();
}

/// Fill a frame's payload, its length and fn_id. `src` may be the frame's own payload area, for a
/// payload already written in place. BL16 zero-pads the payload to 8 B. seq is left for seal().
/// With `reply_to`, a CUQR frame is a reply (CUQS) echoing its request; BL16 and BLV ignore it.
inline void stage(const RingGeometry &ring, void *frame, std::uint32_t fn_id, const void *src,
                  std::size_t len, const ReplyTo *reply_to = nullptr) {
    const std::size_t cap = data_capacity(ring);
    TP_CHECK(len <= cap, "payload (%zu B) exceeds the %zu B frame", len, cap);
    if (ring.mode == Mode::BL16) {
        Payload *p = static_cast<Payload *>(frame);
        std::uint64_t value = 0;
        if (len != 0) {
            std::memcpy(&value, src, len);
        }
        p->value = value;
        p->decoder_id = fn_id;
        return;
    }
    std::uint8_t *payload = data(ring, frame);
    if (len != 0 && src != payload) {
        std::memmove(payload, src, len);
    }
    if (ring.mode == Mode::CUQR) {
        const bool failed = reply_to && reply_to->status != 0;
        *detail::cuqr(frame) = CuqrHeader{
            .magic = reply_to ? CUQS_MAGIC : CUQR_MAGIC,
            .word = reply_to ? static_cast<std::uint32_t>(reply_to->status) : fn_id,
            .len = failed ? 0U : static_cast<std::uint32_t>(len),
            .request_id = reply_to ? reply_to->request_id : 0,
            .t_ptp_ns = reply_to ? reply_to->t_ptp_ns : 0,
        };
        return;
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
inline void seal(const RingGeometry &ring, void *frame, std::uint32_t seq) {
    detail::require_mode(ring.mode);
    if (ring.mode == Mode::BL16) {
        std::atomic_ref<std::uint32_t>(static_cast<Payload *>(frame)->seq_num)
            .store(seq, std::memory_order_release);
        return;
    }
    if (ring.mode == Mode::CUQR) {
        // The transport signals arrival, so this is a plain store. A reply keeps the request_id it
        // echoes.
        CuqrHeader *h = detail::cuqr(frame);
        if (h->magic == CUQR_MAGIC) {
            h->request_id = seq;
        }
        return;
    }
    detail::blv(frame)->seq = seq;
    std::atomic_ref<std::uint32_t>(*detail::completion(ring, frame))
        .store(seq, std::memory_order_release);
}

/// Whether the frame for sequence number `expect` has arrived in `frame`; if so, decode it into
/// `out`. The polled word is read with acquire ordering, so everything written before it is
/// visible.
inline bool poll(const RingGeometry &ring, void *frame, std::uint32_t expect, FrameView &out) {
    if (ring.mode == Mode::BL16) {
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
    detail::require_mode(ring.mode);
    if (ring.mode == Mode::CUQR) {
        // Called once the transport has signalled the arrival, as the header has no completion
        // word. request_id is the sender's own (a stock client need not count from 1), so it is
        // returned rather than compared; a controller checks the echo of its own.
        const CuqrHeader *h = detail::cuqr(frame);
        TP_CHECK(h->magic == CUQR_MAGIC || h->magic == CUQS_MAGIC,
                 "slot holds no CUQR frame (magic 0x%08x)", h->magic);
        TP_CHECK(h->len <= data_capacity(ring),
                 "CUQR frame claims %u B, more than its %zu B slot holds", h->len,
                 data_capacity(ring));
        const bool reply = h->magic == CUQS_MAGIC;
        out = FrameView{
            .data = data(ring, frame),
            .len = h->len,
            .fn_id = reply ? 0 : h->word,
            .seq = expect,
            .status = reply ? static_cast<std::int32_t>(h->word) : 0,
            .request_id = h->request_id,
            .t_ptp_ns = h->t_ptp_ns,
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
    TP_CHECK(h->len <= data_capacity(ring), "BLV frame claims %u B, more than its %zu B slot holds",
             h->len, data_capacity(ring));
    out = FrameView{
        .data = data(ring, frame),
        .len = h->len,
        .fn_id = h->fn_id,
        .seq = expect,
    };
    return true;
}

} // namespace catalyst::transport::common::frame
