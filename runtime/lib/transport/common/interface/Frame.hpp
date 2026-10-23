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
// Backends build and read frames through these functions rather than through the Payload struct,
// so a frame layout other than the 16 B one can be added here without touching each backend.
// Only the 16 B layout, BL16 (see Payload in WireProtocol.hpp), is implemented so far.

#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "Error.hpp"
#include "WireProtocol.hpp"

namespace catalyst::transport::common::frame {

/// A frame layout. Matches the CATALYST_TRANSPORT_FRAME_* bit a backend's caps report.
enum class Mode : std::uint8_t {
    BL16 = 0, // 16 B: 8 B of data, the fn_id, then seq (Payload)
    BLV = 1,  // variable length; not implemented yet
    CUQR = 2, // cudaq-realtime compatible; not implemented yet
};

/// How a ring of frames is laid out: `slots` frames, each starting `stride` bytes after the last.
struct RingGeometry {
    Mode mode = Mode::BL16;
    std::uint32_t slots = static_cast<std::uint32_t>(K_RING_SLOTS); // a power of two
    std::uint32_t stride = sizeof(PayloadSlot);

    std::size_t region_bytes() const { return std::size_t{slots} * stride; }
    /// Byte offset of the slot round `cursor` uses.
    std::size_t slot_offset(std::uint64_t cursor) const {
        return static_cast<std::size_t>(cursor & (slots - 1)) * stride;
    }
};

/// The ring every BL16 backend uses: 256 slots of 64 B.
inline constexpr RingGeometry BL16_RING{};

/// A received frame, decoded.
struct FrameView {
    std::uint8_t *data = nullptr; // the payload
    std::uint32_t len = 0;        // payload bytes the frame carries (BL16: always 8)
    std::uint32_t fn_id = 0;      // which function on the coprocessor handles it
    std::uint32_t seq = 0;        // the round's sequence number, cursor + 1
};

namespace detail {
inline void require_bl16(Mode mode) {
    TP_CHECK(mode == Mode::BL16, "frame mode %d is not implemented", static_cast<int>(mode));
}
inline Payload *bl16(Mode mode, void *frame) {
    require_bl16(mode);
    return static_cast<Payload *>(frame);
}
} // namespace detail

/// Bytes on the wire for one frame.
inline std::size_t frame_bytes(Mode mode) {
    detail::require_bl16(mode);
    return sizeof(Payload);
}

/// Payload bytes one frame carries.
inline std::size_t data_capacity(Mode mode) {
    detail::require_bl16(mode);
    return PAYLOAD_DATA_BYTES;
}

/// Where a frame's payload starts, for a caller that writes or reads it in place.
inline std::uint8_t *data(Mode mode, void *frame) {
    return reinterpret_cast<std::uint8_t *>(&detail::bl16(mode, frame)->value);
}

/// Fill a frame's payload and fn_id, zero-padding the payload. `src` may be the frame's own data
/// area, for a payload already written in place. seq is left for seal().
inline void stage(Mode mode, void *frame, std::uint32_t fn_id, const void *src, std::size_t len) {
    Payload *p = detail::bl16(mode, frame);
    TP_CHECK(len <= PAYLOAD_DATA_BYTES, "payload (%zu B) exceeds the %zu B frame", len,
             PAYLOAD_DATA_BYTES);
    std::uint64_t value = 0;
    if (len != 0) {
        std::memcpy(&value, src, len);
    }
    p->value = value;
    p->decoder_id = fn_id;
}

/// Set a staged frame's seq, the word a receiver polls, which makes the frame complete.
inline void seal(Mode mode, void *frame, std::uint32_t seq) {
    detail::bl16(mode, frame)->seq_num = seq;
}

/// Whether the frame for sequence number `expect` has arrived in `frame`; if so, decode it into
/// `out`. The seq is read with acquire ordering, so the payload written before it is visible.
inline bool poll(Mode mode, void *frame, std::uint32_t expect, FrameView &out) {
    Payload *p = detail::bl16(mode, frame);
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

} // namespace catalyst::transport::common::frame
