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

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "Frame.hpp"
#include "WireProtocol.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace catalyst::transport::common;

TEST_CASE("Payload is the 16 B wire frame") {
    STATIC_REQUIRE(sizeof(Payload) == 16);
    STATIC_REQUIRE(offsetof(Payload, value) == 0);
    STATIC_REQUIRE(offsetof(Payload, decoder_id) == 8);
    STATIC_REQUIRE(offsetof(Payload, seq_num) == 12);
}

TEST_CASE("PayloadSlot is a 64 B slot") {
    STATIC_REQUIRE(sizeof(PayloadSlot) == 64);
    STATIC_REQUIRE(alignof(PayloadSlot) == 64);
    STATIC_REQUIRE(offsetof(PayloadSlot, p) == 0);
}

TEST_CASE("Ring geometry and constants match the trampoline") {
    STATIC_REQUIRE(K_RING_SLOTS == 256);
    STATIC_REQUIRE((K_RING_SLOTS & (K_RING_SLOTS - 1)) == 0);
    STATIC_REQUIRE(REGION_BYTES == K_RING_SLOTS * sizeof(PayloadSlot));
    STATIC_REQUIRE(SIGNAL_EVERY == 64);
    STATIC_REQUIRE(SALT == 0xC0DE1515u);
    STATIC_REQUIRE(DEMO_SYNDROME == 0x0123456789ABCDEFull);
}

TEST_CASE("the BL16 frame codec writes exactly today's Payload bytes") {
    namespace frame = catalyst::transport::common::frame;
    const std::uint8_t data[3] = {0xAB, 0xCD, 0xEF};
    alignas(64) std::uint8_t encoded[64];
    std::memset(encoded, 0x5A, sizeof(encoded)); // stale bytes the codec must overwrite
    frame::stage(frame::BL16_RING, encoded, 7, data, sizeof(data));
    frame::seal(frame::BL16_RING, encoded, 42);

    Payload expected{};
    std::memcpy(&expected.value, data, sizeof(data));
    expected.decoder_id = 7;
    expected.seq_num = 42;
    CHECK(frame::frame_bytes(frame::BL16_RING) == sizeof(Payload));
    CHECK(std::memcmp(encoded, &expected, sizeof(Payload)) == 0);

    frame::FrameView view;
    CHECK_FALSE(frame::poll(frame::BL16_RING, encoded, 41, view));
    REQUIRE(frame::poll(frame::BL16_RING, encoded, 42, view));
    CHECK(view.data == encoded);
    CHECK(view.len == PAYLOAD_DATA_BYTES);
    CHECK(view.fn_id == 7);
    CHECK(view.seq == 42);
}

TEST_CASE("the BL16 codec zero-pads a payload written in place") {
    namespace frame = catalyst::transport::common::frame;
    for (std::size_t len : {0, 3, 8}) {
        alignas(64) std::uint8_t encoded[64];
        std::memset(encoded, 0x5A, sizeof(encoded));
        std::uint8_t *payload = frame::data(frame::BL16_RING, encoded);
        frame::stage(frame::BL16_RING, encoded, 7, payload, len);

        std::uint8_t expected[PAYLOAD_DATA_BYTES] = {};
        std::memset(expected, 0x5A, len);
        CHECK(std::memcmp(payload, expected, sizeof(expected)) == 0);
    }
}

TEST_CASE("the BL16 ring is today's ring") {
    namespace frame = catalyst::transport::common::frame;
    CHECK(frame::BL16_RING.region_bytes() == REGION_BYTES);
    CHECK(frame::BL16_RING.slot_offset(1) == sizeof(PayloadSlot));
    CHECK(frame::BL16_RING.slot_offset(K_RING_SLOTS + 3) == 3 * sizeof(PayloadSlot));
}

TEST_CASE("a frame mode without a codec is refused") {
    namespace frame = catalyst::transport::common::frame;
    const frame::RingGeometry cuqr{.mode = frame::Mode::CUQR};
    CHECK_THROWS_WITH(frame::frame_bytes(cuqr),
                      Catch::Matchers::ContainsSubstring("not implemented"));
}

TEST_CASE("a BLV ring uses the smallest power-of-two slot that holds its frames") {
    namespace frame = catalyst::transport::common::frame;
    // 32 B header + payload + 4 B completion word, rounded up to a power of two of at least 64.
    CHECK(frame::geometry_for(frame::Mode::BLV, 8).stride == 64);
    CHECK(frame::geometry_for(frame::Mode::BLV, 28).stride == 64);
    CHECK(frame::geometry_for(frame::Mode::BLV, 29).stride == 128);
    CHECK(frame::geometry_for(frame::Mode::BLV, 1024).stride == 2048);
    const auto ring = frame::geometry_for(frame::Mode::BLV, 64);
    CHECK(ring.mode == frame::Mode::BLV);
    CHECK(ring.slots == K_RING_SLOTS);
    CHECK(frame::data_capacity(ring) == 128 - 32 - 4);
    CHECK(frame::frame_bytes(ring) == 128);
    CHECK(ring.region_bytes() == K_RING_SLOTS * 128);
    CHECK_THROWS_WITH(frame::geometry_for(frame::Mode::BL16, 9),
                      Catch::Matchers::ContainsSubstring("at most 8 B"));
}

TEST_CASE("a session's config names its frame layout") {
    namespace frame = catalyst::transport::common::frame;
    const auto blv = frame::geometry_from_config("dev=rxe0;frame=blv;frame_payload=64;gid=1");
    CHECK(blv.mode == frame::Mode::BLV);
    CHECK(blv.stride == 128);
    CHECK(frame::geometry_from_config("dev=rxe0;gid=1").stride == frame::BL16_RING.stride);
    CHECK(frame::geometry_from_config("frame=bl16").mode == frame::Mode::BL16);
    CHECK_THROWS_WITH(frame::geometry_from_config("frame=wide"),
                      Catch::Matchers::ContainsSubstring("'bl16' or 'blv'"));
    CHECK_THROWS_WITH(frame::geometry_from_config("frame=blv;frame_payload=lots"),
                      Catch::Matchers::ContainsSubstring("'frame_payload'"));
}

TEST_CASE("a BLV frame carries its payload, length and fn_id, and completes last") {
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::BLV, 64);
    alignas(64) std::uint8_t slot[128];
    std::memset(slot, 0x5A, sizeof(slot));
    std::uint8_t payload[50];
    for (std::size_t i = 0; i < sizeof(payload); ++i) {
        payload[i] = static_cast<std::uint8_t>(i);
    }
    frame::stage(ring, slot, 3, payload, sizeof(payload));

    frame::FrameView view;
    CHECK_FALSE(frame::poll(ring, slot, 9, view)); // not sealed yet
    frame::seal(ring, slot, 9);
    CHECK_FALSE(frame::poll(ring, slot, 8, view));
    REQUIRE(frame::poll(ring, slot, 9, view));
    CHECK(view.data == slot + sizeof(frame::BlvHeader));
    CHECK(view.len == sizeof(payload));
    CHECK(view.fn_id == 3);
    CHECK(view.seq == 9);
    CHECK(std::memcmp(view.data, payload, sizeof(payload)) == 0);

    frame::BlvHeader header;
    std::memcpy(&header, slot, sizeof(header));
    CHECK(header.magic == frame::BLV_MAGIC);
    CHECK(std::memcmp(slot, "BLV1", 4) == 0);
    CHECK(header.seq == 9);
    std::uint32_t completion = 0;
    std::memcpy(&completion, slot + sizeof(slot) - sizeof(completion), sizeof(completion));
    CHECK(completion == 9);
}

TEST_CASE("a BLV payload can be written in place before staging") {
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::BLV, 16);
    alignas(64) std::uint8_t slot[64] = {};
    std::uint8_t *payload = frame::data(ring, slot);
    std::memset(payload, 0x11, 16);
    frame::stage(ring, slot, 1, payload, 16);
    frame::seal(ring, slot, 1);
    frame::FrameView view;
    REQUIRE(frame::poll(ring, slot, 1, view));
    CHECK(view.len == 16);
    CHECK(view.data[15] == 0x11);
}

TEST_CASE("a BLV slot without a valid header is refused") {
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::BLV, 16);
    alignas(64) std::uint8_t slot[64] = {};
    frame::seal(ring, slot, 1); // a completion word, but no header
    frame::FrameView view;
    CHECK_THROWS_WITH(frame::poll(ring, slot, 1, view),
                      Catch::Matchers::ContainsSubstring("no BLV frame"));
}
