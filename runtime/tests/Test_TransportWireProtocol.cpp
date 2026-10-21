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
#include <vector>

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

TEST_CASE("the BL16 ring is today's ring") {
    namespace frame = catalyst::transport::common::frame;
    CHECK(frame::BL16_RING.region_bytes() == REGION_BYTES);
    CHECK(frame::BL16_RING.slot_offset(1) == sizeof(PayloadSlot));
    CHECK(frame::BL16_RING.slot_offset(K_RING_SLOTS + 3) == 3 * sizeof(PayloadSlot));
}

TEST_CASE("a frame mode without a codec is refused") {
    namespace frame = catalyst::transport::common::frame;
    const frame::RingGeometry unknown{.mode = static_cast<frame::Mode>(3)};
    CHECK_THROWS_WITH(frame::frame_bytes(unknown),
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
                      Catch::Matchers::ContainsSubstring("'bl16', 'blv' or 'cuqr'"));
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

namespace {
// cudaq-realtime's wire structs (cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h),
// copied so the tests check our bytes against theirs.
struct __attribute__((packed)) RPCHeader {
    std::uint32_t magic, function_id, arg_len, request_id;
    std::uint64_t ptp_timestamp;
};
struct __attribute__((packed)) RPCResponse {
    std::uint32_t magic;
    std::int32_t status;
    std::uint32_t result_len, request_id;
    std::uint64_t ptp_timestamp;
};
} // namespace

TEST_CASE("CUQR frames are byte-identical to cudaq-realtime's RPCHeader and RPCResponse") {
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::CUQR, 16);
    CHECK(ring.stride == 64);
    CHECK(frame::data_capacity(ring) == 40);
    std::vector<std::uint8_t> slot(ring.stride, 0xAA);
    const std::uint8_t payload[3] = {1, 2, 3};

    frame::stage(ring, slot.data(), 0x7ED8BE82u, payload, sizeof(payload));
    frame::seal(ring, slot.data(), 42);
    const RPCHeader req{0x43555152u, 0x7ED8BE82u, 3, 42, 0};
    CHECK(std::memcmp(slot.data(), &req, sizeof(req)) == 0);
    CHECK(std::vector<std::uint8_t>(slot.begin(), slot.begin() + 4) ==
          std::vector<std::uint8_t>{0x52, 0x51, 0x55, 0x43}); // "RQUC" on the wire
    CHECK(std::memcmp(slot.data() + 24, payload, sizeof(payload)) == 0);
    CHECK(frame::wire_bytes(ring, slot.data()) == 27);

    const frame::ReplyTo ok{.request_id = 42, .t_ptp_ns = 0x1122334455667788ull, .status = 0};
    frame::stage(ring, slot.data(), 0, payload, sizeof(payload), &ok);
    frame::seal(ring, slot.data(), 7); // a reply keeps the request_id it echoes
    const RPCResponse rep{0x43555153u, 0, 3, 42, 0x1122334455667788ull};
    CHECK(std::memcmp(slot.data(), &rep, sizeof(rep)) == 0);

    const frame::ReplyTo failed{.request_id = 42, .t_ptp_ns = 0, .status = 1};
    frame::stage(ring, slot.data(), 0, payload, sizeof(payload), &failed);
    const RPCResponse err{0x43555153u, 1, 0, 42, 0};
    CHECK(std::memcmp(slot.data(), &err, sizeof(err)) == 0);
    frame::FrameView v;
    REQUIRE(frame::poll(ring, slot.data(), 1, v));
    CHECK(v.status == 1);
    CHECK(v.len == 0);
}

TEST_CASE("a CUQR request_id is opaque: poll decodes whatever the peer sent") {
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_for(frame::Mode::CUQR, 8);
    std::vector<std::uint8_t> slot(ring.stride, 0);
    const RPCHeader req{0x43555152u, 9, 8, 0xDEADBEEFu, 77}; // a stock client's own id
    std::memcpy(slot.data(), &req, sizeof(req));
    frame::FrameView v;
    REQUIRE(frame::poll(ring, slot.data(), /*expect=*/5, v));
    CHECK(v.fn_id == 9);
    CHECK(v.len == 8);
    CHECK(v.request_id == 0xDEADBEEFu);
    CHECK(v.t_ptp_ns == 77);
    CHECK(v.data == slot.data() + 24);
}

TEST_CASE("a CUQR slot without a valid frame is not decoded, so it can be dropped") {
    namespace frame = catalyst::transport::common::frame;
    // cudaq-realtime drops such a message and keeps serving; poll reports it rather than throwing.
    const auto ring = frame::geometry_for(frame::Mode::CUQR, 8);
    std::vector<std::uint8_t> slot(ring.stride, 0);
    frame::FrameView v;
    CHECK_FALSE(frame::poll(ring, slot.data(), 1, v));
    const RPCHeader too_long{0x43555152u, 0, 1000, 1, 0};
    std::memcpy(slot.data(), &too_long, sizeof(too_long));
    CHECK_FALSE(frame::poll(ring, slot.data(), 1, v));
}

TEST_CASE("a session's config can ask for CUQR frames") {
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_from_config("dev=rxe0;frame=cuqr;frame_payload=100");
    CHECK(ring.mode == frame::Mode::CUQR);
    CHECK(ring.stride == 128); // 24 + 100 -> 128
}

TEST_CASE("a CUQR session's config can set the slot count, as a cudaq-realtime server does") {
    namespace frame = catalyst::transport::common::frame;
    const auto ring = frame::geometry_from_config("frame=cuqr;frame_payload=232;frame_slots=8");
    CHECK(ring.slots == 8);
    CHECK(ring.stride == 256); // cudaq-realtime's default cpu_roce slot
    CHECK_THROWS_WITH(frame::geometry_from_config("frame=cuqr;frame_slots=6"),
                      Catch::Matchers::ContainsSubstring("power of two"));
}

TEST_CASE("a CUQR session's config can set the slot size, as a cudaq-realtime server does") {
    namespace frame = catalyst::transport::common::frame;
    // A server's replies fill its whole slot, so our slots match it even for small messages.
    const auto ring =
        frame::geometry_from_config("frame=cuqr;frame_payload=8;frame_slot_bytes=256");
    CHECK(ring.stride == 256);
    CHECK(frame::data_capacity(ring) == 232);
    CHECK_THROWS_WITH(
        frame::geometry_from_config("frame=cuqr;frame_payload=300;frame_slot_bytes=256"),
        Catch::Matchers::ContainsSubstring("324 B frame"));
    CHECK_THROWS_WITH(frame::geometry_from_config("frame=cuqr;frame_slot_bytes=200"),
                      Catch::Matchers::ContainsSubstring("power of two"));
}
