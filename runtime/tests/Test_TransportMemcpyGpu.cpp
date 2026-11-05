// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "CpuControllerSession.hpp"
#include "GpuCoprocessorSession.hpp"
#include "WireProtocol.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace catalyst::transport;
using namespace catalyst::transport::memcpy;

namespace {
std::string pair_cfg(std::uint16_t port) { return "pair=p" + std::to_string(port); }

// The config of a session carrying up to `max_payload` bytes each way in BLV frames.
std::string blv_cfg(std::uint16_t port, std::size_t max_payload) {
    return pair_cfg(port) + ";frame=blv;frame_payload=" + std::to_string(max_payload);
}

// Replies with each payload byte plus one, for the payload size given as ctx.
std::size_t increment_fn(const void *in, std::size_t /*in_len*/, std::uint32_t /*fn_id*/, void *out,
                         std::size_t out_cap, void *ctx) {
    const std::size_t n = *static_cast<const std::size_t *>(ctx);
    if (out_cap < n) {
        return 0;
    }
    const auto *request = static_cast<const std::uint8_t *>(in);
    auto *reply = static_cast<std::uint8_t *>(out);
    for (std::size_t i = 0; i < n; ++i) {
        reply[i] = static_cast<std::uint8_t>(request[i] + 1); // wraps 255 to 0
    }
    return n;
}
} // namespace

TEST_CASE("memcpy CPU controller can drive the local GPU coprocessor", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19003};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    GpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));

    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);

    MemRegion request = coprocessor.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_reply = coprocessor.exchange_keys(request);

    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    coprocessor.establish_channel(desc, request, peer_reply);

    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    controller.start();
    coprocessor.start();
    coprocessor.set_coprocessor_launcher(nullptr, nullptr); // built-in GPU echo

    const std::uint64_t request_word = 0x0123456789ABCDEFull;
    controller.write_data_slot(&request_word, sizeof(request_word), /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);

    std::uint64_t reply_word = 0;
    void *outs[1] = {&reply_word};
    std::uint64_t out_bytes[1] = {sizeof(reply_word)};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);

    CHECK(reply_word == request_word);
}

TEST_CASE("memcpy GPU coprocessor wraps a ring of the slot count its config sets",
          "[transport_memcpy]") {
    for (const std::uint32_t slots : {1u, 8u}) {
        ConnectInfo ci{.peer = "loopback", .oob_port = static_cast<std::uint16_t>(19110 + slots)};
        const std::string cfg = pair_cfg(ci.oob_port) + ";frame_slots=" + std::to_string(slots);
        CpuControllerSession controller(cfg);
        GpuCoprocessorSession coprocessor(cfg);
        REQUIRE(controller.connect(ci) == 0);
        REQUIRE(coprocessor.connect(ci) == 0);
        MemRegion reply = controller.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
        PeerRef peer_request = controller.exchange_keys(reply);
        MemRegion request = coprocessor.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
        PeerRef peer_reply = coprocessor.exchange_keys(request);
        ChannelDesc desc{.transport = "memcpy"};
        controller.establish_channel(desc, reply, peer_request);
        coprocessor.establish_channel(desc, request, peer_reply);

        controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
        coprocessor.set_coprocessor_launcher(nullptr, nullptr); // built-in GPU echo
        controller.start();
        coprocessor.start();
        // Enough rounds to reuse every slot several times.
        for (std::uint64_t round = 0; round < 3 * slots + 1; ++round) {
            const std::uint64_t request_word = 0x3000 + round;
            controller.write_data_slot(&request_word, sizeof(request_word), /*decoder_id=*/0);
            REQUIRE(controller.kick(0) == 0);
            std::uint64_t reply_word = 0;
            void *outs[1] = {&reply_word};
            std::uint64_t out_bytes[1] = {sizeof(reply_word)};
            REQUIRE(controller.collect(outs, out_bytes, 1) == 0);
            REQUIRE(reply_word == request_word);
        }
        coprocessor.stop();
    }
}

TEST_CASE("memcpy GPU coprocessor returns a reply shorter than its 8 B correction",
          "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19032};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    GpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));

    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);

    MemRegion request = coprocessor.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_reply = coprocessor.exchange_keys(request);

    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    coprocessor.establish_channel(desc, request, peer_reply);

    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint32_t));
    controller.start();
    coprocessor.start();
    coprocessor.set_coprocessor_launcher(nullptr, nullptr); // built-in GPU echo

    const std::uint64_t request_word = 0x0123456789ABCDEFull;
    controller.write_data_slot(&request_word, sizeof(request_word), /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);

    std::uint32_t reply_word = 0;
    void *outs[1] = {&reply_word};
    std::uint64_t out_bytes[1] = {sizeof(reply_word)};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);

    CHECK(reply_word == static_cast<std::uint32_t>(request_word)); // the leading 4 bytes
}

TEST_CASE("a launch-once GPU coprocessor echoes BLV messages through its persistent kernel",
          "[transport_memcpy]") {
    constexpr std::size_t in_bytes = 300;
    constexpr std::size_t out_bytes = in_bytes + 10;
    ConnectInfo ci{.peer = "loopback", .oob_port = 19038};
    CpuControllerSession controller(blv_cfg(ci.oob_port, out_bytes));
    GpuCoprocessorSession coprocessor(blv_cfg(ci.oob_port, out_bytes));
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(out_bytes, MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);
    MemRegion request = coprocessor.alloc_memory(in_bytes, MemKind::CpuRam);
    PeerRef peer_reply = coprocessor.exchange_keys(request);
    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    coprocessor.establish_channel(desc, request, peer_reply);

    controller.commit_work_item(0, in_bytes, out_bytes);
    coprocessor.set_coprocessor_launcher(nullptr, nullptr); // built-in GPU echo
    controller.start();
    coprocessor.start();

    // Past ring_slots rounds, so every slot is reused with a new payload.
    for (std::uint32_t round = 0; round < 300; ++round) {
        std::uint8_t payload[in_bytes];
        for (std::size_t i = 0; i < in_bytes; ++i) {
            payload[i] = static_cast<std::uint8_t>(i * 7 + round);
        }
        controller.write_data_slot(payload, in_bytes, /*decoder_id=*/round);
        REQUIRE(controller.kick(0) == 0);
        std::uint8_t got[out_bytes];
        std::memset(got, 0xAA, sizeof(got));
        void *outs[1] = {got};
        std::uint64_t outs_bytes[1] = {out_bytes};
        REQUIRE(controller.collect(outs, outs_bytes, 1) == 0);
        REQUIRE(std::memcmp(got, payload, in_bytes) == 0);
        for (std::size_t i = in_bytes; i < out_bytes; ++i) {
            REQUIRE(got[i] == 0); // zero past what the kernel's echo wrote
        }
    }
}

TEST_CASE("memcpy rejects a second local GPU coprocessor on the same pair", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19016};
    GpuCoprocessorSession first(pair_cfg(ci.oob_port));
    GpuCoprocessorSession second(pair_cfg(ci.oob_port));
    REQUIRE(first.connect(ci) == 0);
    REQUIRE_THROWS_AS(second.connect(ci), std::runtime_error);
}

TEST_CASE("a per-message GPU coprocessor runs a host function on wide messages",
          "[transport_memcpy]") {
    constexpr std::size_t bytes = 300;
    ConnectInfo ci{.peer = "loopback", .oob_port = 19031};
    CpuControllerSession controller(blv_cfg(ci.oob_port, bytes));
    GpuCoprocessorSession coprocessor(blv_cfg(ci.oob_port, bytes), /*gpu_device=*/0,
                                      /*per_message=*/true);
    CHECK(coprocessor.coprocessor_fn_convention() == CoprocConvention::PerMessage);
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(bytes, MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);
    MemRegion request = coprocessor.alloc_memory(bytes, MemKind::CpuRam);
    PeerRef peer_reply = coprocessor.exchange_keys(request);
    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    coprocessor.establish_channel(desc, request, peer_reply);

    std::size_t n = bytes;
    controller.commit_work_item(0, bytes, bytes);
    coprocessor.set_coprocessor_fn(increment_fn, &n);
    REQUIRE_THROWS_AS(coprocessor.set_coprocessor_launcher(nullptr, nullptr), std::runtime_error);
    controller.start();
    coprocessor.start();

    std::uint8_t payload[bytes];
    for (std::size_t i = 0; i < bytes; ++i) {
        payload[i] = static_cast<std::uint8_t>(i);
    }
    controller.write_data_slot(payload, bytes, /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);

    std::uint8_t got[bytes] = {};
    void *outs[1] = {got};
    std::uint64_t out_bytes[1] = {bytes};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);
    for (std::size_t i = 0; i < bytes; ++i) {
        REQUIRE(got[i] == static_cast<std::uint8_t>(i + 1));
    }
}

TEST_CASE("a launch-once GPU coprocessor refuses a per-message function and wide messages",
          "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19046};
    // The controller's frames carry the wider message, the coprocessor's only 8 B, as with a
    // hand-written config; the coprocessor refuses it.
    CpuControllerSession controller(blv_cfg(ci.oob_port, 16));
    GpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));
    CHECK(coprocessor.coprocessor_fn_convention() == CoprocConvention::LaunchOnce);
    REQUIRE_THROWS_AS(coprocessor.set_coprocessor_fn(increment_fn, nullptr), std::runtime_error);
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(16, MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);
    MemRegion request = coprocessor.alloc_memory(16, MemKind::CpuRam);
    PeerRef peer_reply = coprocessor.exchange_keys(request);
    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    coprocessor.establish_channel(desc, request, peer_reply);

    controller.commit_work_item(0, 16, 8);
    controller.start();
    coprocessor.set_coprocessor_launcher(nullptr, nullptr); // built-in GPU echo
    coprocessor.start();
    const std::uint8_t payload[16] = {};
    controller.write_data_slot(payload, sizeof(payload), /*decoder_id=*/0);
    REQUIRE_THROWS_AS(controller.kick(0), std::runtime_error);
}
