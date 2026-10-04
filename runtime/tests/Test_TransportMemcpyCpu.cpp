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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "CpuControllerSession.hpp"
#include "CpuCoprocessorSession.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace catalyst::transport;
using namespace catalyst::transport::memcpy;

// Provided by libsteane_coprocessor_cpu — the same CoprocessorFn plugin production
// dlopens. Linking it in lets the test exercise a real decoder end-to-end.
extern "C" int steane_coprocessor(const void *in, std::size_t in_len, void *out,
                                  std::size_t out_cap, void *ctx);

namespace {
// Per-test pair keys. Matching keys between the controller and its paired coprocessor is what
// the memcpy backend uses to rendezvous, mirroring what inject-transport-session emits at
// compile time as the `key` attribute on each transport.create.
std::string pair_cfg(std::uint16_t port) { return "pair=p" + std::to_string(port); }

int invert_fn(const void *in, std::size_t in_len, void *out, std::size_t out_cap, void * /*ctx*/) {
    std::uint64_t v = 0;
    std::memcpy(&v, in, std::min(in_len, sizeof(v)));
    v = ~v;
    const std::size_t n = std::min(out_cap, sizeof(v));
    std::memcpy(out, &v, n);
    return 0;
}

int failing_fn(const void *, std::size_t, void *, std::size_t, void *) { return 1; }
} // namespace

TEST_CASE("memcpy round-trip echoes through peer memory", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19001};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    CpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));

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
    coprocessor.set_coprocessor_fn(nullptr, nullptr); // built-in echo
    controller.start();
    coprocessor.start();

    const std::uint64_t request_word = 0x0123456789ABCDEFull;
    controller.write_data_slot(&request_word, sizeof(request_word), /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);

    std::uint64_t reply_word = 0;
    void *outs[1] = {&reply_word};
    std::uint64_t out_bytes[1] = {sizeof(reply_word)};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);

    CHECK(reply_word == request_word);
}

TEST_CASE("memcpy uses the bound coprocessor function", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19002};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    CpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));

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
    coprocessor.set_coprocessor_fn(invert_fn, nullptr);
    controller.start();
    coprocessor.start();

    const std::uint64_t request_word = 0x0123456789ABCDEFull;
    controller.write_data_slot(&request_word, sizeof(request_word), /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);

    std::uint64_t reply_word = 0;
    void *outs[1] = {&reply_word};
    std::uint64_t out_bytes[1] = {sizeof(reply_word)};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);

    CHECK(reply_word == ~request_word);
}

TEST_CASE("memcpy treats the coprocessor return value as a status code", "[transport_memcpy]") {
    CpuCoprocessorSession coprocessor("pair=status-code");
    coprocessor.set_coprocessor_fn(failing_fn, nullptr);
    coprocessor.start();

    common::Payload request{};
    std::uint64_t reply = 0;
    REQUIRE_THROWS_AS(coprocessor.process_message(&request, sizeof(request), &reply, sizeof(reply)),
                      std::runtime_error);
}

TEST_CASE("memcpy round-trip drives the steane decoder plugin", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19003};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    CpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));

    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(sizeof(std::int64_t), MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);

    MemRegion request = coprocessor.alloc_memory(sizeof(std::int64_t), MemKind::CpuRam);
    PeerRef peer_reply = coprocessor.exchange_keys(request);

    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    coprocessor.establish_channel(desc, request, peer_reply);

    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::int64_t));
    coprocessor.set_coprocessor_fn(&steane_coprocessor, nullptr);
    controller.start();
    coprocessor.start();

    // Syndrome [1, 0, 1] packs (check 0 as MSB) to 0b101 == 5, which the LUT maps to qubit 3.
    // Padded into the 8-byte value slot so the coprocessor sees the syndrome at frame offset 0.
    const std::uint8_t syndrome_bytes[8] = {1, 0, 1, 0, 0, 0, 0, 0};
    controller.write_data_slot(syndrome_bytes, sizeof(syndrome_bytes), /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);

    std::int64_t err_idx = 0;
    void *outs[1] = {&err_idx};
    std::uint64_t out_bytes[1] = {sizeof(err_idx)};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);

    CHECK(err_idx == 3);
}

// ---- Pairing invariants ------------------------------------------------------

TEST_CASE("memcpy rejects a second controller on the same pair", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19010};
    CpuControllerSession first(pair_cfg(ci.oob_port));
    CpuControllerSession second(pair_cfg(ci.oob_port));
    REQUIRE(first.connect(ci) == 0);
    REQUIRE_THROWS_AS(second.connect(ci), std::runtime_error);
}

TEST_CASE("memcpy rejects a second coprocessor on the same pair", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19011};
    CpuCoprocessorSession first(pair_cfg(ci.oob_port));
    CpuCoprocessorSession second(pair_cfg(ci.oob_port));
    REQUIRE(first.connect(ci) == 0);
    REQUIRE_THROWS_AS(second.connect(ci), std::runtime_error);
}

TEST_CASE("memcpy accepts a rebind after the prior session is destroyed", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19012};
    {
        CpuControllerSession ctrl(pair_cfg(ci.oob_port));
        CpuCoprocessorSession co(pair_cfg(ci.oob_port));
        REQUIRE(ctrl.connect(ci) == 0);
        REQUIRE(co.connect(ci) == 0);
    }
    CpuControllerSession ctrl2(pair_cfg(ci.oob_port));
    CpuCoprocessorSession co2(pair_cfg(ci.oob_port));
    REQUIRE_NOTHROW(ctrl2.connect(ci));
    REQUIRE_NOTHROW(co2.connect(ci));
}

// ---- Data-path preconditions -------------------------------------------------

TEST_CASE("memcpy rejects a non-zero work_item_idx", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19013};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    CpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    (void)controller.exchange_keys(reply);
    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, PeerRef{});

    REQUIRE_THROWS_AS(controller.commit_work_item(/*work_item_idx=*/1, sizeof(std::uint64_t),
                                                  sizeof(std::uint64_t)),
                      std::runtime_error);

    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    REQUIRE_THROWS_AS(controller.kick(/*work_item_idx=*/1), std::runtime_error);
}

TEST_CASE("memcpy rejects a second commit_work_item", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19014};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    CpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    REQUIRE_THROWS_AS(controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t)),
                      std::runtime_error);
}

TEST_CASE("memcpy collect rejects more than a single reply slot", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19015};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    CpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    (void)controller.exchange_keys(reply);
    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, PeerRef{});
    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));

    std::uint64_t r0 = 0;
    std::uint64_t r1 = 0;
    void *outs[2] = {&r0, &r1};
    std::uint64_t out_bytes[2] = {sizeof(r0), sizeof(r1)};
    REQUIRE_THROWS_AS(controller.collect(outs, out_bytes, 2), std::runtime_error);
}

TEST_CASE("memcpy incumbent coprocessor survives a rejected second coprocessor",
          "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19020};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    CpuCoprocessorSession incumbent(pair_cfg(ci.oob_port));
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(incumbent.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);
    MemRegion request = incumbent.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_reply = incumbent.exchange_keys(request);
    ChannelDesc desc{.transport = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    incumbent.establish_channel(desc, request, peer_reply);
    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    incumbent.set_coprocessor_fn(nullptr, nullptr); // built-in echo
    controller.start();
    incumbent.start();

    {
        // A rejected connect's dtor must not clear the incumbent's binding.
        CpuCoprocessorSession rejected(pair_cfg(ci.oob_port));
        REQUIRE_THROWS_AS(rejected.connect(ci), std::runtime_error);
    }

    const std::uint64_t word = 0xDEADBEEFCAFEBABEull;
    controller.write_data_slot(&word, sizeof(word), /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);
    std::uint64_t got = 0;
    void *outs[1] = {&got};
    std::uint64_t out_bytes[1] = {sizeof(got)};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);
    CHECK(got == word);
}

// Match RDMA: reject in/out_bytes > wire payload instead of truncating in kick().
TEST_CASE("memcpy rejects in/out_bytes exceeding the wire payload", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19021};
    CpuControllerSession controller(pair_cfg(ci.oob_port));
    CpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    REQUIRE_THROWS_AS(controller.commit_work_item(0, /*in_bytes=*/32, sizeof(std::uint64_t)),
                      std::runtime_error);
    REQUIRE_THROWS_AS(controller.commit_work_item(0, sizeof(std::uint64_t), /*out_bytes=*/32),
                      std::runtime_error);
}

// Distinct pair keys stay isolated even on the same peer+oob_port.
TEST_CASE("memcpy pair keys isolate independent sessions", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19022};

    CpuControllerSession ctrl_alpha("pair=alpha");
    CpuCoprocessorSession co_alpha("pair=alpha");
    REQUIRE(ctrl_alpha.connect(ci) == 0);
    REQUIRE(co_alpha.connect(ci) == 0);

    CpuControllerSession ctrl_beta("pair=beta");
    CpuCoprocessorSession co_beta("pair=beta");
    REQUIRE_NOTHROW(ctrl_beta.connect(ci));
    REQUIRE_NOTHROW(co_beta.connect(ci));
}

// No pair key means no way to rendezvous; refuse rather than silently cross-pair.
TEST_CASE("memcpy rejects a session with no pair key in config", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19023};
    CpuControllerSession controller;
    REQUIRE_THROWS_AS(controller.connect(ci), std::runtime_error);
}

// The worker thread reads fn_/ctx_ without synchronization; mutating them after start() would
// race and could tear across the two reads. Enforce the interface's bind-before-start contract.
TEST_CASE("memcpy rejects set_coprocessor_fn after start()", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19024};
    CpuCoprocessorSession coprocessor(pair_cfg(ci.oob_port));
    REQUIRE(coprocessor.connect(ci) == 0);
    coprocessor.start();
    REQUIRE_THROWS_AS(coprocessor.set_coprocessor_fn(nullptr, nullptr), std::runtime_error);
    coprocessor.stop();
    // After stop(), the worker is gone and rebinding is safe again.
    REQUIRE_NOTHROW(coprocessor.set_coprocessor_fn(nullptr, nullptr));
}
