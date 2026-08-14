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
#include <stdexcept>
#include <string>

#include "CpuControllerSession.hpp"
#include "GpuCoprocessorSession.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace catalyst::transport;
using namespace catalyst::transport::memcpy;

namespace {
std::string pair_cfg(std::uint16_t port) { return "pair=p" + std::to_string(port); }
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

TEST_CASE("memcpy rejects a second local GPU coprocessor on the same pair", "[transport_memcpy]") {
    ConnectInfo ci{.peer = "loopback", .oob_port = 19016};
    GpuCoprocessorSession first(pair_cfg(ci.oob_port));
    GpuCoprocessorSession second(pair_cfg(ci.oob_port));
    REQUIRE(first.connect(ci) == 0);
    REQUIRE_THROWS_AS(second.connect(ci), std::runtime_error);
}
