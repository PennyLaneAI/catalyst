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

#include "LocalCpuControllerSession.hpp"
#include "LocalCpuCoprocessorSession.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace catalyst::transport;
using namespace catalyst::transport::local_copy;

namespace {
std::size_t invert_fn(const void *in, std::size_t in_len, void *out, std::size_t out_cap,
                      void * /*ctx*/) {
    std::uint64_t v = 0;
    std::memcpy(&v, in, std::min(in_len, sizeof(v)));
    v = ~v;
    const std::size_t n = std::min(out_cap, sizeof(v));
    std::memcpy(out, &v, n);
    return n;
}
} // namespace

TEST_CASE("local_copy round-trip echoes through peer memory", "[transport_local_copy]") {
    LocalCpuControllerSession controller;
    LocalCpuCoprocessorSession coprocessor;

    ConnectInfo ci{.peer = "loopback", .oob_port = 19001};
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);

    MemRegion request = coprocessor.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_reply = coprocessor.exchange_keys(request);

    ChannelDesc desc{.data_path = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    coprocessor.establish_channel(desc, request, peer_reply);

    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    controller.start();
    coprocessor.start();
    coprocessor.set_coprocessor_fn(nullptr, nullptr); // built-in echo

    const std::uint64_t request_word = 0x0123456789ABCDEFull;
    controller.write_data_slot(&request_word, sizeof(request_word), /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);

    std::uint64_t reply_word = 0;
    void *outs[1] = {&reply_word};
    std::uint64_t out_bytes[1] = {sizeof(reply_word)};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);

    CHECK(reply_word == request_word);
}

TEST_CASE("local_copy uses the bound coprocessor function", "[transport_local_copy]") {
    LocalCpuControllerSession controller;
    LocalCpuCoprocessorSession coprocessor;

    ConnectInfo ci{.peer = "loopback", .oob_port = 19002};
    REQUIRE(controller.connect(ci) == 0);
    REQUIRE(coprocessor.connect(ci) == 0);

    MemRegion reply = controller.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_request = controller.exchange_keys(reply);

    MemRegion request = coprocessor.alloc_memory(sizeof(std::uint64_t), MemKind::CpuRam);
    PeerRef peer_reply = coprocessor.exchange_keys(request);

    ChannelDesc desc{.data_path = "memcpy"};
    controller.establish_channel(desc, reply, peer_request);
    coprocessor.establish_channel(desc, request, peer_reply);

    controller.commit_work_item(0, sizeof(std::uint64_t), sizeof(std::uint64_t));
    controller.start();
    coprocessor.start();
    coprocessor.set_coprocessor_fn(invert_fn, nullptr);

    const std::uint64_t request_word = 0x0123456789ABCDEFull;
    controller.write_data_slot(&request_word, sizeof(request_word), /*decoder_id=*/0);
    REQUIRE(controller.kick(0) == 0);

    std::uint64_t reply_word = 0;
    void *outs[1] = {&reply_word};
    std::uint64_t out_bytes[1] = {sizeof(reply_word)};
    REQUIRE(controller.collect(outs, out_bytes, 1) == 0);

    CHECK(reply_word == ~request_word);
}
