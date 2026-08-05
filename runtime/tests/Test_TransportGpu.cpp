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

#include "GpuCoprocessorSession.hpp"
#include "GpuRuntime.hpp"
#include "WireProtocol.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace catalyst::transport::gpu_verbs;
using namespace catalyst::transport::common;

TEST_CASE("Wire Payload is the 16 B frame") {
    STATIC_REQUIRE(sizeof(Payload) == 16);
    STATIC_REQUIRE(offsetof(Payload, value) == 0);
    STATIC_REQUIRE(offsetof(Payload, decoder_id) == 8);
    STATIC_REQUIRE(offsetof(Payload, seq_num) == 12);
}

TEST_CASE("PayloadSlot is a 64 B aligned ring slot") {
    STATIC_REQUIRE(sizeof(PayloadSlot) == 64);
    STATIC_REQUIRE(alignof(PayloadSlot) == 64);
    STATIC_REQUIRE(K_RING_SLOTS == 256);
    STATIC_REQUIRE(REGION_BYTES == K_RING_SLOTS * sizeof(PayloadSlot));
}

TEST_CASE("HandoffSlot carries correction + trailing seq, fits one 16 B store") {
    STATIC_REQUIRE(offsetof(HandoffSlot, correction) == 0);
    STATIC_REQUIRE(sizeof(HandoffSlot) == 16);
    STATIC_REQUIRE(alignof(HandoffSlot) == 16);
}
