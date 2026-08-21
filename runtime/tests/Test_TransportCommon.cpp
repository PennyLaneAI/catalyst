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

#include <string>

#include "BackendConfig.hpp"
#include "Context.hpp"
#include "QpState.hpp"

#include <catch2/catch_test_macros.hpp>
#include <infiniband/verbs.h>

using namespace catalyst::transport::common;

static bool have_rxe() {
    int n = 0;
    ibv_device **devs = ibv_get_device_list(&n);
    bool found = false;
    for (int i = 0; i < n; i++) {
        if (std::string(ibv_get_device_name(devs[i])) == "rxe0") {
            found = true;
        }
    }
    if (devs) {
        ibv_free_device_list(devs);
    }
    return found;
}

TEST_CASE("parse_backend_config requires an explicit dev and gid", "[common]") {
    SECTION("both keys present, order-independent") {
        const BackendConfig a = parse_backend_config("dev=rxe0;gid=1");
        CHECK(a.dev == "rxe0");
        CHECK(a.gid == 1);
        const BackendConfig b = parse_backend_config("gid=3;dev=mlx5_1");
        CHECK(b.dev == "mlx5_1");
        CHECK(b.gid == 3);
        // 0 is a valid GID index, not a "missing" sentinel.
        CHECK(parse_backend_config("dev=rxe0;gid=0").gid == 0);
    }
    SECTION("unknown keys are ignored, required ones still enforced") {
        const BackendConfig c = parse_backend_config("foo=bar;dev=rxe0;gid=2");
        CHECK(c.dev == "rxe0");
        CHECK(c.gid == 2);
    }
    SECTION("a missing key is rejected rather than defaulted") {
        CHECK_THROWS_AS(parse_backend_config(""), TransportError);
        CHECK_THROWS_AS(parse_backend_config("dev=rxe0"), TransportError);
        CHECK_THROWS_AS(parse_backend_config("gid=1"), TransportError);
        CHECK_THROWS_AS(parse_backend_config("junk"), TransportError);
    }
    SECTION("an empty or malformed value is rejected") {
        CHECK_THROWS_AS(parse_backend_config("dev=;gid=1"), TransportError);
        CHECK_THROWS_AS(parse_backend_config("dev=rxe0;gid="), TransportError);
        // std::atoi would have silently turned each of these into gid 0.
        CHECK_THROWS_AS(parse_backend_config("dev=rxe0;gid=abc"), TransportError);
        CHECK_THROWS_AS(parse_backend_config("dev=rxe0;gid=1x"), TransportError);
        CHECK_THROWS_AS(parse_backend_config("dev=rxe0;gid=-1"), TransportError);
    }
}

TEST_CASE("QpState transitions gate the RC bring-up edges", "[common]") {
    REQUIRE(is_valid_transition(QpState::RESET, QpState::INIT));
    REQUIRE(is_valid_transition(QpState::INIT, QpState::RTR));
    REQUIRE(is_valid_transition(QpState::RTR, QpState::RTS));
    REQUIRE(is_valid_transition(QpState::RTS, QpState::ERROR));
    REQUIRE(is_valid_transition(QpState::RTS, QpState::RESET));
    REQUIRE_FALSE(is_valid_transition(QpState::RESET, QpState::RTR));
    REQUIRE_FALSE(is_valid_transition(QpState::INIT, QpState::RTS));
}

TEST_CASE("Context opens rxe0 with an active port", "[common]") {
    if (!have_rxe()) {
        SKIP("no rxe0 RDMA device");
    }
    Context ctx("rxe0");
    REQUIRE(ctx.get() != nullptr);
    ibv_port_attr pa = ctx.port_attr(1);
    REQUIRE(pa.state == IBV_PORT_ACTIVE);
    REQUIRE(pa.active_mtu > 0);
}
