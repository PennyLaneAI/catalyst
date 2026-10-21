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
#include "ConfigParser.hpp"
#include "Context.hpp"
#include "FpgaBackendConfig.hpp"
#include "QpState.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
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

TEST_CASE("configparser::find returns the last value of a key", "[common]") {
    CHECK(configparser::find("a=1;b=2;a=3", "a") == "3");
    CHECK(configparser::find("a=1;b=x=y", "b") == "x=y");
    CHECK_FALSE(configparser::find("a=1", "b").has_value());
}

TEST_CASE("configparser::parse_u64 reads decimal and 0x hex, naming the key on error", "[common]") {
    CHECK(configparser::parse_u64("42", "k") == 42);
    CHECK(configparser::parse_u64("0x80000000", "k") == 0x80000000ull);
    CHECK_THROWS_AS(configparser::parse_u64("", "k"), TransportError);
    CHECK_THROWS_AS(configparser::parse_u64("0x", "k"), TransportError);
    CHECK_THROWS_AS(configparser::parse_u64("12ab", "k"), TransportError);
    CHECK_THROWS_AS(configparser::parse_u64("-1", "k"), TransportError);
}

TEST_CASE("parse_fpga_config has defaults and reads the FPGA keys", "[common]") {
    const FpgaConfig d = parse_fpga_config("");
    CHECK(d.dev == "xib_0");
    CHECK(d.gid == 1);
    CHECK(d.ring == K_RING_SLOTS);
    CHECK(d.stride_log2 == 6);
    CHECK_FALSE(d.data_mem.has_value());

    const FpgaConfig c =
        parse_fpga_config("dev=mlx5_1;gid=3;ring=128;stride_log2=7;data_mem=pl;reply_mem=bram");
    CHECK(c.dev == "mlx5_1");
    CHECK(c.gid == 3);
    CHECK(c.ring == 128);
    CHECK(c.stride_log2 == 7);
    CHECK(c.data_mem == catalyst::transport::MemKind::Ddr);
    CHECK(c.reply_mem == catalyst::transport::MemKind::Other);

    CHECK_THROWS(parse_fpga_config("data_mem=dram"));
    CHECK_THROWS_AS(parse_fpga_config("gid=abc"), TransportError);
}

TEST_CASE("cuqr_tx picks the CUQR completion verb", "[common]") {
    CHECK_FALSE(parse_backend_config("dev=rxe0;gid=1").cuqr_write_imm);
    CHECK_FALSE(parse_backend_config("dev=rxe0;gid=1;cuqr_tx=send").cuqr_write_imm);
    CHECK(parse_backend_config("dev=rxe0;gid=1;cuqr_tx=write_imm").cuqr_write_imm);
    CHECK_THROWS_WITH(parse_backend_config("dev=rxe0;gid=1;cuqr_tx=read"),
                      Catch::Matchers::ContainsSubstring("'send' or 'write_imm'"));
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
