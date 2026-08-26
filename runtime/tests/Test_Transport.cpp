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

// Unit tests for the transport CAPI session registry and per-call behavior

#include <cstdint>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include "TransportCAPI.h"

using Catch::Matchers::ContainsSubstring;

namespace {
CatalystTransportSession *make(std::int32_t role, const char *key) {
    return __catalyst__transport__create(STUB_BACKEND_PATH, "cfg", role, key);
}

CatalystTransportSession *make_memcpy_controller(const char *key) {
    return __catalyst__transport__create(MEMCPY_CONTROLLER_BACKEND_PATH, "",
                                         CATALYST_TRANSPORT_ROLE_CONTROLLER, key);
}

CatalystTransportSession *make_memcpy_coprocessor(const char *key) {
    return __catalyst__transport__create(MEMCPY_COPROCESSOR_BACKEND_PATH, "",
                                         CATALYST_TRANSPORT_ROLE_COPROCESSOR, key);
}
} // namespace

TEST_CASE("check aborts on a non-OK status and a null session", "[transport]") {
    __catalyst__transport__check(CATALYST_TRANSPORT_OK, "ok");
    auto *s = make(CATALYST_TRANSPORT_ROLE_CONTROLLER, "check_live");
    REQUIRE(s != nullptr);
    __catalyst__transport__check_session(s, "ok");
    __catalyst__transport__destroy(s);

    REQUIRE_THROWS_WITH(__catalyst__transport__check(CATALYST_TRANSPORT_ERR, "stage_payload"),
                        ContainsSubstring("stage_payload failed (error)"));
    REQUIRE_THROWS_WITH(__catalyst__transport__check(CATALYST_TRANSPORT_ERR_TIMEOUT, "collect"),
                        ContainsSubstring("collect failed (timeout)"));
    REQUIRE_THROWS_WITH(__catalyst__transport__check(CATALYST_TRANSPORT_ERR_STUCK, "collect"),
                        ContainsSubstring("stuck"));
    REQUIRE_THROWS_WITH(__catalyst__transport__check(CATALYST_TRANSPORT_ERR_MEMORY, "post"),
                        ContainsSubstring("memory"));
    REQUIRE_THROWS_WITH(__catalyst__transport__check_session(nullptr, "get_session"),
                        ContainsSubstring("get_session: null session"));
}

TEST_CASE("create registers a session resolvable by (role, key)", "[transport]") {
    auto *s = make(CATALYST_TRANSPORT_ROLE_CONTROLLER, "reg_ctrl");
    REQUIRE(s != nullptr);
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_CONTROLLER, "reg_ctrl") == s);
    // Unknown key and mismatched role both miss.
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_CONTROLLER, "reg_absent") ==
          nullptr);
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_COPROCESSOR, "reg_ctrl") ==
          nullptr);
    __catalyst__transport__destroy(s);
    // destroy unregisters.
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_CONTROLLER, "reg_ctrl") ==
          nullptr);
}

TEST_CASE("role disambiguates the same key", "[transport]") {
    auto *ct = make(CATALYST_TRANSPORT_ROLE_CONTROLLER, "dis_key");
    auto *co = make(CATALYST_TRANSPORT_ROLE_COPROCESSOR, "dis_key");
    REQUIRE(ct != nullptr);
    REQUIRE(co != nullptr);
    REQUIRE(ct != co);
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_CONTROLLER, "dis_key") == ct);
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_COPROCESSOR, "dis_key") == co);
    __catalyst__transport__destroy(ct);
    __catalyst__transport__destroy(co);
}

TEST_CASE("an empty key is not registered", "[transport]") {
    auto *s = make(CATALYST_TRANSPORT_ROLE_COPROCESSOR, "");
    REQUIRE(s != nullptr);
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_COPROCESSOR, "") == nullptr);
    __catalyst__transport__destroy(s);
}

TEST_CASE("re-create under the same key overwrites", "[transport]") {
    auto *s1 = make(CATALYST_TRANSPORT_ROLE_CONTROLLER, "ovr_key");
    auto *s2 = make(CATALYST_TRANSPORT_ROLE_CONTROLLER, "ovr_key");
    REQUIRE(s1 != s2);
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_CONTROLLER, "ovr_key") == s2);
    __catalyst__transport__destroy(s1);
    // s1 was already overwritten, so s2 remains resolvable until its own destroy.
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_CONTROLLER, "ovr_key") == s2);
    __catalyst__transport__destroy(s2);
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_CONTROLLER, "ovr_key") ==
          nullptr);
}

TEST_CASE("get_session on an unregistered role/key returns null", "[transport]") {
    CHECK(__catalyst__transport__get_session(CATALYST_TRANSPORT_ROLE_CONTROLLER, "never_created") ==
          nullptr);
}

TEST_CASE("set_coprocessor_fn: an empty symbol binds the built-in echo", "[transport]") {
    auto *s = make(CATALYST_TRANSPORT_ROLE_COPROCESSOR, "");
    REQUIRE(s != nullptr);
    CHECK(__catalyst__transport__set_coprocessor_fn(s, "") == CATALYST_TRANSPORT_OK);
    CHECK(__catalyst__transport__set_coprocessor_fn(s, nullptr) == CATALYST_TRANSPORT_OK);
    __catalyst__transport__destroy(s);
}

TEST_CASE("set_coprocessor_fn: an unresolved symbol is an error", "[transport]") {
    auto *s = make(CATALYST_TRANSPORT_ROLE_COPROCESSOR, "");
    REQUIRE(s != nullptr);
    CHECK(__catalyst__transport__set_coprocessor_fn(s, "catalyst_no_such_symbol_xyz") ==
          CATALYST_TRANSPORT_ERR);
    __catalyst__transport__destroy(s);
}

TEST_CASE("set_coprocessor_fn on a controller session is an error", "[transport]") {
    auto *s = make(CATALYST_TRANSPORT_ROLE_CONTROLLER, "");
    REQUIRE(s != nullptr);
    CHECK(__catalyst__transport__set_coprocessor_fn(s, "") == CATALYST_TRANSPORT_ERR);
    __catalyst__transport__destroy(s);
}

TEST_CASE("set_coprocessor_fn binds through the setter the backend implements", "[transport]") {
    auto *s = __catalyst__transport__create(STUB_BACKEND_PATH, "launch_once",
                                            CATALYST_TRANSPORT_ROLE_COPROCESSOR, "");
    REQUIRE(s != nullptr);
    CHECK(__catalyst__transport__set_coprocessor_fn(s, "") == CATALYST_TRANSPORT_OK);
    __catalyst__transport__destroy(s);
}

TEST_CASE("null session arguments are rejected without crashing", "[transport]") {
    CHECK(__catalyst__transport__connect(nullptr, "127.0.0.1", 0) == CATALYST_TRANSPORT_ERR);
    CHECK(__catalyst__transport__exchange_keys(nullptr) == CATALYST_TRANSPORT_ERR);
    CHECK(__catalyst__transport__establish_channel(nullptr, "rdma") == CATALYST_TRANSPORT_ERR);
    CHECK(__catalyst__transport__set_coprocessor_fn(nullptr, "") == CATALYST_TRANSPORT_ERR);
    CHECK(__catalyst__transport__set_message_sizes(nullptr, 0, 0, 0) == CATALYST_TRANSPORT_ERR);
    CHECK(__catalyst__transport__post(nullptr, 0) == CATALYST_TRANSPORT_ERR);
    std::uint8_t buf[4] = {};
    CHECK(__catalyst__transport__collect(nullptr, buf, sizeof(buf)) == CATALYST_TRANSPORT_ERR);
    CHECK(__catalyst__transport__request_slot(nullptr) == nullptr);
    CHECK(__catalyst__transport__last_rtt_ns(nullptr) == 0);
    // The void entry points must simply not crash on null.
    __catalyst__transport__start(nullptr);
    __catalyst__transport__stop(nullptr);
    __catalyst__transport__destroy(nullptr);
    SUCCEED();
}

TEST_CASE("commit_work_item rejects a reply larger than the provisioned region", "[transport]") {
    auto *s = make(CATALYST_TRANSPORT_ROLE_CONTROLLER, "");
    REQUIRE(s != nullptr);
    // exchange_keys provisions the local reply region (the stub reports a zero-size region).
    REQUIRE(__catalyst__transport__exchange_keys(s) == CATALYST_TRANSPORT_OK);
    CHECK(__catalyst__transport__set_message_sizes(s, 0, 0, 1) == CATALYST_TRANSPORT_ERR);
    CHECK(__catalyst__transport__set_message_sizes(s, 0, 0, 0) == CATALYST_TRANSPORT_OK);
    __catalyst__transport__destroy(s);
}

TEST_CASE("destroy drains outstanding async tokens without a prior barrier", "[transport]") {
    auto *s = make(CATALYST_TRANSPORT_ROLE_CONTROLLER, "");
    REQUIRE(s != nullptr);
    REQUIRE(__catalyst__transport__connect_async(s, "127.0.0.1", 0) != 0);
    REQUIRE(__catalyst__transport__exchange_keys_async(s) != 0);
    __catalyst__transport__destroy(s);
    SUCCEED();
}

TEST_CASE("create rejects a session key containing ';'", "[transport]") {
    // The key is spliced into the config as `pair=<key>`. A ';' in the key would silently split
    // the config into two entries instead of one; refuse rather than misparse.
    auto *s = __catalyst__transport__create(STUB_BACKEND_PATH, "cfg",
                                            CATALYST_TRANSPORT_ROLE_CONTROLLER, "bad;key");
    CHECK(s == nullptr);
}

TEST_CASE("create rejects a config that already sets the reserved 'pair' key", "[transport]") {
    // `pair=` is reserved for the compiler-emitted session key. If a caller sets it, ours and
    // theirs would coexist and the backend would silently pick one; refuse rather than shadow.
    auto *s = __catalyst__transport__create(STUB_BACKEND_PATH, "pair=caller_supplied",
                                            CATALYST_TRANSPORT_ROLE_CONTROLLER, "reserved_key");
    CHECK(s == nullptr);
}

TEST_CASE("memcpy backend plugins round-trip through the transport CAPI", "[transport]") {
    auto *ct = make_memcpy_controller("memcpy_roundtrip");
    auto *co = make_memcpy_coprocessor("memcpy_roundtrip");
    REQUIRE(ct != nullptr);
    REQUIRE(co != nullptr);

    REQUIRE(__catalyst__transport__connect(ct, "loopback", 19011) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__connect(co, "loopback", 19011) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__exchange_keys(ct) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__exchange_keys(co) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__establish_channel(ct, "memcpy") == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__establish_channel(co, "memcpy") == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__set_message_sizes(
                ct, 0, sizeof(std::uint64_t), sizeof(std::uint64_t)) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__set_coprocessor_fn(co, "") == CATALYST_TRANSPORT_OK);

    __catalyst__transport__start(ct);
    __catalyst__transport__start(co);

    const std::uint64_t request = 0x0123456789ABCDEFull;
    REQUIRE(__catalyst__transport__stage_payload(ct, &request, sizeof(request), 0) ==
            CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__post(ct, 0) == CATALYST_TRANSPORT_OK);

    std::uint64_t reply = 0;
    REQUIRE(__catalyst__transport__collect(ct, &reply, sizeof(reply)) == CATALYST_TRANSPORT_OK);
    CHECK(reply == request);

    __catalyst__transport__destroy(ct);
    __catalyst__transport__destroy(co);
}
