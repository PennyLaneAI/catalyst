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

// Unit tests for the transport CAPI session registry

#include <cstdint>

#include "catch2/catch_test_macros.hpp"

#include "TransportCAPI.h"

namespace {
constexpr std::int32_t kController = CATALYST_TRANSPORT_ROLE_CONTROLLER;
constexpr std::int32_t kCoprocessor = CATALYST_TRANSPORT_ROLE_COPROCESSOR;
constexpr const char *kStub = STUB_BACKEND_PATH;

CatalystTransportSession *make(std::int32_t role, const char *key)
{
    return __catalyst__transport__create(kStub, "cfg", role, key);
}
} // namespace

TEST_CASE("create registers a session resolvable by (role, key)", "[transport]")
{
    auto *s = make(kController, "reg_ctrl");
    REQUIRE(s != nullptr);
    CHECK(__catalyst__transport__get_session(kController, "reg_ctrl") == s);
    // Unknown key and mismatched role both miss.
    CHECK(__catalyst__transport__get_session(kController, "reg_absent") == nullptr);
    CHECK(__catalyst__transport__get_session(kCoprocessor, "reg_ctrl") == nullptr);
    __catalyst__transport__destroy(s);
    // destroy unregisters.
    CHECK(__catalyst__transport__get_session(kController, "reg_ctrl") == nullptr);
}

TEST_CASE("role disambiguates the same key", "[transport]")
{
    auto *ct = make(kController, "dis_key");
    auto *co = make(kCoprocessor, "dis_key");
    REQUIRE(ct != nullptr);
    REQUIRE(co != nullptr);
    REQUIRE(ct != co);
    CHECK(__catalyst__transport__get_session(kController, "dis_key") == ct);
    CHECK(__catalyst__transport__get_session(kCoprocessor, "dis_key") == co);
    __catalyst__transport__destroy(ct);
    __catalyst__transport__destroy(co);
}

TEST_CASE("an empty key is not registered", "[transport]")
{
    auto *s = make(kCoprocessor, "");
    REQUIRE(s != nullptr);
    CHECK(__catalyst__transport__get_session(kCoprocessor, "") == nullptr);
    __catalyst__transport__destroy(s);
}

TEST_CASE("re-create under the same key overwrites", "[transport]")
{
    auto *s1 = make(kController, "ovr_key");
    auto *s2 = make(kController, "ovr_key");
    REQUIRE(s1 != s2);
    CHECK(__catalyst__transport__get_session(kController, "ovr_key") == s2);
    __catalyst__transport__destroy(s1);
    // s1 was already overwritten, so s2 remains resolvable until its own destroy.
    CHECK(__catalyst__transport__get_session(kController, "ovr_key") == s2);
    __catalyst__transport__destroy(s2);
    CHECK(__catalyst__transport__get_session(kController, "ovr_key") == nullptr);
}

TEST_CASE("get_session on an unregistered role/key returns null", "[transport]")
{
    CHECK(__catalyst__transport__get_session(kController, "never_created") == nullptr);
}
