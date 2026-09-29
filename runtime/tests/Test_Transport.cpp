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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "catch2/catch_test_macros.hpp"

#include "TransportCAPI.h"

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

namespace {
// State observed by the scaling coprocessor function below.
std::string g_init_config;
void *g_fini_ctx = nullptr;
struct ScaleCtx {
    std::uint64_t factor;
};
} // namespace

// A per-message coprocessor function configured through `<symbol>_init`: it multiplies the
// payload by the `factor` key of its config. The executable exports these for dlsym.
extern "C" void *capi_test_scale_fn_init(const char *config) {
    g_init_config = config;
    const std::string_view cfg(config);
    const auto at = cfg.find("factor=");
    if (at == std::string_view::npos) {
        return nullptr;
    }
    return new ScaleCtx{std::stoull(std::string(cfg.substr(at + 7)))};
}

extern "C" void capi_test_scale_fn_fini(void *ctx) {
    g_fini_ctx = ctx;
    delete static_cast<ScaleCtx *>(ctx);
}

extern "C" std::size_t capi_test_scale_fn(const void *in, std::size_t, void *out,
                                          std::size_t out_cap, void *ctx) {
    std::uint64_t value = 0;
    std::memcpy(&value, in, sizeof(value));
    value *= static_cast<const ScaleCtx *>(ctx)->factor;
    std::memcpy(out, &value, std::min(out_cap, sizeof(value)));
    return std::min(out_cap, sizeof(value));
}

TEST_CASE("set_coprocessor_fn hands a function its fn. config through <symbol>_init",
          "[transport]") {
    g_init_config.clear();
    g_fini_ctx = nullptr;
    auto *ct = make_memcpy_controller("fn_config");
    auto *co = __catalyst__transport__create(MEMCPY_COPROCESSOR_BACKEND_PATH,
                                             "fn.factor=3;other=1;fn.note=x",
                                             CATALYST_TRANSPORT_ROLE_COPROCESSOR, "fn_config");
    REQUIRE(ct != nullptr);
    REQUIRE(co != nullptr);
    REQUIRE(__catalyst__transport__connect(ct, "loopback", 19012) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__connect(co, "loopback", 19012) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__exchange_keys(ct) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__exchange_keys(co) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__establish_channel(ct, "memcpy") == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__establish_channel(co, "memcpy") == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__set_message_sizes(
                ct, 0, sizeof(std::uint64_t), sizeof(std::uint64_t)) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__set_coprocessor_fn(co, "capi_test_scale_fn") ==
            CATALYST_TRANSPORT_OK);
    CHECK(g_init_config == "factor=3;note=x");

    __catalyst__transport__start(ct);
    __catalyst__transport__start(co);
    const std::uint64_t request = 14;
    REQUIRE(__catalyst__transport__stage_payload(ct, &request, sizeof(request), 0) ==
            CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__post(ct, 0) == CATALYST_TRANSPORT_OK);
    std::uint64_t reply = 0;
    REQUIRE(__catalyst__transport__collect(ct, &reply, sizeof(reply)) == CATALYST_TRANSPORT_OK);
    CHECK(reply == 42);

    __catalyst__transport__destroy(ct);
    CHECK(g_fini_ctx == nullptr);
    __catalyst__transport__destroy(co);
    CHECK(g_fini_ctx != nullptr);
}

TEST_CASE("set_coprocessor_fn fails when <symbol>_init fails", "[transport]") {
    auto *co = __catalyst__transport__create(MEMCPY_COPROCESSOR_BACKEND_PATH, "fn.unrelated=1",
                                             CATALYST_TRANSPORT_ROLE_COPROCESSOR, "fn_init_fails");
    REQUIRE(co != nullptr);
    CHECK(__catalyst__transport__set_coprocessor_fn(co, "capi_test_scale_fn") ==
          CATALYST_TRANSPORT_ERR);
    __catalyst__transport__destroy(co);
}

extern "C" void *catalyst_onnx_coprocessor_init(const char *config);

TEST_CASE("the ONNX coprocessor function rejects a bad config", "[transport]") {
    CHECK(catalyst_onnx_coprocessor_init("") == nullptr); // no model
    CHECK(catalyst_onnx_coprocessor_init("model=m.onnx;colour=blue") == nullptr);
    CHECK(catalyst_onnx_coprocessor_init("model=m.onnx;provider=tpu") == nullptr);
    CHECK(catalyst_onnx_coprocessor_init("model=m.onnx;ort_lib=/no/such/libort.so") == nullptr);
}

TEST_CASE("set_coprocessor_fn configures the ONNX coprocessor function through its _init",
          "[transport]") {
    auto *co = __catalyst__transport__create(MEMCPY_COPROCESSOR_BACKEND_PATH,
                                             "fn.model=/no/such/model.onnx",
                                             CATALYST_TRANSPORT_ROLE_COPROCESSOR, "onnx_fn");
    REQUIRE(co != nullptr);
    // The function and its _init resolve, and it is _init that fails, on the missing model.
    CHECK(__catalyst__transport__set_coprocessor_fn(co, "catalyst_onnx_coprocessor") ==
          CATALYST_TRANSPORT_ERR);
    __catalyst__transport__destroy(co);
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
