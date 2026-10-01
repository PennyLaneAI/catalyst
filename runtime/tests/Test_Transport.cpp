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
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include "catch2/catch_test_macros.hpp"

#include "Transport.hpp"
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
void *g_init_ctx = nullptr; // the ctx the latest init returned
void *g_fini_ctx = nullptr; // the ctx the latest fini released
struct ScaleCtx {
    std::uint64_t factor;
};
} // namespace

// A per-message coprocessor function configured through its CoprocessorFnInfo: it multiplies the
// payload by the `factor` key of its config. The executable exports the function and its info for
// dlsym.
static void *scale_fn_init(const char *config) {
    g_init_config = config;
    const std::string_view cfg(config);
    const auto at = cfg.find("factor=");
    if (at == std::string_view::npos) {
        return nullptr;
    }
    g_init_ctx = new ScaleCtx{std::stoull(std::string(cfg.substr(at + 7)))};
    return g_init_ctx;
}

static void scale_fn_fini(void *ctx) {
    g_fini_ctx = ctx;
    delete static_cast<ScaleCtx *>(ctx);
}

extern "C" const catalyst::transport::CoprocessorFnInfo *capi_test_scale_fn_info() {
    static constexpr catalyst::transport::CoprocessorFnInfo info{
        .abi_version = catalyst::transport::COPROCESSOR_FN_ABI_VERSION,
        .init = &scale_fn_init,
        .fini = &scale_fn_fini,
    };
    return &info;
}

// The scaling function again, with an info from an ABI version newer than this runtime knows.
extern "C" std::size_t capi_test_scale_fn(const void *in, std::size_t in_len, void *out,
                                          std::size_t out_cap, void *ctx);
extern "C" std::size_t capi_test_future_fn(const void *in, std::size_t in_len, void *out,
                                           std::size_t out_cap, void *ctx) {
    return capi_test_scale_fn(in, in_len, out, out_cap, ctx);
}

extern "C" const catalyst::transport::CoprocessorFnInfo *capi_test_future_fn_info() {
    static constexpr catalyst::transport::CoprocessorFnInfo info{
        .abi_version = catalyst::transport::COPROCESSOR_FN_ABI_VERSION + 1,
        .init = &scale_fn_init,
        .fini = &scale_fn_fini,
    };
    return &info;
}

// A function whose info has no version.
extern "C" std::size_t capi_test_unversioned_fn(const void *, std::size_t, void *, std::size_t,
                                                void *) {
    return 0;
}

extern "C" const catalyst::transport::CoprocessorFnInfo *capi_test_unversioned_fn_info() {
    static constexpr catalyst::transport::CoprocessorFnInfo info{
        .abi_version = 0,
        .init = nullptr,
        .fini = nullptr,
    };
    return &info;
}

// An info for the Steane decoder exported by this executable, not by the Steane library that
// defines the function, so the runtime must not use it.
static bool g_foreign_info_used = false;
static void *foreign_info_init(const char *) {
    g_foreign_info_used = true;
    return nullptr;
}
extern "C" const catalyst::transport::CoprocessorFnInfo *steane_coprocessor_info() {
    static constexpr catalyst::transport::CoprocessorFnInfo info{
        .abi_version = 1,
        .init = &foreign_info_init,
        .fini = nullptr,
    };
    return &info;
}

extern "C" std::size_t capi_test_scale_fn(const void *in, std::size_t, void *out,
                                          std::size_t out_cap, void *ctx) {
    std::uint64_t value = 0;
    std::memcpy(&value, in, sizeof(value));
    value *= static_cast<const ScaleCtx *>(ctx)->factor;
    std::memcpy(out, &value, std::min(out_cap, sizeof(value)));
    return std::min(out_cap, sizeof(value));
}

TEST_CASE("set_coprocessor_fn hands a function its fn. config through its info init hook",
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

namespace {
// Connect, exchange keys and establish a memcpy controller and coprocessor on `port`, and commit
// 8 B messages each way.
void establish_memcpy_pair(CatalystTransportSession *ct, CatalystTransportSession *co,
                           std::uint16_t port) {
    REQUIRE(__catalyst__transport__connect(ct, "loopback", port) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__connect(co, "loopback", port) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__exchange_keys(ct) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__exchange_keys(co) == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__establish_channel(ct, "memcpy") == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__establish_channel(co, "memcpy") == CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__set_message_sizes(
                ct, 0, sizeof(std::uint64_t), sizeof(std::uint64_t)) == CATALYST_TRANSPORT_OK);
}

// Redirects std::cerr into a string for the lifetime of the object.
class CerrCapture {
  public:
    CerrCapture() : old_(std::cerr.rdbuf(buf_.rdbuf())) {}
    ~CerrCapture() { std::cerr.rdbuf(old_); }
    CerrCapture(const CerrCapture &) = delete;
    CerrCapture &operator=(const CerrCapture &) = delete;
    std::string str() const { return buf_.str(); }

  private:
    std::ostringstream buf_;
    std::streambuf *old_;
};
} // namespace

TEST_CASE("set_coprocessor_fn fails, and every message then fails, when init fails",
          "[transport]") {
    auto *ct = make_memcpy_controller("fn_init_fails");
    auto *co = __catalyst__transport__create(MEMCPY_COPROCESSOR_BACKEND_PATH, "fn.unrelated=1",
                                             CATALYST_TRANSPORT_ROLE_COPROCESSOR, "fn_init_fails");
    REQUIRE(ct != nullptr);
    REQUIRE(co != nullptr);
    establish_memcpy_pair(ct, co, 19033);
    CHECK(__catalyst__transport__set_coprocessor_fn(co, "capi_test_scale_fn") ==
          CATALYST_TRANSPORT_ERR);

    // The session does not fall back to echo: the message fails.
    __catalyst__transport__start(ct);
    __catalyst__transport__start(co);
    const std::uint64_t request = 14;
    REQUIRE(__catalyst__transport__stage_payload(ct, &request, sizeof(request), 0) ==
            CATALYST_TRANSPORT_OK);
    CHECK(__catalyst__transport__post(ct, 0) == CATALYST_TRANSPORT_ERR);

    __catalyst__transport__destroy(ct);
    __catalyst__transport__destroy(co);
}

TEST_CASE("a rejected set_coprocessor_fn keeps the bound function and its ctx", "[transport]") {
    g_fini_ctx = nullptr;
    auto *ct = make_memcpy_controller("fn_rebind");
    auto *co = __catalyst__transport__create(MEMCPY_COPROCESSOR_BACKEND_PATH, "fn.factor=3",
                                             CATALYST_TRANSPORT_ROLE_COPROCESSOR, "fn_rebind");
    REQUIRE(ct != nullptr);
    REQUIRE(co != nullptr);
    establish_memcpy_pair(ct, co, 19034);
    REQUIRE(__catalyst__transport__set_coprocessor_fn(co, "capi_test_scale_fn") ==
            CATALYST_TRANSPORT_OK);
    void *bound_ctx = g_init_ctx;
    __catalyst__transport__start(ct);
    __catalyst__transport__start(co);

    // A function cannot be rebound once the session has started. The rejected call releases only
    // the ctx it created, never the one the running function uses.
    CHECK(__catalyst__transport__set_coprocessor_fn(co, "capi_test_scale_fn") ==
          CATALYST_TRANSPORT_ERR);
    CHECK(g_fini_ctx != nullptr);
    CHECK(g_fini_ctx != bound_ctx);

    const std::uint64_t request = 14;
    REQUIRE(__catalyst__transport__stage_payload(ct, &request, sizeof(request), 0) ==
            CATALYST_TRANSPORT_OK);
    REQUIRE(__catalyst__transport__post(ct, 0) == CATALYST_TRANSPORT_OK);
    std::uint64_t reply = 0;
    REQUIRE(__catalyst__transport__collect(ct, &reply, sizeof(reply)) == CATALYST_TRANSPORT_OK);
    CHECK(reply == 42);

    __catalyst__transport__destroy(ct);
    __catalyst__transport__destroy(co);
    CHECK(g_fini_ctx == bound_ctx); // released with the session
}

TEST_CASE("set_coprocessor_fn reads the known fields of an info from a newer ABI version",
          "[transport]") {
    g_init_config.clear();
    g_fini_ctx = nullptr;
    auto *co = __catalyst__transport__create(MEMCPY_COPROCESSOR_BACKEND_PATH, "fn.factor=2",
                                             CATALYST_TRANSPORT_ROLE_COPROCESSOR, "fn_future_abi");
    REQUIRE(co != nullptr);
    CHECK(__catalyst__transport__set_coprocessor_fn(co, "capi_test_future_fn") ==
          CATALYST_TRANSPORT_OK);
    CHECK(g_init_config == "factor=2");
    void *bound_ctx = g_init_ctx;
    __catalyst__transport__destroy(co);
    CHECK(g_fini_ctx == bound_ctx);
}

TEST_CASE("set_coprocessor_fn rejects an info without an ABI version", "[transport]") {
    auto *co = make_memcpy_coprocessor("fn_unversioned");
    REQUIRE(co != nullptr);
    CerrCapture cerr;
    CHECK(__catalyst__transport__set_coprocessor_fn(co, "capi_test_unversioned_fn") ==
          CATALYST_TRANSPORT_ERR);
    CHECK(cerr.str().find("has no ABI version") != std::string::npos);
    __catalyst__transport__destroy(co);
}

TEST_CASE("set_coprocessor_fn uses only the info of the library defining the function",
          "[transport]") {
    // The Steane decoder's library exports no info, but this executable exports one under the
    // same name, earlier in the global scope.
    void *steane = dlopen(STEANE_CPU_LIB_PATH, RTLD_NOW | RTLD_GLOBAL);
    REQUIRE(steane != nullptr);
    g_foreign_info_used = false;
    auto *co = make_memcpy_coprocessor("fn_foreign_info");
    REQUIRE(co != nullptr);
    CHECK(__catalyst__transport__set_coprocessor_fn(co, "steane_coprocessor") ==
          CATALYST_TRANSPORT_OK);
    CHECK_FALSE(g_foreign_info_used);
    __catalyst__transport__destroy(co);
    dlclose(steane);
}

#ifdef CATALYST_TEST_ONNX_MODEL
extern "C" const catalyst::transport::CoprocessorFnInfo *catalyst_onnx_coprocessor_info();
extern "C" std::size_t catalyst_onnx_coprocessor(const void *in, std::size_t in_len, void *out,
                                                 std::size_t out_cap, void *ctx);

namespace {
const catalyst::transport::CoprocessorFnInfo &onnx_info() {
    return *catalyst_onnx_coprocessor_info();
}

// The reason the ONNX function's init gives for rejecting `config`, or "" if it accepts it.
std::string onnx_init_error(const std::string &config) {
    CerrCapture cerr;
    void *ctx = onnx_info().init(config.c_str());
    if (ctx) {
        onnx_info().fini(ctx);
        return "";
    }
    return cerr.str();
}

const std::string kModel = "model=" CATALYST_TEST_ONNX_MODEL;
} // namespace

TEST_CASE("the ONNX coprocessor function rejects a bad config before loading onnxruntime",
          "[transport]") {
    // ort_lib names a missing library, so a config check that passed would fail at dlopen instead.
    const std::string no_ort = ";ort_lib=/no/such/libonnxruntime.so";
    CHECK(onnx_init_error(no_ort).find("config needs model=") != std::string::npos);
    CHECK(onnx_init_error(kModel + ";colour=blue" + no_ort).find("unknown config key 'colour'") !=
          std::string::npos);
    CHECK(onnx_init_error(kModel + ";provider=tpu" + no_ort).find("unknown provider 'tpu'") !=
          std::string::npos);
    CHECK(onnx_init_error(kModel + ";device=-1" + no_ort).find("'device' must be a non-negative") !=
          std::string::npos);
    CHECK(onnx_init_error(kModel + ";threads=0" + no_ort).find("threads must be at least 1") !=
          std::string::npos);
    CHECK(onnx_init_error("model=/no/such/model.onnx" + no_ort).find("no model file at") !=
          std::string::npos);
    CHECK(onnx_init_error(kModel + no_ort).find("cannot load onnxruntime") != std::string::npos);
}

TEST_CASE("set_coprocessor_fn configures the ONNX coprocessor function through its info",
          "[transport]") {
    auto *co = __catalyst__transport__create(MEMCPY_COPROCESSOR_BACKEND_PATH,
                                             "fn.model=/no/such/model.onnx",
                                             CATALYST_TRANSPORT_ROLE_COPROCESSOR, "onnx_fn");
    REQUIRE(co != nullptr);
    CerrCapture cerr;
    CHECK(__catalyst__transport__set_coprocessor_fn(co, "catalyst_onnx_coprocessor") ==
          CATALYST_TRANSPORT_ERR);
    // The function and its info resolved, and init rejected the missing model.
    CHECK(cerr.str().find("no model file at '/no/such/model.onnx'") != std::string::npos);
    __catalyst__transport__destroy(co);
}

TEST_CASE("the ONNX coprocessor function runs a model", "[transport]") {
    // Needs an onnxruntime shared library, for example the one in the onnxruntime pip package.
    const char *ort_lib = std::getenv("CATALYST_TEST_ONNXRUNTIME_LIB");
    if (!ort_lib || !*ort_lib) {
        SKIP("set CATALYST_TEST_ONNXRUNTIME_LIB to an onnxruntime shared library");
    }
    CerrCapture cerr;
    CHECK(onnx_info().abi_version == catalyst::transport::COPROCESSOR_FN_ABI_VERSION);
    void *ctx =
        onnx_info().init((kModel + ";provider=cpu;ort_lib=" + std::string(ort_lib)).c_str());
    REQUIRE(ctx != nullptr);
    CHECK(cerr.str().find("on the CPU") != std::string::npos);

    // The test model is the identity on uint8[1, 8]. A frame is the payload, then decoder_id and
    // seq_num.
    std::uint8_t frame[16] = {1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 1, 0, 0, 0};
    std::uint8_t reply[8] = {};
    CHECK(catalyst_onnx_coprocessor(frame, sizeof(frame), reply, sizeof(reply), ctx) == 8);
    CHECK(std::memcmp(reply, frame, 8) == 0);

    // A payload shorter than the model input is an error, not a zero reply.
    CHECK(catalyst_onnx_coprocessor(frame, 12, reply, sizeof(reply), ctx) ==
          catalyst::transport::COPROCESSOR_FN_ERROR);
    onnx_info().fini(ctx);
}
#endif

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
