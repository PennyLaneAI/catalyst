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

// Unit tests for the ONNX coprocessor function, catalyst_onnx_coprocessor

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "catch2/catch_test_macros.hpp"

#include "Transport.hpp"
#include "TransportCAPI.h"
#include "TransportTestUtils.hpp"

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

TEST_CASE("an ONNX coprocessor function outlives the other contexts of its onnxruntime",
          "[transport]") {
    const char *ort_lib = std::getenv("CATALYST_TEST_ONNXRUNTIME_LIB");
    if (!ort_lib || !*ort_lib) {
        SKIP("set CATALYST_TEST_ONNXRUNTIME_LIB to an onnxruntime shared library");
    }
    CerrCapture cerr;
    const std::string config = kModel + ";provider=cpu;ort_lib=" + std::string(ort_lib);
    std::uint8_t frame[16] = {1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 1, 0, 0, 0};
    std::uint8_t reply[8] = {};

    void *first = onnx_info().init(config.c_str());
    void *second = onnx_info().init(config.c_str());
    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    onnx_info().fini(first);
    CHECK(catalyst_onnx_coprocessor(frame, sizeof(frame), reply, sizeof(reply), second) == 8);
    onnx_info().fini(second);

    // A context created after every earlier one has finished starts and runs as the first did.
    void *third = onnx_info().init(config.c_str());
    REQUIRE(third != nullptr);
    std::memset(reply, 0, sizeof(reply));
    CHECK(catalyst_onnx_coprocessor(frame, sizeof(frame), reply, sizeof(reply), third) == 8);
    CHECK(std::memcmp(reply, frame, 8) == 0);
    onnx_info().fini(third);
}

TEST_CASE("the ONNX coprocessor function requires the message sizes to match its model",
          "[transport]") {
    const char *ort_lib = std::getenv("CATALYST_TEST_ONNXRUNTIME_LIB");
    if (!ort_lib || !*ort_lib) {
        SKIP("set CATALYST_TEST_ONNXRUNTIME_LIB to an onnxruntime shared library");
    }
    // The test model is uint8[1, 8] in and uint8[1, 8] out.
    const std::string base = kModel + ";provider=cpu;ort_lib=" + std::string(ort_lib);
    CHECK(onnx_init_error(base + ";in_bytes=8;out_bytes=8").empty());
    CHECK(onnx_init_error(base + ";in_bytes=9;out_bytes=8")
              .find("the model input is 8 B, but the controller sends 9 B (in_bytes)") !=
          std::string::npos);
    CHECK(onnx_init_error(base + ";in_bytes=8;out_bytes=16")
              .find("the model output is 8 B, but the controller expects 16 B (out_bytes)") !=
          std::string::npos);
}

TEST_CASE("the ONNX coprocessor function rejects a model whose output is not a tensor",
          "[transport]") {
    const char *ort_lib = std::getenv("CATALYST_TEST_ONNXRUNTIME_LIB");
    if (!ort_lib || !*ort_lib) {
        SKIP("set CATALYST_TEST_ONNXRUNTIME_LIB to an onnxruntime shared library");
    }
    // The test model takes uint8[1, 8] and returns a sequence holding it.
    const std::string config =
        "model=" CATALYST_TEST_ONNX_SEQUENCE_MODEL ";provider=cpu;ort_lib=" + std::string(ort_lib) +
        ";in_bytes=8;out_bytes=8";
    CHECK(onnx_init_error(config).find("the model's output must be a tensor") != std::string::npos);
}
