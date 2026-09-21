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

#include <filesystem>
#include <fstream>
#include <string>

#include "catch2/catch_test_macros.hpp"

#include "ExecutionContext.hpp"
#include "NullQubit.hpp"
#include "RecordingQuantumDevice.hpp"
#include "RuntimeCAPI.h"
#include "TestUtils.hpp"

#include <unistd.h>

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Devices;

namespace {

auto uniqueTempFile(const std::string &tag) -> std::string {
    auto path = std::filesystem::temp_directory_path() /
                ("catalyst_resources_" + tag + "_" + std::to_string(::getpid()) + ".json");
    std::filesystem::remove(path);
    return path.string();
}

auto readFile(const std::string &path) -> std::string {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

} // namespace

TEST_CASE("splitDeviceKwargs peels record_device_calls from factory kwargs",
          "[RecordingQuantumDevice]") {
    auto none = splitDeviceKwargs("{'track_resources': True}");
    CHECK(none.record_path.empty());
    CHECK(none.factory_kwargs == "{'track_resources': True}");

    auto only = splitDeviceKwargs("{'record_device_calls': '/tmp/resources.json'}");
    CHECK(only.record_path == "/tmp/resources.json");
    CHECK(only.factory_kwargs == "{}");

    auto mixed = splitDeviceKwargs(
        "{'track_resources': True, 'record_device_calls': '/tmp/resources.json'}");
    CHECK(mixed.record_path == "/tmp/resources.json");
    CHECK(mixed.factory_kwargs.find("record_device_calls") == std::string::npos);
    CHECK(mixed.factory_kwargs.find("track_resources") != std::string::npos);

    auto encoded =
        splitDeviceKwargs("{'record_device_calls': '/tmp/resources%20with%2Cpunctuation.json'}");
    CHECK(encoded.record_path == "/tmp/resources with,punctuation.json");
}

TEST_CASE("RecordingQuantumDevice reuses NullQubit resource tracking", "[RecordingQuantumDevice]") {
    auto path = uniqueTempFile("forward");
    {
        auto device =
            std::make_unique<RecordingQuantumDevice>(std::make_unique<NullQubit>(""), path);
        auto qubits = device->AllocateQubits(2);
        device->NamedOperation("Hadamard", {}, {qubits[0]}, false, {}, {}, {});
        device->NamedOperation("RX", {0.5}, {qubits[1]}, true, {qubits[0]}, {true}, {});
        auto observable_x = device->Observable(ObsId::PauliX, {}, {qubits[0]});
        auto observable_z = device->Observable(ObsId::PauliZ, {}, {qubits[0]});
        device->Expval(observable_x);
        device->Expval(observable_z);
        device->ReleaseQubits(qubits);
    }

    auto resources = readFile(path);
    CHECK(resources.find("\"num_wires\": 2") != std::string::npos);
    CHECK(resources.find("\"num_gates\": 2") != std::string::npos);
    CHECK(resources.find("\"Hadamard\": 1") != std::string::npos);
    CHECK(resources.find("\"C(Adjoint(RX))\": 1") != std::string::npos);
    CHECK(resources.find("\"expval(PauliX)\": 1") != std::string::npos);
    CHECK(resources.find("\"expval(PauliZ)\": 1") != std::string::npos);
}

TEST_CASE("RecordingQuantumDevice replaces an existing record file", "[RecordingQuantumDevice]") {
    auto path = uniqueTempFile("replace");
    {
        std::ofstream stale(path);
        stale << "stale contents";
    }

    // Destruction must not propagate an exception: doing so aborts the host process.
    {
        auto device =
            std::make_unique<RecordingQuantumDevice>(std::make_unique<NullQubit>(""), path);
        auto qubits = device->AllocateQubits(1);
        device->NamedOperation("Hadamard", {}, {qubits[0]}, false, {}, {}, {});
        device->ReleaseQubits(qubits);
    }

    auto resources = readFile(path);
    CHECK(resources.find("stale contents") == std::string::npos);
    CHECK(resources.find("\"Hadamard\": 1") != std::string::npos);
}

TEST_CASE("device kwargs select wrapped vs raw pooled devices", "[RecordingQuantumDevice]") {
    auto path = uniqueTempFile("pool");
    {
        ExecutionContext ctx;
        auto wrapped = ctx.getOrCreateDevice("null.qubit", "null_qubit",
                                             "{'record_device_calls': '" + path + "'}", false);
        REQUIRE(wrapped);
        wrapped->getQuantumDevicePtr()->AllocateQubits(1);
        ctx.deactivateDevice(wrapped.get());

        auto wrapped_again = ctx.getOrCreateDevice(
            "null.qubit", "null_qubit", "{'record_device_calls': '" + path + "'}", false);
        CHECK(wrapped_again.get() == wrapped.get());
        ctx.deactivateDevice(wrapped_again.get());

        auto raw = ctx.getOrCreateDevice(std::string{"null.qubit"}, std::string{"null_qubit"},
                                         std::string{}, false);
        CHECK(raw.get() != wrapped.get());
    }

    auto resources = readFile(path);
    CHECK(resources.find("\"total_allocations\": 1") != std::string::npos);
}

TEST_CASE("CAPI device_init records NullQubit resources for a wrapped device",
          "[RecordingQuantumDevice]") {
    auto path = uniqueTempFile("capi");
    __catalyst__rt__initialize(nullptr);

    std::string rtd_lib = "null.qubit";
    std::string rtd_name = "null_qubit";
    std::string rtd_kwargs = "{'record_device_calls': '" + path + "'}";
    __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                (int8_t *)rtd_kwargs.c_str(), 0, false);

    QUBIT *q0 = __catalyst__rt__qubit_allocate();
    __catalyst__qis__Hadamard(q0, NO_MODIFIERS);
    __catalyst__rt__qubit_release(q0);
    __catalyst__rt__device_release();
    __catalyst__rt__finalize();

    auto resources = readFile(path);
    CHECK(resources.find("\"num_gates\": 1") != std::string::npos);
    CHECK(resources.find("\"Hadamard\": 1") != std::string::npos);
}
