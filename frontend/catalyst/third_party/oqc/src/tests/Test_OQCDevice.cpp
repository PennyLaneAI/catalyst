// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <catch2/catch.hpp>
#include <pybind11/embed.h>

#include "OQCDevice.cpp"
#include "OQCRunner.hpp"
#include "RuntimeCAPI.h"

using namespace Catalyst::Runtime::Device;

TEST_CASE("Test the OQCDevice constructor", "[openqasm]")
{
    auto device = OQCDevice("{shots : 100}");
    CHECK(device.GetNumQubits() == 0);

    REQUIRE_THROWS_WITH(device.Measure(0), Catch::Contains("unsupported by device"));
}

TEST_CASE("Test qubits allocation OpenQasmDevice", "[openqasm]")
{
    std::unique_ptr<OQCDevice> device = std::make_unique<OQCDevice>("{shots : 100}");

    device->AllocateQubits(3);
    CHECK(device->GetNumQubits() == 3);
    CHECK(device->GetDeviceShots() == 100);
}

TEST_CASE("Test the bell pair circuit", "[openqasm]")
{
    std::unique_ptr<OQCDevice> device = std::make_unique<OQCDevice>("{shots : 100}");

    constexpr size_t n = 2;
    auto wires = device->AllocateQubits(n);

    device->NamedOperation("Hadamard", {}, {wires[0]}, false);
    device->NamedOperation("CNOT", {}, {wires[0], wires[1]}, false);

    std::string toqasm = "OPENQASM 2.0;\n"
                         "include \"qelib1.inc\";\n"
                         "qreg qubits[2];\n"
                         "creg cbits[2];\n"
                         "h qubits[0];\n"
                         "cx qubits[0], qubits[1];\n";

    CHECK(device->Circuit() == toqasm);

    device->ReleaseQubits(wires);
    auto wiresnew = device->AllocateQubits(4);
    device->NamedOperation("CNOT", {}, {wiresnew[2], wiresnew[3]}, false);
    device->NamedOperation("Hadamard", {}, {wiresnew[2]}, false);
    std::string toqasmempty = "OPENQASM 2.0;\n"
                              "include \"qelib1.inc\";\n"
                              "qreg qubits[4];\n"
                              "creg cbits[4];\n"
                              "cx qubits[2], qubits[3];\n"
                              "h qubits[2];\n";

    CHECK(device->Circuit() == toqasmempty);
}

TEST_CASE("Test counts", "[openqasm][counts]")
{
    // This test needs a python interpreter to execute the OQC python script
    // inside the `OQCDevice`'s `PartialCounts' method.
    if (!Py_IsInitialized()) {
        pybind11::initialize_interpreter();
    }

    std::unique_ptr<OQCDevice> device = std::make_unique<OQCDevice>("{shots : 100}");
    auto wires = device->AllocateQubits(2);

    device->NamedOperation("Hadamard", {}, {wires[0]}, false);

    std::vector<double> eigvals(4);
    std::vector<int64_t> counts(4);
    DataView<double, 1> eigvals_view(eigvals);
    DataView<int64_t, 1> counts_view(counts);

    REQUIRE_THROWS_WITH(device->PartialCounts(eigvals_view, counts_view, {wires[0], wires[1]}),
                        Catch::Contains("OQC credentials not found in environment variables"));
}
