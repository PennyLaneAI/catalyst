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

#include "openqasm/OpenQasmDevice.hpp"

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime::Device;

TEST_CASE("Test the OpenQasmDevice constructor", "[openqasm]")
{
    auto device = OpenQasmDevice();
    CHECK(device.GetNumQubits() == 0);

    REQUIRE_THROWS_WITH(
        device.Circuit(),
        Catch::Contains(
            "[Function:toOpenQasm] Error in Catalyst Runtime: Invalid number of quantum register"));
}

TEST_CASE("Test qubits allocation OpenQasmDevice", "[openqasm]")
{
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>();

    constexpr size_t n = 3;
    device->AllocateQubits(1);
    CHECK(device->GetNumQubits() == 1);

    auto wires = device->AllocateQubits(n - 1);
    CHECK(device->GetNumQubits() == n);
    CHECK(wires.size() == n);
    CHECK(wires[0] == 1);
    CHECK(wires[n - 1] == n);
}

TEST_CASE("Test the bell pair circuit", "[openqasm]")
{
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>();

    constexpr size_t n = 2;
    auto wires = device->AllocateQubits(n);

    device->NamedOperation("Hadamard", {}, {wires[0]}, false);
    device->NamedOperation("CNOT", {}, {wires[0], wires[1]}, false);

    device->Measure(wires[0]);
    device->Measure(wires[1]);

    std::string toqasm = "OPENQASM 3.0;\n"
                         "qubit[2] qubits;\n"
                         "bit[2] bits;\n"
                         "h qubits[0];\n"
                         "cnot qubits[0], qubits[1];\n"
                         "bits[0] = measure qubits[0];\n"
                         "bits[1] = measure qubits[1];\n";

    CHECK(device->Circuit() == toqasm);
    // device->ExecuteCircuit();
}

TEST_CASE("Test a simple circuit", "[openqasm]")
{
    std::unique_ptr<OpenQasmDevice> device = std::make_unique<OpenQasmDevice>();

    constexpr size_t n = 5;
    auto wires = device->AllocateQubits(n);

    device->NamedOperation("PauliX", {}, {wires[0]}, false);
    device->NamedOperation("PauliY", {}, {wires[1]}, false);
    device->NamedOperation("PauliZ", {}, {wires[2]}, false);
    device->NamedOperation("RX", {0.6}, {wires[4]}, false);
    device->NamedOperation("CNOT", {}, {wires[0], wires[3]}, false);
    device->NamedOperation("Toffoli", {}, {wires[0], wires[3], wires[4]}, false);

    device->Measure(wires[0]);
    device->Measure(wires[1]);
    device->Measure(wires[2]);
    device->Measure(wires[3]);
    device->Measure(wires[4]);

    std::string toqasm = "OPENQASM 3.0;\n"
                         "qubit[5] qubits;\n"
                         "bit[5] bits;\n"
                         "x qubits[0];\n"
                         "y qubits[1];\n"
                         "z qubits[2];\n"
                         "rx(0.6) qubits[4];\n"
                         "cnot qubits[0], qubits[3];\n"
                         "ccnot qubits[0], qubits[3], qubits[4];\n"
                         "bits[0] = measure qubits[0];\n"
                         "bits[1] = measure qubits[1];\n"
                         "bits[2] = measure qubits[2];\n"
                         "bits[3] = measure qubits[3];\n"
                         "bits[4] = measure qubits[4];\n";

    CHECK(device->Circuit() == toqasm);
    device->ExecuteCircuit();
}
