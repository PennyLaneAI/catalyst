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

#include "OQCDevice.cpp"
#include "OQCRunner.hpp"
#include "RuntimeCAPI.h"

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime::Device;

TEST_CASE("Test OpenQasmRunner base class", "[openqasm]")
{
    // check the coverage support
    OQCRunnerBase runner{};
    REQUIRE_THROWS_WITH(runner.runCircuit("", "", 0),
                        Catch::Contains("[Function:runCircuit] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Probs("", "", 0, 0),
                        Catch::Contains("[Function:Probs] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Sample("", "", 0, 0),
                        Catch::Contains("[Function:Sample] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Expval("", "", 0),
                        Catch::Contains("[Function:Expval] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Var("", "", 0),
                        Catch::Contains("[Function:Var] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.State("", "", 0, 0),
                        Catch::Contains("[Function:State] Error in Catalyst Runtime: "
                                        "Not implemented method"));

    REQUIRE_THROWS_WITH(runner.Gradient("", "", 0, 0),
                        Catch::Contains("[Function:Gradient] Error in Catalyst Runtime: "
                                        "Not implemented method"));
}

TEST_CASE("Test the OQCDevice constructor", "[openqasm]")
{
    auto device = OQCDevice("{shots : 100}");
    CHECK(device.GetNumQubits() == 0);

    REQUIRE_THROWS_WITH(device.PrintState(), Catch::Contains("Unsupported functionality"));
    REQUIRE_THROWS_WITH(device.AllocateQubit(), Catch::Contains("Unsupported functionality"));
    REQUIRE_THROWS_WITH(device.Measure(0), Catch::Contains("Unsupported functionality"));
    REQUIRE_THROWS_WITH(device.Expval(0), Catch::Contains("Unsupported functionality"));
    REQUIRE_THROWS_WITH(device.Var(0), Catch::Contains("Unsupported functionality"));
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

    device->ReleaseAllQubits();
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
