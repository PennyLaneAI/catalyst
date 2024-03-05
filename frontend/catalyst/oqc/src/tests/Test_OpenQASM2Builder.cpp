// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <typeinfo>

#include "OpenQASM2Builder.hpp"

#include <catch2/catch.hpp>

#define TYPE_INFO(x) std::string(typeid(x).name())

using namespace Catalyst::Runtime::OpenQasm2;

TEST_CASE("Test lookup openqasm gate names from QIR -> OpenQasm map", "[openqasm]")
{
    // Check lookup supported gates
    CHECK(lookup_qasm_gate_name("PauliX") == "x");
    CHECK(lookup_qasm_gate_name("Hadamard") == "h");
    CHECK(lookup_qasm_gate_name("CNOT") == "cx");
    CHECK(lookup_qasm_gate_name("SWAP") == "swap");
    CHECK(lookup_qasm_gate_name("Toffoli") == "ccx");

    // Check lookup an unsupported gate
    REQUIRE_THROWS_WITH(
        lookup_qasm_gate_name("ABC"),
        Catch::Contains("The given QIR gate name is not supported by the OpenQASM builder"));
}

TEST_CASE("Test QasmRegister(type=Qubit) from OpenQasmBuilder", "[openqasm]")
{
    auto reg = QASMRegister(RegisterType::Qubit, "qubits", 5);
    CHECK(reg.getName() == "qubits");
    CHECK(reg.getSize() == 5);

    reg.updateSize(10);
    CHECK(reg.getSize() == 10);

    reg.resetSize();
    CHECK(reg.getSize() == 0);

    reg.updateSize(10);

    std::string case1 = "qreg qubits[10];\n";
    CHECK(reg.toOpenQASM2(RegisterMode::Alloc) == case1);

    std::string case2 = "reset qubits;\n";
    CHECK(reg.toOpenQASM2(RegisterMode::Reset) == case2);

    REQUIRE_THROWS_WITH(reg.toOpenQASM2(static_cast<RegisterMode>(4)),
                        Catch::Contains("[Function:toOpenQASM2] Error in Catalyst Runtime: "
                                        "Unsupported OpenQasm register mode"));
}

TEST_CASE("Test QasmRegister(type=Bit) from OpenQasmBuilder", "[openqasm]")
{
    auto reg = QASMRegister(RegisterType::Bit, "bits", 5);
    CHECK(reg.getName() == "bits");
    CHECK(reg.getSize() == 5);

    reg.updateSize(10);
    CHECK(reg.getSize() == 10);

    reg.resetSize();
    CHECK(reg.getSize() == 0);

    reg.updateSize(10);

    std::string case1 = "creg bits[10];\n";
    CHECK(reg.toOpenQASM2(RegisterMode::Alloc) == case1);

    std::string case2 = "reset bits;\n";
    CHECK(reg.toOpenQASM2(RegisterMode::Reset) == case2);

    // Check edge cases
    auto reg_buggy = QASMRegister(static_cast<RegisterType>(3), "random", 5);
    REQUIRE_THROWS_WITH(reg_buggy.toOpenQASM2(RegisterMode::Alloc),
                        Catch::Contains("[Function:toOpenQASM2] Error in Catalyst Runtime: "
                                        "Unsupported OpenQasm register type"));

    REQUIRE_THROWS_WITH(reg.toOpenQASM2(static_cast<RegisterMode>(4)),
                        Catch::Contains("[Function:toOpenQASM2] Error in Catalyst Runtime: "
                                        "Unsupported OpenQasm register mode"));
}


TEST_CASE("Test QasmGate from OpenQasmBuilder", "[openqasm]")
{
    auto qubits = QASMRegister(RegisterType::Qubit, "q", 5);

    // Check a supported gate without params
    auto gate1 = QASMGate("PauliX", {}, {0});
    CHECK(gate1.getName() == "x");
    CHECK(gate1.getParams().empty());
    CHECK(gate1.getWires().size() == 1);
    CHECK(gate1.getWires()[0] == 0);

    std::string gate1_toqasm = "x q[0];\n";
    CHECK(gate1.toOpenQASM2(qubits) == gate1_toqasm);

    // Check a ctrl gate without params
    auto gate2 = QASMGate("SWAP", {}, {0, 2});

    std::string gate2_toqasm = "swap q[0], q[2];\n";
    CHECK(gate2.toOpenQASM2(qubits) == gate2_toqasm);

    // Check a gate with params (value)
    auto gate3 = QASMGate("RX", {0.361731}, {2});

    std::string gate3_toqasm = "rx(0.36) q[2];\n";
    CHECK(gate3.toOpenQASM2(qubits, 2) == gate3_toqasm);

    std::string gate3_toqasm_2 = "rx(0.3617) q[2];\n";
    CHECK(gate3.toOpenQASM2(qubits, 4) == gate3_toqasm_2);

    // Check a random gate with several params (value)
    // not a valid gate! This is just for testing...
    auto gate31 = QASMGate("RX", {0.123, 0.456}, {2});

    std::string gate31_toqasm = "rx(0.123, 0.456) q[2];\n";
    CHECK(gate31.toOpenQASM2(qubits, 3) == gate31_toqasm);

    // Check a CNOT gate
    auto gate41 = QASMGate("CNOT", {}, {0, 2});

    std::string gate41_toqasm = "cx q[0], q[2];\n";
    CHECK(gate41.toOpenQASM2(qubits) == gate41_toqasm);

    // Check edge cases
    REQUIRE_THROWS_WITH(
        QASMGate("ABC", {}, {}),
        Catch::Contains("[Function:lookup_qasm_gate_name] Error in Catalyst Runtime: The given QIR "
                        "gate name is not supported by the OpenQASM builder"));
}