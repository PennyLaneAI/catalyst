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

using namespace Catalyst::Runtime::OpenQASM2;

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

TEST_CASE("Test QasmMeasure from OpenQasmBuilder", "[openqasm]")
{
    auto qubits = QASMRegister(RegisterType::Qubit, "q", 5);
    auto cbits = QASMRegister(RegisterType::Bit, "c", 5);

    auto mz1 = QASMMeasure(0, 0);
    CHECK(mz1.getQubit() == 0);
    CHECK(mz1.getBit() == 0);

    std::string mz1_toqasm = "measure q[0] -> c[0];\n";
    CHECK(mz1.toOpenQASM2(qubits, cbits) == mz1_toqasm);

    auto mz2 = QASMMeasure(1, 1);
    CHECK(mz2.getQubit() == 1);
    CHECK(mz2.getBit() == 1);

    std::string mz2_toqasm = "measure q[1] -> c[1];\n";
    CHECK(mz2.toOpenQASM2(qubits, cbits) == mz2_toqasm);

    auto bits = QASMRegister(RegisterType::Bit, "bit", 2);

    std::string mz1_res_toqasm = "measure q[0] -> c[0];\n";
    CHECK(mz1.toOpenQASM2(qubits, cbits) == mz1_res_toqasm);

    std::string mz2_res_toqasm = "measure q[1] -> c[1];\n";
    CHECK(mz2.toOpenQASM2(qubits, cbits) == mz2_res_toqasm);
}

TEST_CASE("Test OpenQasmBuilder with dumping the circuit header, gates, and measure", "[openqasm]")
{
    auto builder = OpenQASM2Builder();

    builder.AddRegisters("q", 5, "c", 5);

    builder.AddGate("PauliX", {}, {0});
    builder.AddGate("Hadamard", {}, {1});
    builder.AddGate("SWAP", {}, {0, 1});
    builder.AddGate("RZ", {0.12}, {1});
    builder.AddGate("RX", {}, {0});

    builder.AddMeasurement(0, 0);
    builder.AddMeasurement(1, 1);

    auto toqasm = "OPENQASM 2.0;\n"
                  "include \"qelib1.inc\";\n"
                  "qreg q[5];\n"
                  "creg c[5];\n"
                  "x q[0];\n"
                  "h q[1];\n"
                  "swap q[0], q[1];\n"
                  "rz(0.12) q[1];\n"
                  "rx q[0];\n"
                  "measure q[0] -> c[0];\n"
                  "measure q[1] -> c[1];\n";

    CHECK(builder.toOpenQASM2() == toqasm);
}

TEST_CASE("Test OpenQasmBuilder with dumping the circuit header, gates, and measure all",
          "[openqasm]")
{
    auto builder = OpenQASM2Builder();

    builder.AddRegisters("q", 5, "c", 5);

    builder.AddGate("PauliX", {}, {0});
    builder.AddGate("Hadamard", {}, {1});
    builder.AddGate("SWAP", {}, {0, 1});
    builder.AddGate("RZ", {0.12}, {1});
    builder.AddGate("RX", {}, {0});

    builder.AddMeasurements();

    auto toqasm = "OPENQASM 2.0;\n"
                  "include \"qelib1.inc\";\n"
                  "qreg q[5];\n"
                  "creg c[5];\n"
                  "x q[0];\n"
                  "h q[1];\n"
                  "swap q[0], q[1];\n"
                  "rz(0.12) q[1];\n"
                  "rx q[0];\n"
                  "measure q -> c;\n";

    CHECK(builder.toOpenQASM2() == toqasm);
}