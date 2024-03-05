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

    std::string case1 = "qubit[10] qubits;\n";
    CHECK(reg.toOpenQASM2(RegisterMode::Alloc) == case1);

    std::string case2 = "reset qubits;\n";
    CHECK(reg.toOpenQASM2(RegisterMode::Reset) == case2);

    REQUIRE_THROWS_WITH(reg.toOpenQASM2(static_cast<RegisterMode>(4)),
                        Catch::Contains("[Function:toOpenQASM2] Error in Catalyst Runtime: "
                                        "Unsupported OpenQasm register mode"));
}