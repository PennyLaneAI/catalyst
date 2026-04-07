// Copyright 2023-2025 Xanadu Quantum Technologies Inc.

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

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "OpenQasmBuilder.hpp"

#define TYPE_INFO(x) std::string(typeid(x).name())

using namespace Catch::Matchers;
using namespace Catalyst::Runtime::Device::OpenQasm;

TEST_CASE("Test lookup openqasm gate names from QIR -> OpenQasm map", "[openqasm]")
{
    // Check lookup supported gates
    CHECK(lookup_qasm_gate_name("PauliX") == "x");
    CHECK(lookup_qasm_gate_name("Hadamard") == "h");
    CHECK(lookup_qasm_gate_name("CNOT") == "cnot");
    CHECK(lookup_qasm_gate_name("SWAP") == "swap");
    CHECK(lookup_qasm_gate_name("Toffoli") == "ccnot");

    // Check lookup an unsupported gate
    REQUIRE_THROWS_WITH(
        lookup_qasm_gate_name("ABC"),
        ContainsSubstring("The given QIR gate name is not supported by the OpenQASM builder"));
}

TEST_CASE("Test QasmVariable(type=Float) from OpenQasmBuilder", "[openqasm]")
{
    auto var = QasmVariable(VariableType::Float, "alpha");
    CHECK(var.getType() == VariableType::Float);
    CHECK(var.getName() == "alpha");

    std::string toqasm = "input float alpha;\n";
    CHECK(var.toOpenQasm() == toqasm);

    // Check edge cases
    auto var_buggy = QasmVariable(static_cast<VariableType>(2), "beta");
    REQUIRE_THROWS_WITH(
        var_buggy.toOpenQasm(),
        ContainsSubstring(
            "[Function:toOpenQasm] Error in Catalyst Runtime: Unsupported OpenQasm variable type"));
}

TEST_CASE("Test QasmRegister(type=Qubit) from OpenQasmBuilder", "[openqasm]")
{
    auto reg = QasmRegister(RegisterType::Qubit, "qubits", 5);
    CHECK(reg.getName() == "qubits");
    CHECK(reg.getSize() == 5);

    reg.updateSize(10);
    CHECK(reg.getSize() == 10);

    reg.resetSize();
    CHECK(reg.getSize() == 0);

    reg.updateSize(10);
    CHECK(!reg.isValidSlice({}));
    CHECK(reg.isValidSlice({0, 1, 4, 9}));
    CHECK(!reg.isValidSlice({0, 10}));

    std::string case1 = "qubit[10] qubits;\n";
    CHECK(reg.toOpenQasm(RegisterMode::Alloc) == case1);

    std::string case2 = "reset qubits;\n";
    CHECK(reg.toOpenQasm(RegisterMode::Reset) == case2);

    std::string case3 = "qubits[0], qubits[1], qubits[3]";
    CHECK(reg.toOpenQasm(RegisterMode::Slice, {0, 1, 3}) == case3);

    // Check edge cases
    REQUIRE_THROWS_WITH(
        reg.toOpenQasm(RegisterMode::Slice, {0, 10}),
        ContainsSubstring(
            "[Function:toOpenQasm] Error in Catalyst Runtime: Assertion: isValidSlice(slice)"));

    REQUIRE_THROWS_WITH(reg.toOpenQasm(static_cast<RegisterMode>(4), {0, 5}),
                        ContainsSubstring("[Function:toOpenQasm] Error in Catalyst Runtime: "
                                          "Unsupported OpenQasm register mode"));
}

TEST_CASE("Test QasmRegister(type=Bit) from OpenQasmBuilder", "[openqasm]")
{
    auto reg = QasmRegister(RegisterType::Bit, "bits", 5);
    CHECK(reg.getName() == "bits");
    CHECK(reg.getSize() == 5);

    reg.updateSize(10);
    CHECK(reg.getSize() == 10);

    reg.resetSize();
    CHECK(reg.getSize() == 0);

    reg.updateSize(10);
    CHECK(!reg.isValidSlice({}));
    CHECK(reg.isValidSlice({0, 1, 4, 9}));
    CHECK(!reg.isValidSlice({0, 10}));

    std::string case1 = "bit[10] bits;\n";
    CHECK(reg.toOpenQasm(RegisterMode::Alloc) == case1);

    std::string case2 = "reset bits;\n";
    CHECK(reg.toOpenQasm(RegisterMode::Reset) == case2);

    std::string case3 = "bits[0], bits[1], bits[3]";
    CHECK(reg.toOpenQasm(RegisterMode::Slice, {0, 1, 3}) == case3);

    // Check edge cases
    auto reg_buggy = QasmRegister(static_cast<RegisterType>(3), "random", 5);
    REQUIRE_THROWS_WITH(reg_buggy.toOpenQasm(RegisterMode::Alloc, {}),
                        ContainsSubstring("[Function:toOpenQasm] Error in Catalyst Runtime: "
                                          "Unsupported OpenQasm register type"));

    REQUIRE_THROWS_WITH(
        reg.toOpenQasm(RegisterMode::Slice, {0, 10}),
        ContainsSubstring(
            "[Function:toOpenQasm] Error in Catalyst Runtime: Assertion: isValidSlice(slice)"));

    REQUIRE_THROWS_WITH(reg.toOpenQasm(static_cast<RegisterMode>(4), {0, 5}),
                        ContainsSubstring("[Function:toOpenQasm] Error in Catalyst Runtime: "
                                          "Unsupported OpenQasm register mode"));
}

TEST_CASE("Test QasmGate from OpenQasmBuilder", "[openqasm]")
{
    auto qubits = QasmRegister(RegisterType::Qubit, "q", 5);

    // Check a supported gate without params
    auto gate1 = QasmGate("PauliX", {}, {}, {0}, false);
    CHECK(gate1.getName() == "x");
    CHECK(gate1.getParams().empty());
    CHECK(gate1.getWires().size() == 1);
    CHECK(gate1.getWires()[0] == 0);
    CHECK(gate1.getInverse() == false);

    std::string gate1_toqasm = "x q[0];\n";
    CHECK(gate1.toOpenQasm(qubits) == gate1_toqasm);

    // Check a ctrl gate without params
    auto gate2 = QasmGate("SWAP", {}, {}, {0, 2}, false);

    std::string gate2_toqasm = "swap q[0], q[2];\n";
    CHECK(gate2.toOpenQasm(qubits) == gate2_toqasm);

    // Check a gate with params (value)
    auto gate3 = QasmGate("RX", {0.361731}, {}, {2}, false);

    std::string gate3_toqasm = "rx(0.36) q[2];\n";
    CHECK(gate3.toOpenQasm(qubits, 2) == gate3_toqasm);

    std::string gate3_toqasm_2 = "rx(0.3617) q[2];\n";
    CHECK(gate3.toOpenQasm(qubits, 4) == gate3_toqasm_2);

    // Check a gate with params (name)
    auto gate4 = QasmGate("RX", {}, {"gamma"}, {2}, false);
    std::string gate4_toqasm = "rx(gamma) q[2];\n";
    CHECK(gate4.toOpenQasm(qubits, 2) == gate4_toqasm);

    // Check the QubitUnitary gate
    std::vector<std::complex<double>> mat{
        {0, 0},
        {0, -1},
        {0, 1},
        {0, 0},
    };
    auto gate5 = QasmGate(mat, {2}, false);
    CHECK(gate5.getMatrix() == mat);
    std::string gate5_toqasm = "#pragma braket unitary([[0, 0-1im], [0+1im, 0]]) q[2]\n";
    CHECK(gate5.toOpenQasm(qubits, 2) == gate5_toqasm);

    // Check a random gate with several params (value)
    // not a valid gate! This is just for testing...
    auto gate31 = QasmGate("RX", {0.123, 0.456}, {}, {2}, false);

    std::string gate31_toqasm = "rx(0.123, 0.456) q[2];\n";
    CHECK(gate31.toOpenQasm(qubits, 3) == gate31_toqasm);

    // Check a random gate with several params (name)
    // not a valid gate! This is just for testing...
    auto gate41 = QasmGate("RX", {}, {"alpha", "gamma"}, {2}, false);
    std::string gate41_toqasm = "rx(alpha, gamma) q[2];\n";
    CHECK(gate41.toOpenQasm(qubits, 2) == gate41_toqasm);

    // Check edge cases
    REQUIRE_THROWS_WITH(
        QasmGate("ABC", {}, {}, {}, true),
        ContainsSubstring(
            "[Function:lookup_qasm_gate_name] Error in Catalyst Runtime: The given QIR "
            "gate name is not supported by the OpenQASM builder"));

    REQUIRE_THROWS_WITH(
        QasmGate("RX", {0.314}, {"gamma"}, {2}, false),
        ContainsSubstring("[Function:QasmGate] Error in Catalyst Runtime: Parametric gates are "
                          "currently supported via either their values or names but not both."));
}

TEST_CASE("Test QasmMeasure from OpenQasmBuilder", "[openqasm]")
{
    auto qubits = QasmRegister(RegisterType::Qubit, "q", 5);

    auto mz1 = QasmMeasure(0, 0);
    CHECK(mz1.getWire() == 0);
    CHECK(mz1.getBit() == 0);

    std::string mz1_toqasm = "measure q[0];\n";
    CHECK(mz1.toOpenQasm(qubits) == mz1_toqasm);

    auto mz2 = QasmMeasure(1, 1);
    CHECK(mz2.getWire() == 1);
    CHECK(mz2.getBit() == 1);

    std::string mz2_toqasm = "measure q[1];\n";
    CHECK(mz2.toOpenQasm(qubits) == mz2_toqasm);

    auto bits = QasmRegister(RegisterType::Bit, "bit", 2);

    std::string mz1_res_toqasm = "bit[0] = measure q[0];\n";
    CHECK(mz1.toOpenQasm(bits, qubits) == mz1_res_toqasm);

    std::string mz2_res_toqasm = "bit[1] = measure q[1];\n";
    CHECK(mz2.toOpenQasm(bits, qubits) == mz2_res_toqasm);
}

TEST_CASE("Test MatrixBuilder", "[openqasm]")
{
    SECTION("matrix(2 * 2) vector(complex(double))")
    {
        std::vector<std::complex<double>> mat{
            {0, 0},
            {-0.1488, 0.360849},
            {0, 0},
            {-0.8818, -0.26456},
        };

        std::string expected = "[[0, -0.1+0.4im], [0, -0.9-0.3im]]";
        auto result = MatrixBuilder::toOpenQasm(mat, 2, 1);
        CHECK(result == expected);
    }

    SECTION("matrix(4 * 2) vector(complex(double))")
    {
        std::vector<std::complex<double>> mat{
            {-0.67094, -0.63044}, {-0.148854, 0.36084}, {-0.23763, 0.309679}, {-0.88183, -0.26456},
            {-0.67094, -0.63044}, {-0.148854, 0.36084}, {-0.23763, 0.309679}, {-0.88183, -0.26456},
        };

        std::string expected = "[[-0.671-0.63im, -0.149+0.361im], [-0.238+0.31im, -0.882-0.265im], "
                               "[-0.671-0.63im, -0.149+0.361im], [-0.238+0.31im, -0.882-0.265im]]";
        auto result = MatrixBuilder::toOpenQasm(mat, 2, 3);
        CHECK(result == expected);
    }

    SECTION("matrix(2 * 2) vector(double)")
    {
        std::vector<double> mat{
            -0.670,
            0.1488,
            -0.237,
            0.8818,
        };

        std::string expected = "[[-0.7, 0.1], [-0.2, 0.9]]";
        auto result = MatrixBuilder::toOpenQasm(mat, 2, 1);
        CHECK(result == expected);
    }

    SECTION("matrix(4 * 4) vector(double)")
    {
        std::vector<double> mat{
            -0.67094, 0.63044,  -0.148854, 0.36084, -0.23763, 0.309679, -0.88183, 0.26456,
            0.67094,  -0.63044, -0.148854, 0.36084, -0.23763, 0.309679, -0.88183, 0.26456,
        };

        std::string expected = "[[-0.671, 0.63, -0.149, 0.361], [-0.238, 0.31, -0.882, 0.265], "
                               "[0.671, -0.63, -0.149, 0.361], [-0.238, 0.31, -0.882, 0.265]]";
        auto result = MatrixBuilder::toOpenQasm(mat, 4, 3);
        CHECK(result == expected);
    }
}

TEST_CASE("Test QasmNamedObs from OpenQasmBuilder", "[openqasm]")
{
    auto qubits = QasmRegister(RegisterType::Qubit, "q", 5);

    auto obs_x = QasmNamedObs("PauliX", {0});
    CHECK(obs_x.getName() == "x");
    CHECK(obs_x.getWires()[0] == 0);
    CHECK(obs_x.toOpenQasm(qubits) == "x(q[0])");

    auto obs_h = QasmNamedObs("Hadamard", {3});
    CHECK(obs_h.getName() == "h");
    CHECK(obs_h.getWires()[0] == 3);
    CHECK(obs_h.toOpenQasm(qubits) == "h(q[3])");
}

TEST_CASE("Test QasmTensorObs from OpenQasmBuilder", "[openqasm]")
{
    auto qubits = QasmRegister(RegisterType::Qubit, "q", 5);

    auto obs_x = std::shared_ptr<QasmNamedObs>{new QasmNamedObs("PauliX", {0})};
    auto obs_y = std::shared_ptr<QasmNamedObs>{new QasmNamedObs("PauliY", {1})};
    auto obs_z = std::shared_ptr<QasmNamedObs>{new QasmNamedObs("PauliZ", {2})};
    auto obs_h = std::shared_ptr<QasmNamedObs>{new QasmNamedObs("Hadamard", {3})};
    auto obs_x2 = std::shared_ptr<QasmNamedObs>{new QasmNamedObs("PauliX", {2})};

    auto tp_x2 = QasmTensorObs(obs_x2);
    CHECK(tp_x2.getName() == "QasmTensorObs");
    CHECK(tp_x2.getWires()[0] == 2);
    CHECK(tp_x2.toOpenQasm(qubits) == "x(q[2])");

    auto tp_xyzh = QasmTensorObs(obs_x, obs_y, obs_z, obs_h);
    CHECK(tp_xyzh.getName() == "QasmTensorObs");
    std::vector<size_t> all_wires{0, 1, 2, 3};
    CHECK(tp_xyzh.getWires() == all_wires);
    CHECK(tp_xyzh.toOpenQasm(qubits) == "x(q[0]) @ y(q[1]) @ z(q[2]) @ h(q[3])");

    REQUIRE_THROWS_WITH(
        QasmTensorObs(obs_z, obs_x2),
        ContainsSubstring(
            "[Function:QasmTensorObs] Error in Catalyst Runtime: Invalid list of total wires"));
}

TEST_CASE("Test QasmHamiltonianObs from OpenQasmBuilder", "[openqasm]")
{
    auto qubits = QasmRegister(RegisterType::Qubit, "q", 5);

    auto obs_x = std::shared_ptr<QasmObs>{new QasmNamedObs("PauliX", {0})};
    auto obs_y = std::shared_ptr<QasmObs>{new QasmNamedObs("PauliY", {1})};
    auto obs_z = std::shared_ptr<QasmObs>{new QasmNamedObs("PauliZ", {2})};
    auto obs_h = std::shared_ptr<QasmObs>{new QasmNamedObs("Hadamard", {3})};
    auto obs_x2 = std::shared_ptr<QasmObs>{new QasmNamedObs("PauliX", {2})};

    auto tp_xx2h = std::shared_ptr<QasmObs>{new QasmTensorObs(obs_x, obs_x2, obs_h)};

    auto hl_x2 = QasmHamiltonianObs::create({0.2}, {obs_x2});
    CHECK(hl_x2->getName() == "QasmHamiltonianObs");
    CHECK(hl_x2->getWires()[0] == 2);
    CHECK(hl_x2->toOpenQasm(qubits) == "0.2 * x(q[2])");

    auto hl_mix = QasmHamiltonianObs::create({0.3, 0.5, 0.1}, {obs_y, obs_z, tp_xx2h});
    CHECK(hl_mix->getName() == "QasmHamiltonianObs");
    std::vector<size_t> all_wires{0, 1, 2, 3};
    CHECK(hl_mix->getWires() == all_wires);
    CHECK(hl_mix->toOpenQasm(qubits) ==
          "0.3 * y(q[1]) + 0.5 * z(q[2]) + 0.1 * x(q[0]) @ x(q[2]) @ h(q[3])");

    REQUIRE_THROWS_WITH(
        QasmHamiltonianObs::create({0.3}, {obs_y, obs_z}),
        ContainsSubstring("[Function:QasmHamiltonianObs] Error in Catalyst Runtime: "
                          "Assertion: obs.size() == coeffs.size()"));
}

TEMPLATE_TEST_CASE("Test OpenQasmBuilder with dumping the circuit header", "[openqasm]",
                   OpenQasmBuilder, BraketBuilder)
{
    auto builder = TestType();
    builder.Register(RegisterType::Qubit, "qubits", 5);

    std::string toqasm;
    if (TYPE_INFO(TestType) == TYPE_INFO(OpenQasmBuilder)) {
        toqasm = "OPENQASM 3.0;\n"
                 "qubit[5] qubits;\n"
                 "reset qubits;\n";
    }
    else if (TYPE_INFO(TestType) == TYPE_INFO(BraketBuilder)) {
        toqasm = "OPENQASM 3.0;\n"
                 "qubit[5] qubits;\n"
                 "bit[5] bits;\n"
                 "bits = measure qubits;\n";
    }

    CHECK(builder.toOpenQasm() == toqasm);

    // Check edge cases
    builder.Register(RegisterType::Qubit, "qubits2", 3);

    REQUIRE_THROWS_WITH(builder.toOpenQasm(),
                        ContainsSubstring("[Function:toOpenQasm] Error in Catalyst Runtime: Invalid"
                                          " number of quantum registers"));
}

TEMPLATE_TEST_CASE("Test OpenQasmBuilder with invalid number of measurement results registers",
                   "[openqasm]", OpenQasmBuilder, BraketBuilder)
{
    auto builder = OpenQasmBuilder();
    CHECK(builder.getNumQubits() == 0);
    CHECK(builder.getNumBits() == 0);

    builder.Register(RegisterType::Qubit, "qubits", 5);

    builder.Register(RegisterType::Bit, "bits1", 2);
    builder.Register(RegisterType::Bit, "bits2", 3);

    CHECK(builder.getNumQubits() == 5);
    CHECK(builder.getNumBits() == 5);

    // Check edge cases
    REQUIRE_THROWS_WITH(builder.toOpenQasm(),
                        ContainsSubstring("[Function:toOpenQasm] Error in Catalyst Runtime: Invalid"
                                          " number of measurement results registers"));

    REQUIRE_THROWS_WITH(
        builder.Register(static_cast<RegisterType>(3), "qubits", 5),
        ContainsSubstring(
            "[Function:Register] Error in Catalyst Runtime: Unsupported OpenQasm register type"));
}

TEMPLATE_TEST_CASE("Test OpenQasmBuilder with dumping the circuit header, gates, and measure",
                   "[openqasm]", OpenQasmBuilder, BraketBuilder)
{
    auto builder = TestType();

    builder.Register(RegisterType::Qubit, "q", 2);

    builder.Gate("PauliX", {}, {}, {0}, false);
    builder.Gate("Hadamard", {}, {}, {1}, false);
    builder.Gate("SWAP", {}, {}, {0, 1}, false);
    builder.Gate("RZ", {0.12}, {}, {1}, false);
    builder.Gate("RX", {}, {"alpha"}, {0}, false);
    builder.Gate("ISWAP", {}, {}, {0, 1}, false);
    builder.Gate("PSWAP", {0.34}, {}, {0, 1}, false);

    std::string toqasm;
    if (TYPE_INFO(TestType) == TYPE_INFO(OpenQasmBuilder)) {
        builder.Register(RegisterType::Bit, "b", 2);

        builder.Measure(0, 0);
        builder.Measure(1, 1);

        toqasm = "OPENQASM 3.0;\n"
                 "input float alpha;\n"
                 "qubit[2] q;\n"
                 "bit[2] b;\n"
                 "x q[0];\n"
                 "h q[1];\n"
                 "swap q[0], q[1];\n"
                 "rz(0.12) q[1];\n"
                 "rx(alpha) q[0];\n"
                 "iswap q[0], q[1];\n"
                 "pswap(0.34) q[0], q[1];\n"
                 "b[0] = measure q[0];\n"
                 "b[1] = measure q[1];\n"
                 "reset q;\n";
    }
    else if (TYPE_INFO(TestType) == TYPE_INFO(BraketBuilder)) {
        toqasm = "OPENQASM 3.0;\n"
                 "input float alpha;\n"
                 "qubit[2] q;\n"
                 "bit[2] bits;\n"
                 "x q[0];\n"
                 "h q[1];\n"
                 "swap q[0], q[1];\n"
                 "rz(0.12) q[1];\n"
                 "rx(alpha) q[0];\n"
                 "iswap q[0], q[1];\n"
                 "pswap(0.34) q[0], q[1];\n"
                 "bits = measure q;\n";
    }

    CHECK(builder.toOpenQasm() == toqasm);
}

TEMPLATE_TEST_CASE("Test OpenQasmBuilder with dumping a circuit without measurement results",
                   "[openqasm]", OpenQasmBuilder, BraketBuilder)
{
    auto builder = TestType();

    builder.Register(RegisterType::Qubit, "q", 2);

    builder.Gate("PauliX", {}, {}, {0}, false);
    builder.Gate("Hadamard", {}, {}, {1}, false);

    builder.Measure(0, 0);
    builder.Measure(0, 1);

    std::string toqasm;
    if (TYPE_INFO(TestType) == TYPE_INFO(OpenQasmBuilder)) {
        toqasm = "OPENQASM 3.0;\n"
                 "qubit[2] q;\n"
                 "x q[0];\n"
                 "h q[1];\n"
                 "measure q[0];\n"
                 "measure q[1];\n"
                 "reset q;\n";
    }
    else if (TYPE_INFO(TestType) == TYPE_INFO(BraketBuilder)) {
        toqasm = "OPENQASM 3.0;\n"
                 "qubit[2] q;\n"
                 "bit[2] bits;\n"
                 "x q[0];\n"
                 "h q[1];\n"
                 "bits = measure q;\n";
    }

    CHECK(builder.toOpenQasm() == toqasm);
}

TEMPLATE_TEST_CASE("Test OpenQasmBuilder with custom instructions", "[openqasm]", OpenQasmBuilder,
                   BraketBuilder)
{
    auto builder = TestType();

    builder.Register(RegisterType::Qubit, "q", 2);

    builder.Gate("PauliX", {}, {}, {0}, false);
    builder.Gate("Hadamard", {}, {}, {1}, false);

    builder.Measure(0, 0);
    builder.Measure(0, 1);

    if (TYPE_INFO(TestType) == TYPE_INFO(OpenQasmBuilder)) {
        REQUIRE_THROWS_WITH(
            builder.toOpenQasmWithCustomInstructions(""),
            ContainsSubstring("Error in Catalyst Runtime: Unsupported functionality"));
    }
    else if (TYPE_INFO(TestType) == TYPE_INFO(BraketBuilder)) {
        std::string toqasm = "OPENQASM 3.0;\n"
                             "qubit[2] q;\n"
                             "x q[0];\n"
                             "h q[1];\n";

        std::string expval_pragma_str = "#pragma braket result expectation x(q[0]) @ y([q1])\n";
        CHECK(builder.toOpenQasmWithCustomInstructions(expval_pragma_str) ==
              toqasm + expval_pragma_str);

        std::string var_pragma_str = "#pragma braket result variance x(q[0]) @ y([q1])\n";
        CHECK(builder.toOpenQasmWithCustomInstructions(var_pragma_str) == toqasm + var_pragma_str);

        std::string state_pragma_str = "#pragma braket result state_vector\n";
        CHECK(builder.toOpenQasmWithCustomInstructions(state_pragma_str) ==
              toqasm + state_pragma_str);
    }
}
