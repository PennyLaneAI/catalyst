#include "InstructionStrBuilder.hpp"
#include "TestUtils.hpp"

TEST_CASE("string building is correct for instructions that require up to one parameter ",
          "[InstructionStrBuilder]")
{
    InstructionStrBuilder str_builder;
    CHECK(str_builder.get_simple_op_str("AllocateQubit", "") == "AllocateQubit()");
    CHECK(str_builder.get_simple_op_str("AllocateQubits", 5) == "AllocateQubits(5)");
    CHECK(str_builder.get_simple_op_str("ReleaseQubit", "") == "ReleaseQubit()");
    CHECK(str_builder.get_simple_op_str("ReleaseAllQubits", "") == "ReleaseAllQubits()");
    CHECK(str_builder.get_simple_op_str("State", "") == "State()");
    CHECK(str_builder.get_simple_op_str("Probs", "") == "Probs()");
    CHECK(str_builder.get_simple_op_str("Measure", 0) == "Measure(0)");
}

TEST_CASE("string building is correct for instructions that require a parameter of type ObsIdType",
          "[InstructionStrBuilder]")
{
    InstructionStrBuilder str_builder;
    auto id = str_builder.create_obs_str(ObsId::PauliX, {}, {0});

    CHECK(str_builder.get_op_with_obs_str("Expval", id) == "Expval(PauliX(0))");
    CHECK(str_builder.get_op_with_obs_str("Var", id) == "Var(PauliX(0))");
}

TEST_CASE("string building is correct for NamedOperation", "[InstructionStrBuilder]")
{
    InstructionStrBuilder str_builder;

    CHECK(str_builder.get_named_op_str("NamedOperation", {3.14, 0.705}, {}) ==
          "NamedOperation(3.140000, 0.705000)");
    CHECK(str_builder.get_named_op_str("NamedOperation", {3.14, 0.705}, {0}) ==
          "NamedOperation(3.140000, 0.705000, wires=[0])");
    CHECK(str_builder.get_named_op_str("NamedOperation", {}, {0}) == "NamedOperation(wires=[0])");
    CHECK(str_builder.get_named_op_str("NamedOperation", {}, {0}, true) ==
          "NamedOperation(wires=[0], inverse=true)");
    CHECK(str_builder.get_named_op_str("NamedOperation", {}, {1}, false, {0}) ==
          "NamedOperation(wires=[1], control=[0])");
    CHECK(str_builder.get_named_op_str("NamedOperation", {}, {1}, false, {0}, {true}) ==
          "NamedOperation(wires=[1], control=[0], control_value=[1])");
    CHECK(str_builder.get_named_op_str("PartialProbs", {}, {0}) == "PartialProbs(wires=[0])");
}

TEST_CASE("string building is correct for MatrixOperation", "[InstructionStrBuilder]")
{
    InstructionStrBuilder str_builder;
    std::vector<std::complex<double>> v = {std::complex<double>(0.707, -0.707),
                                           std::complex<double>(0.0), std::complex<double>(0.0),
                                           std::complex<double>(1.0)};

    CHECK(str_builder.get_matrix_op_str(v, {}) == "MatrixOperation([0.707 - 0.707i, 0, 0, 1])");
    CHECK(str_builder.get_matrix_op_str(v, {0}) ==
          "MatrixOperation([0.707 - 0.707i, 0, 0, 1], wires=[0])");
    CHECK(str_builder.get_matrix_op_str(v, {0}, true) ==
          "MatrixOperation([0.707 - 0.707i, 0, 0, 1], wires=[0], inverse=true)");
    CHECK(str_builder.get_matrix_op_str(v, {1}, false, {0}) ==
          "MatrixOperation([0.707 - 0.707i, 0, 0, 1], wires=[1], control=[0])");
    CHECK(str_builder.get_matrix_op_str(v, {1}, false, {0}, {true}) ==
          "MatrixOperation([0.707 - 0.707i, 0, 0, 1], wires=[1], control=[0], control_value=[1])");
}

TEST_CASE("registers correctly a new observable", "[InstructionStrBuilder]")
{
    InstructionStrBuilder str_builder;

    ObsId obs_id1 = ObsId::PauliX;
    ObsId obs_id2 = ObsId::Hermitian;

    CHECK(str_builder.create_obs_str(obs_id1, {}, {0}) == 0);
    CHECK(str_builder.create_obs_str(obs_id2, {1, 0, 0, 1}, {0}) == 1);
    CHECK(str_builder.get_obs_str(0) == "PauliX(0)");
    CHECK(str_builder.get_obs_str(1) == "Hermitian([1, 0, 0, 1], wires=[0])");

    // test tensor product observable
    CHECK(str_builder.create_tensor_obs_str({0, 1}) == 2);
    CHECK(str_builder.get_obs_str(2) == "PauliX(0) ⊗ Hermitian([1, 0, 0, 1], wires=[0])");

    // test hamiltonian observable
    CHECK(str_builder.create_hamiltonian_obs_str({0.1, 0.2}, {0, 1}) == 3);
    CHECK(str_builder.get_obs_str(3) == "0.1*PauliX(0) + 0.2*Hermitian([1, 0, 0, 1], wires=[0])");

    CHECK(str_builder.create_hamiltonian_obs_str({0.0, 0.2}, {0, 1}) == 4);
    CHECK(str_builder.get_obs_str(4) == "0.2*Hermitian([1, 0, 0, 1], wires=[0])");

    CHECK(str_builder.create_hamiltonian_obs_str({0.1, -0.2}, {0, 1}) == 5);
    CHECK(str_builder.get_obs_str(5) == "0.1*PauliX(0) - 0.2*Hermitian([1, 0, 0, 1], wires=[0])");
}

TEST_CASE("string building for Counts() and PartialCounts()", "[InstructionStrBuilder]")
{
    InstructionStrBuilder str_builder;
    CHECK(str_builder.get_distribution_op_str("Counts", 100) == "Counts(shots=100)");
    CHECK(str_builder.get_distribution_op_str("PartialCounts", 100, {0}) ==
          "PartialCounts(shots=100, wires=[0])");
}
