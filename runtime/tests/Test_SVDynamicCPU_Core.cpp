// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "LinearAlgebra.hpp"
#include "StateVectorLQubitDynamic.hpp"
#include "Util.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"
#include <StateVectorLQubit.hpp>

#include "TestHelpers.hpp"

using namespace Pennylane::LightningQubit;

TEMPLATE_TEST_CASE("StateVectorLQubitDynamic::StateVectorLQubitDynamic",
                   "[StateVectorLQubitDynamic]", float, double)
{
    using PrecisionT = TestType;

    SECTION("StateVectorLQubitDynamic")
    {
        CHECK(!std::is_constructible_v<StateVectorLQubitDynamic<>>);
    }
    SECTION("StateVectorLQubitDynamic<TestType>")
    {
        CHECK(!std::is_constructible_v<StateVectorLQubitDynamic<TestType>>);
    }
    SECTION("StateVectorLQubitDynamic<TestType> {size_t}")
    {
        CHECK(std::is_constructible_v<StateVectorLQubitDynamic<TestType>, size_t>);
        const size_t num_qubits = 4;
        StateVectorLQubitDynamic<PrecisionT> sv(num_qubits);

        CHECK(sv.getNumQubits() == 4);
        CHECK(sv.getLength() == 16);
        CHECK(sv.getDataVector().size() == 16);
    }
    SECTION("StateVectorLQubitDynamic<TestType> {const "
            "StateVectorLQubitDynamic<TestType>&}")
    {
        CHECK(std::is_copy_constructible_v<StateVectorLQubitDynamic<TestType>>);
    }
    SECTION("StateVectorLQubitDynamic<TestType> {StateVectorLQubitDynamic<TestType>&&}")
    {
        CHECK(std::is_move_constructible_v<StateVectorLQubitDynamic<TestType>>);
    }
    SECTION("Aligned 256bit statevector")
    {
        const auto memory_model = CPUMemoryModel::Aligned256;
        StateVectorLQubitDynamic<PrecisionT> sv(4, Threading::SingleThread, memory_model);
        /* Even when we allocate 256 bit aligend memory, it is possible that the
         * alignment happens to be 512 bit */
        CHECK(((getMemoryModel(sv.getDataVector().data()) == CPUMemoryModel::Aligned256) ||
               (getMemoryModel(sv.getDataVector().data()) == CPUMemoryModel::Aligned512)));
    }

    SECTION("Aligned 512bit statevector")
    {
        const auto memory_model = CPUMemoryModel::Aligned512;
        StateVectorLQubitDynamic<PrecisionT> sv(4, Threading::SingleThread, memory_model);
        CHECK((getMemoryModel(sv.getDataVector().data()) == CPUMemoryModel::Aligned512));
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitDynamic::applyMatrix with std::vector",
                   "[StateVectorLQubitDynamic]", float, double)
{
    using PrecisionT = TestType;
    SECTION("Test wrong matrix size")
    {
        std::vector<std::complex<TestType>> m(7, 0.0);
        const size_t num_qubits = 4;
        StateVectorLQubitDynamic<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m, {0, 1}),
                            Catch::Contains("The size of matrix does not match with the given"));
    }

    SECTION("Test wrong number of wires")
    {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorLQubitDynamic<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m, {0}),
                            Catch::Contains("The size of matrix does not match with the given"));
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitDynamic::applyMatrix with a pointer",
                   "[StateVectorLQubitDynamic]", float, double)
{
    using PrecisionT = TestType;
    SECTION("Test wrong matrix")
    {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorLQubitDynamic<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m.data(), {}), Catch::Contains("must be larger than 0"));
    }

    SECTION("Test with different number of wires")
    {
        std::default_random_engine re{1337};
        const size_t num_qubits = 5;
        for (size_t num_wires = 1; num_wires < num_qubits; num_wires++) {
            StateVectorLQubitDynamic<PrecisionT> sv1(num_qubits);
            StateVectorLQubitDynamic<PrecisionT> sv2(num_qubits);

            std::vector<size_t> wires(num_wires);
            std::iota(wires.begin(), wires.end(), 0);

            const auto m = Pennylane::Util::randomUnitary<PrecisionT>(re, num_wires);
            sv1.applyMatrix(m, wires);
            Gates::GateImplementationsPI::applyMultiQubitOp<PrecisionT>(sv2.getData(), num_qubits,
                                                                        m.data(), wires, false);
            CHECK(sv1.getDataVector() == approx(sv2.getDataVector()).margin(PrecisionT{1e-5}));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitDynamic::applyOperations", "[StateVectorLQubitDynamic]", float,
                   double)
{
    using PrecisionT = TestType;

    std::mt19937 re{1337};

    SECTION("Test invalid arguments without params")
    {
        const size_t num_qubits = 4;
        StateVectorLQubitDynamic<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyOperations({"PauliX", "PauliY"}, {{0}}, {false, false}),
                            Catch::Contains("must all be equal")); // invalid wires
        REQUIRE_THROWS_WITH(sv.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false}),
                            Catch::Contains("must all be equal")); // invalid inverse
    }

    SECTION("applyOperations without params works as expected")
    {
        const size_t num_qubits = 3;
        StateVectorLQubitDynamic<PrecisionT> sv1(num_qubits);

        sv1.updateData(Pennylane::Util::createRandomStateVectorData<PrecisionT>(re, num_qubits));
        StateVectorLQubitDynamic<PrecisionT> sv2 = sv1;

        sv1.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false, false});

        sv2.applyOperation("PauliX", {0}, false);
        sv2.applyOperation("PauliY", {1}, false);

        CHECK(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test invalid arguments with params")
    {
        const size_t num_qubits = 4;
        StateVectorLQubitDynamic<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyOperations({"RX", "RY"}, {{0}}, {false, false}, {{0.0}, {0.0}}),
                            Catch::Contains("must all be equal")); // invalid wires
        REQUIRE_THROWS_WITH(sv.applyOperations({"RX", "RY"}, {{0}, {1}}, {false}, {{0.0}, {0.0}}),
                            Catch::Contains("must all be equal")); // invalid inverse

        REQUIRE_THROWS_WITH(sv.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false}, {{0.0}}),
                            Catch::Contains("must all be equal")); // invalid params
    }

    SECTION("applyOperations with params works as expected")
    {
        const size_t num_qubits = 3;
        StateVectorLQubitDynamic<PrecisionT> sv1(num_qubits);

        sv1.updateData(Pennylane::Util::createRandomStateVectorData<PrecisionT>(re, num_qubits));
        StateVectorLQubitDynamic<PrecisionT> sv2 = sv1;

        sv1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false}, {{0.1}, {0.2}});

        sv2.applyOperation("RX", {0}, false, {0.1});
        sv2.applyOperation("RY", {1}, false, {0.2});

        CHECK(sv1.getDataVector() == approx(sv2.getDataVector()));
    }
}
