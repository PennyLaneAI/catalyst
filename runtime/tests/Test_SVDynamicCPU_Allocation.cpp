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
#include "StateVectorDynamicCPU.hpp"
#include "StateVectorRawCPU.hpp"
#include "Util.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::getSubsystemPurity /allocation",
                   "[StateVectorDynamicCPU]", float, double)
{
    using PrecisionT = TestType;

    SECTION("Test getSubsystemPurity for a state-vector with RX-RY")
    {
        constexpr size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false}, {{0.1}, {0.2}});

        REQUIRE(sv1.getSubsystemPurity(0) == approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.getSubsystemPurity(1) == approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.getSubsystemPurity(2) == approx(std::complex<PrecisionT>{1, 0}));
    }

    SECTION("Test checkSubsystemPurity for a state-vector with RX-RY")
    {
        constexpr size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false}, {{0.1}, {0.2}});
        REQUIRE((sv1.checkSubsystemPurity(0) && sv1.checkSubsystemPurity(1)));
        REQUIRE(sv1.checkSubsystemPurity(2));
    }

    SECTION("Test getSubsystemPurity for a state-vector with CNOT-RY")
    {
        constexpr size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"CNOT", "RY"}, {{0, 1}, {1}}, {false, false}, {{}, {0.2}});

        REQUIRE(sv1.getSubsystemPurity(0) == approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.getSubsystemPurity(1) == approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.checkSubsystemPurity(2));
    }

    SECTION("Test getSubsystemPurity for a custom state-vector")
    {
        constexpr size_t num_qubits = 2;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        std::vector<std::complex<PrecisionT>> data{
            {1 / 2, 0}, {1 / 2, 0}, {-1 / 2, 0}, {-1 / 2, 0}};

        sv1.updateData(data);

        REQUIRE(sv1.getSubsystemPurity(0) != approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.getSubsystemPurity(1) != approx(std::complex<PrecisionT>{1, 0}));
    }
}

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::allocateWire /allocation", "[StateVectorDynamicCPU]",
                   float, double)
{
    using PrecisionT = TestType;

    SECTION("applyOperations with released wires")
    {
        constexpr size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);
        size_t new_idx = sv1.allocateWire();
        sv1.releaseWire(new_idx);

        REQUIRE((sv1.getNumQubits() == num_qubits && new_idx == num_qubits));

        REQUIRE_NOTHROW(
            sv1.applyOperations({"PauliX", "PauliY"}, {{new_idx + 1}, {1}}, {false, false}));
    }

    SECTION("Test counting wires for a simple state-vector")
    {
        constexpr size_t num_qubits = 10;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.releaseWire(1);
        sv1.releaseWire(3);
        sv1.releaseWire(4);
        sv1.releaseWire(6);
        sv1.allocateWire();
        sv1.allocateWire();
        sv1.allocateWire();
        sv1.allocateWire();

        REQUIRE(sv1.getNumQubits() == 10);
    }

    SECTION("Test the validity of the shranked state vector in the second half")
    {
        constexpr size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);
        std::vector<std::complex<PrecisionT>> data(1 << num_qubits, {0.0, 0.0});
        data[(1 << num_qubits) - 1] = {1.0, 0.0};
        sv1.updateData(data);

        sv1.releaseWire(0);

        REQUIRE(sv1.getNumQubits() == 3);
    }

    SECTION("Test allocation/deallocation of a customed state-vector")
    {
        StateVectorDynamicCPU<PrecisionT> sv1(0);
        size_t idx_0 = sv1.allocateWire(); // 1, 0

        std::vector<std::complex<PrecisionT>> expected_data{{1.0, 0.0}, {0.0, 0.0}};

        REQUIRE(sv1.getDataVector() == approx(expected_data));

        sv1.applyOperation("Hadamard", {idx_0}, false, {});
        expected_data[0] = std::complex<PrecisionT>(0.707107, 0);
        expected_data[1] = std::complex<PrecisionT>(0.707107, 0);

        REQUIRE(sv1.getDataVector() == approx(expected_data));

        sv1.allocateWire();

        std::vector<std::complex<PrecisionT>> expected_data_n2{
            {0.707107, 0}, {0.0, 0.0}, {0.707107, 0}, {0.0, 0.0}};

        REQUIRE(sv1.getDataVector() == approx(expected_data_n2));

        sv1.allocateWire();

        std::vector<std::complex<PrecisionT>> expected_data_n3{
            {0.707107, 0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
            {0.707107, 0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

        REQUIRE(sv1.getDataVector() == approx(expected_data_n3));

        sv1.releaseWire(0);

        REQUIRE(sv1.getDataVector() == approx(expected_data_n2));

        sv1.releaseWire(0);

        REQUIRE(sv1.getDataVector() == approx(expected_data));

        sv1.releaseWire(0);

        REQUIRE(sv1.getDataVector()[0] == approx(std::complex<PrecisionT>(1.0, 0.0)));
    }

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=0")
    {
        StateVectorDynamicCPU<PrecisionT> sv1(0);

        std::vector<std::complex<PrecisionT>> expected_data{{1, 0}};
        REQUIRE(sv1.getDataVector() == approx(expected_data));

        size_t idx_0 = sv1.allocateWire();

        expected_data.push_back({0, 0});
        REQUIRE(sv1.getDataVector() == approx(expected_data));

        sv1.applyOperation("Hadamard", {idx_0}, false, {});

        StateVectorDynamicCPU<PrecisionT> sv2 = sv1;

        size_t new_idx = sv1.allocateWire();
        sv1.applyOperation("RX", {new_idx}, false, {0.3});

        sv1.releaseWire(0);
        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=1")
    {
        constexpr size_t num_qubits = 1;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);
        sv1.applyOperation("Hadamard", {0}, false, {});

        StateVectorDynamicCPU<PrecisionT> sv2 = sv1;

        size_t new_idx = sv1.allocateWire();
        sv1.applyOperation("RX", {new_idx}, false, {0.3});

        sv1.releaseWire(0);
        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=2")
    {
        constexpr size_t num_qubits = 2;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);
        sv1.applyOperations({"RX", "CNOT"}, {{0}, {0, 1}}, {false, false}, {{0.4}, {}});

        StateVectorDynamicCPU<PrecisionT> sv2 = sv1;

        size_t new_idx = sv1.allocateWire();
        sv1.applyOperation("RX", {new_idx}, false, {0.3});

        sv1.releaseWire(0);

        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=3")
    {
        constexpr size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"RX", "SWAP"}, {{0}, {0, 2}}, {false, false}, {{0.4}, {}});

        StateVectorDynamicCPU<PrecisionT> sv2{num_qubits - 1};
        sv2.applyOperations({"RX", "SWAP"}, {{0}, {0, 1}}, {false, false}, {{0.4}, {}});

        sv1.releaseWire(1);
        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=4")
    {
        constexpr size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations(
            {"RX", "SWAP", "RY", "Hadamard", "RZ", "CNOT"}, {{0}, {1, 2}, {1}, {3}, {2}, {1, 3}},
            {false, false, false, false, false, false}, {{0.4}, {}, {0.6}, {}, {0.8}, {}});

        std::vector<std::complex<TestType>> result{
            {0.651289, -0.27536},
            {0.651289, -0.27536},
        };

        sv1.releaseWire(1);
        sv1.releaseWire(1);
        sv1.releaseWire(1);

        REQUIRE(sv1.getDataVector() == approx(result));
    }
}
