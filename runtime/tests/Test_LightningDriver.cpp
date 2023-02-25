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

#include <numeric>
#include <string>

#include "BaseUtils.hpp"
#include "LightningUtils.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Simulator;

TEST_CASE("lightning Basis vector", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    QubitIdType q = sim->AllocateQubit();
    q = sim->AllocateQubit();
    q = sim->AllocateQubit();

    sim->ReleaseQubit(q);

    auto state = sim->State();
    REQUIRE(state[0].real() == Approx(1.0).epsilon(1e-5));
    REQUIRE(state[0].imag() == Approx(0.0).epsilon(1e-5));
    REQUIRE(state[1].real() == Approx(0.0).epsilon(1e-5));
    REQUIRE(state[1].imag() == Approx(0.0).epsilon(1e-5));
    REQUIRE(state[2].real() == Approx(0.0).epsilon(1e-5));
    REQUIRE(state[2].imag() == Approx(0.0).epsilon(1e-5));
    REQUIRE(state[3].real() == Approx(0.0).epsilon(1e-5));
    REQUIRE(state[3].imag() == Approx(0.0).epsilon(1e-5));
}

TEST_CASE("Qubit allocatation and deallocation", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    constexpr size_t n = 1;
    constexpr size_t sz = (1UL << n);

    QubitIdType q;
    for (size_t i = 0; i < n; i++) {
        q = sim->AllocateQubit();
    }

    REQUIRE(n == static_cast<size_t>(q) + 1);

    VectorCplxT<double> state = sim->State();

    REQUIRE(state.size() == (1ul << n));
    REQUIRE(state[0].real() == Approx(1.0).epsilon(1e-5));
    REQUIRE(state[0].imag() == Approx(0.0).epsilon(1e-5));

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < sz; i++) {
        sum += state[i];
    }

    REQUIRE(sum.real() == Approx(0.0).epsilon(1e-5));
    REQUIRE(sum.imag() == Approx(0.0).epsilon(1e-5));

#ifndef _KOKKOS
    for (size_t i = n; i > 0; i--) {
        REQUIRE(state.size() == sz);

        sim->ReleaseQubit(i - 1);
        sim->AllocateQubit();
        state = sim->State();
    }
#else
    for (size_t i = n; i > 0; i--) {
        sim->ReleaseQubit(i - 1);
    }
#endif
}

TEST_CASE("test AllocateQubits", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    REQUIRE(sim->AllocateQubits(0).size() == 0);

    auto &&q = sim->AllocateQubits(2);

    sim->ReleaseQubit(q[0]);

    auto state = sim->State();
    REQUIRE(state[0].real() == Approx(1.0).epsilon(1e-5));
}

TEST_CASE("compute register tests", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    constexpr size_t n = 10;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    // allocate a few qubits
    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    // release some of them
    sim->ReleaseQubit(n - 1);
    sim->ReleaseQubit(n - 2);

    const size_t new_n = n - 2;

    // check the correctness
    std::vector<QubitIdType> Qs_expected(new_n);
    std::iota(Qs_expected.begin(), Qs_expected.end(), static_cast<QubitIdType>(0));

    for (size_t i = 0; i < new_n; i++) {
        REQUIRE(Qs_expected[i] == Qs[i]);
    }
}

TEST_CASE("Check an unsupported operation", "[lightning]")
{
    REQUIRE_THROWS_WITH(
        Lightning::lookup_gates(Lightning::simulator_gate_info, "UnsupportedGateName"),
        Catch::Contains("The given operation is not supported by the simulator"));
}

TEST_CASE("QuantumDevice object test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 10;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->NamedOperation("Identity", {}, {Qs[0]}, false);
    sim->NamedOperation("Identity", {}, {Qs[2]}, false);
    sim->NamedOperation("Identity", {}, {Qs[4]}, false);
    sim->NamedOperation("Identity", {}, {Qs[6]}, false);
    sim->NamedOperation("Identity", {}, {Qs[8]}, false);

    VectorCplxT<double> out_state = sim->State();

    REQUIRE(out_state[0].real() == Approx(1.0).epsilon(1e-5));
    REQUIRE(out_state[0].imag() == Approx(0.0).epsilon(1e-5));

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < out_state.size(); i++) {
        sum += out_state[i];
    }

    REQUIRE(sum.real() == Approx(0.0).epsilon(1e-5));
    REQUIRE(sum.imag() == Approx(0.0).epsilon(1e-5));

    for (size_t i = 0; i < n; i++) {
        sim->ReleaseQubit(static_cast<QubitIdType>(i));
        // 0, 1, 2, ..., 9
    }

#ifndef _KOKKOS
    for (size_t i = 10; i < n + 10; i++) {
        REQUIRE(i == sim->AllocateQubit());
        // 10, 11, ..., 19
    }

    for (size_t i = 10; i < n + 10; i++) {
        sim->ReleaseQubit(static_cast<QubitIdType>(i));
        // 10, 11, ..., 19
    }

    for (size_t i = 20; i < n + 20; i++) {
        REQUIRE(i == sim->AllocateQubit());
        // 20, 21, ..., 29
    }
#endif
}
