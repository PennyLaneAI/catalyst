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

#include "Utils.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"

#include "LightningSimulator.hpp"
#include "LightningKokkosSimulator.hpp"

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Simulator;

TEMPLATE_TEST_CASE("lightning Basis vector", "[Driver]", LightningSimulator, LightningKokkosSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();
    
    QubitIdType q = sim->AllocateQubit();
    q = sim->AllocateQubit();
    q = sim->AllocateQubit();

    sim->ReleaseQubit(q);

    auto state = sim->State();
    CHECK(state[0].real() == Approx(1.0).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0.0).epsilon(1e-5));
    CHECK(state[1].real() == Approx(0.0).epsilon(1e-5));
    CHECK(state[1].imag() == Approx(0.0).epsilon(1e-5));
    CHECK(state[2].real() == Approx(0.0).epsilon(1e-5));
    CHECK(state[2].imag() == Approx(0.0).epsilon(1e-5));
    CHECK(state[3].real() == Approx(0.0).epsilon(1e-5));
    CHECK(state[3].imag() == Approx(0.0).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("Qubit allocatation and deallocation", "[Driver]", LightningSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    constexpr size_t n = 1;
    constexpr size_t sz = (1UL << n);

    QubitIdType q;
    for (size_t i = 0; i < n; i++) {
        q = sim->AllocateQubit();
    }

    CHECK(n == static_cast<size_t>(q) + 1);

    std::vector<std::complex<double>> state = sim->State();

    CHECK(state.size() == (1UL << n));
    CHECK(state[0].real() == Approx(1.0).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0.0).epsilon(1e-5));

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < sz; i++) {
        sum += state[i];
    }

    CHECK(sum.real() == Approx(0.0).epsilon(1e-5));
    CHECK(sum.imag() == Approx(0.0).epsilon(1e-5));

    for (size_t i = n; i > 0; i--) {
        CHECK(state.size() == sz);

        sim->ReleaseQubit(i - 1);
        sim->AllocateQubit();
        state = sim->State();
    }
}

TEMPLATE_TEST_CASE("Qubit allocatation and deallocation [lightning.kokkos]", "[Driver]", LightningKokkosSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

    constexpr size_t n = 1;
    constexpr size_t sz = (1UL << n);

    QubitIdType q;
    for (size_t i = 0; i < n; i++) {
        q = sim->AllocateQubit();
    }

    CHECK(n == static_cast<size_t>(q) + 1);

    std::vector<std::complex<double>> state = sim->State();

    CHECK(state.size() == (1UL << n));
    CHECK(state[0].real() == Approx(1.0).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0.0).epsilon(1e-5));

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < sz; i++) {
        sum += state[i];
    }

    CHECK(sum.real() == Approx(0.0).epsilon(1e-5));
    CHECK(sum.imag() == Approx(0.0).epsilon(1e-5));

    for (size_t i = n; i > 0; i--) {
        sim->ReleaseQubit(i - 1);
    }
}

TEMPLATE_TEST_CASE("test AllocateQubits", "[Driver]", LightningSimulator, LightningKokkosSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();
    
    CHECK(sim->AllocateQubits(0).size() == 0);

    auto &&q = sim->AllocateQubits(2);

    sim->ReleaseQubit(q[0]);

    auto state = sim->State();
    CHECK(state[0].real() == Approx(1.0).epsilon(1e-5));
}

TEMPLATE_TEST_CASE("test DeviceShots", "[Driver]", LightningSimulator, LightningKokkosSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();
    
    CHECK(sim->GetDeviceShots() == 1000);

    sim->SetDeviceShots(500);

    CHECK(sim->GetDeviceShots() == 500);
}

TEMPLATE_TEST_CASE("compute register tests", "[Driver]", LightningSimulator, LightningKokkosSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();
    
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
        CHECK(Qs_expected[i] == Qs[i]);
    }
}

TEMPLATE_TEST_CASE("Check an unsupported operation", "[Driver]", LightningSimulator, LightningKokkosSimulator)
{
    REQUIRE_THROWS_WITH(
        Lightning::lookup_gates(Lightning::simulator_gate_info, "UnsupportedGateName"),
        Catch::Contains("The given operation is not supported by the simulator"));
}

TEMPLATE_TEST_CASE("QuantumDevice object test [lightning.qubit]", "[Driver]", LightningSimulator)
{
    std::unique_ptr<TestType> sim = std::make_unique<TestType>();

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

    std::vector<std::complex<double>> out_state = sim->State();

    CHECK(out_state[0].real() == Approx(1.0).epsilon(1e-5));
    CHECK(out_state[0].imag() == Approx(0.0).epsilon(1e-5));

    std::complex<double> sum{0, 0};
    for (size_t i = 1; i < out_state.size(); i++) {
        sum += out_state[i];
    }

    CHECK(sum.real() == Approx(0.0).epsilon(1e-5));
    CHECK(sum.imag() == Approx(0.0).epsilon(1e-5));

    for (size_t i = 0; i < n; i++) {
        sim->ReleaseQubit(static_cast<QubitIdType>(i));
        // 0, 1, 2, ..., 9
    }

    for (size_t i = 10; i < n + 10; i++) {
        CHECK(static_cast<QubitIdType>(i) == sim->AllocateQubit());
        // 10, 11, ..., 19
    }

    for (size_t i = 10; i < n + 10; i++) {
        sim->ReleaseQubit(static_cast<QubitIdType>(i));
        // 10, 11, ..., 19
    }

    for (size_t i = 20; i < n + 20; i++) {
        CHECK(static_cast<QubitIdType>(i) == sim->AllocateQubit());
        // 20, 21, ..., 29
    }
}
