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

#include "cmath"

#include "RuntimeCAPI.h"
#include "Types.h"

#include "CacheManager.hpp"
#include "LightningUtils.hpp"
#include "QuantumDevice.hpp"

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Simulator;

TEST_CASE("NameObs test with invalid number of wires", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    REQUIRE_THROWS_WITH(sim->Observable(ObsId::PauliX, {}, {1}),
                        Catch::Contains("Invalid number of wires"));
}

TEST_CASE("NameObs test with invalid given wires for NamedObs", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    sim->AllocateQubit();

    REQUIRE_THROWS_WITH(sim->Observable(ObsId::PauliX, {}, {1}),
                        Catch::Contains("Invalid given wires"));
}

TEST_CASE("HermitianObs test with invalid number of wires", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    REQUIRE_THROWS_WITH(sim->Observable(ObsId::Hermitian, {}, {1}),
                        Catch::Contains("Invalid number of wires"));
}

TEST_CASE("HermitianObs test with invalid given wires for HermitianObs", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    sim->AllocateQubit();

    REQUIRE_THROWS_WITH(sim->Observable(ObsId::Hermitian, {}, {1}),
                        Catch::Contains("Invalid given wires"));
}

TEST_CASE("Check an unsupported observable", "[lightning]")
{
    REQUIRE_THROWS_WITH(Lightning::lookup_obs<Lightning::simulator_observable_support_size>(
                            Lightning::simulator_observable_support, static_cast<ObsId>(10)),
                        Catch::Contains("The given observable is not supported by the simulator"));
}

TEST_CASE("Measurement collapse test with 2 wires", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    auto m = sim->Measure(Qs[0]);
    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    MemRefT<std::complex<double>, 1> buffer{state.data(), state.data(), 0, {state.size()}, {1}};
    MemRefView<std::complex<double>, 1> view(&buffer, state.size());
    sim->State(view);

    // LCOV_EXCL_START
    // This is conditional over the measurement result
    if (*m) {
        CHECK(pow(std::abs(std::real(state[2])), 2) + pow(std::abs(std::imag(state[2])), 2) ==
              Approx(1.0).margin(1e-5));
    }
    else {
        CHECK(pow(std::abs(std::real(state[0])), 2) + pow(std::abs(std::imag(state[0])), 2) ==
              Approx(1.0).margin(1e-5));
    }
    // LCOV_EXCL_STOP
}

TEST_CASE("Measurement collapse concrete logical qubit difference", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    constexpr size_t n = 1;
    // The first time an array is allocated, logical and concrete qubits
    // are the same.
    std::vector<QubitIdType> Qs = sim->AllocateQubits(n);
    sim->ReleaseAllQubits();

    // Now in this the concrete qubits are shifted by n.
    Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->Measure(Qs[0]);
    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    MemRefT<std::complex<double>, 1> buffer{state.data(), state.data(), 0, {state.size()}, {1}};
    MemRefView<std::complex<double>, 1> view(&buffer, state.size());
    sim->State(view);

    // LCOV_EXCL_START
    bool is_zero = pow(std::abs(std::real(state[0])), 2) + pow(std::abs(std::imag(state[0])), 2) ==
                   Approx(1.0).margin(1e-5);
    bool is_one = pow(std::abs(std::real(state[1])), 2) + pow(std::abs(std::imag(state[1])), 2) ==
                  Approx(1.0).margin(1e-5);
    bool is_valid = is_zero ^ is_one;
    CHECK(is_valid);
    // LCOV_EXCL_STOP
}

TEST_CASE("Mid-circuit measurement naive test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    QubitIdType q;

    q = sim->AllocateQubit();

    sim->NamedOperation("PauliX", {}, {q}, false);

    auto m = sim->Measure(q);

    CHECK(*m);
}

TEST_CASE("Expval(ObsT) test with invalid key for cached observables", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    REQUIRE_THROWS_WITH(sim->Expval(0), Catch::Contains("Invalid key for cached observables"));
}

TEST_CASE("Expval(NamedObs) test with numWires=1", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    CHECK(sim->Expval(px) == Approx(1.0).margin(1e-5));
    CHECK(sim->Expval(py) == Approx(.0).margin(1e-5));
    CHECK(sim->Expval(pz) == Approx(-1.0).margin(1e-5));
}

TEST_CASE("Expval(HermitianObs) test with numWires=1", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});

    CHECK(sim->Expval(h1) == Approx(.0).margin(1e-5));
    CHECK(sim->Expval(h2) == Approx(.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(NamedObs)) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpx = sim->TensorObservable({px});
    ObsIdType tpy = sim->TensorObservable({py});
    ObsIdType tpz = sim->TensorObservable({pz});

    CHECK(sim->Expval(tpx) == Approx(1.0).margin(1e-5));
    CHECK(sim->Expval(tpy) == Approx(.0).margin(1e-5));
    CHECK(sim->Expval(tpz) == Approx(-1.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(NamedObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});
    ObsIdType tpxz = sim->TensorObservable({px, pz});

    REQUIRE_THROWS_WITH(sim->TensorObservable({px, py, pz}),
                        Catch::Contains("All wires in observables must be disjoint."));

    CHECK(sim->Expval(tpxy) == Approx(0.0).margin(1e-5));
    CHECK(sim->Expval(tpxz) == Approx(-1.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(HermitianObs))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tph1 = sim->TensorObservable({h1});
    ObsIdType tph2 = sim->TensorObservable({h2});

    CHECK(sim->Expval(tph1) == Approx(.0).margin(1e-5));
    CHECK(sim->Expval(tph2) == Approx(.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(HermitianObs[]))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(4, {1.0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tp = sim->TensorObservable({h1, h2});

    CHECK(sim->Expval(tp) == Approx(.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(Obs[]))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};

    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tp = sim->TensorObservable({px, h, pz});

    CHECK(sim->Expval(tp) == Approx(-3.0).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian(NamedObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hxyz = sim->HamiltonianObservable({0.4, 0.8, 0.2}, {px, py, pz});

    CHECK(sim->Expval(hxyz) == Approx(0.2).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian(TensorObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});
    ObsIdType tpxz = sim->TensorObservable({px, pz});
    ObsIdType hxyz = sim->HamiltonianObservable({0.2, 0.6}, {tpxy, tpxz});

    CHECK(sim->Expval(hxyz) == Approx(-.6).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian(Hermitian[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hxhz = sim->HamiltonianObservable({0.2, 0.3, 0.6}, {px, h, pz});

    CHECK(sim->Expval(hxhz) == Approx(0.5).margin(1e-5));
}

TEST_CASE("Expval(Hamiltonian({TensorProd, Hermitian}[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tp = sim->TensorObservable({px, pz});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hhtp = sim->HamiltonianObservable({0.5, 0.3}, {h, tp});

    CHECK(sim->Expval(hhtp) == Approx(1.2).margin(1e-5));
}

TEST_CASE("Var(NamedObs) test with numWires=4", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[0]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[3]});

    CHECK(sim->Var(px) == Approx(.0).margin(1e-5));
    CHECK(sim->Var(py) == Approx(1.0).margin(1e-5));
    CHECK(sim->Var(pz) == Approx(.0).margin(1e-5));
}

// TODO: Remove this after the next release of PennyLane-Lightning
#if defined(_KOKKOS)

TEST_CASE("Var(HermitianObs) test with numWires=1", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});

    CHECK(sim->Var(h1) == Approx(.0).margin(1e-5));
    CHECK(sim->Var(h2) == Approx(1.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(NamedObs)) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpx = sim->TensorObservable({px});
    ObsIdType tpy = sim->TensorObservable({py});
    ObsIdType tpz = sim->TensorObservable({pz});

    CHECK(sim->Var(tpx) == Approx(.0).margin(1e-5));
    CHECK(sim->Var(tpy) == Approx(1.0).margin(1e-5));
    CHECK(sim->Var(tpz) == Approx(.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(NamedObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});
    ObsIdType tpxz = sim->TensorObservable({px, pz});

    CHECK(sim->Var(tpxy) == Approx(1.0).margin(1e-5));
    CHECK(sim->Var(tpxz) == Approx(0.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(HermitianObs))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tph1 = sim->TensorObservable({h1});
    ObsIdType tph2 = sim->TensorObservable({h2});

    CHECK(sim->Var(tph1) == Approx(.0).margin(1e-5));
    CHECK(sim->Var(tph2) == Approx(1.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(HermitianObs[]))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(4, {1.0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = sim->Observable(ObsId::Hermitian, mat1, {Qs[1]});
    ObsIdType h2 = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tp = sim->TensorObservable({h1, h2});

    CHECK(sim->Var(tp) == Approx(2.0).margin(1e-5));
}

TEST_CASE("Var(TensorProd(Obs[]))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};

    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType tp = sim->TensorObservable({px, h, pz});

    CHECK(sim->Var(tp) == Approx(4.0).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian(NamedObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType hxyz = sim->HamiltonianObservable({0.4, 0.8, 0.2}, {px, py, pz});

    CHECK(sim->Var(hxyz) == Approx(0.64).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian(TensorObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = sim->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tpxy = sim->TensorObservable({px, py});
    ObsIdType tpxz = sim->TensorObservable({px, pz});
    ObsIdType hxyz = sim->HamiltonianObservable({0.2, 0.6}, {tpxy, tpxz});

    CHECK(sim->Var(hxyz) == Approx(0.04).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian(Hermitian[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hxhz = sim->HamiltonianObservable({0.2, 0.3, 0.6}, {px, h, pz});

    CHECK(sim->Var(hxhz) == Approx(0.36).margin(1e-5));
}

TEST_CASE("Var(Hamiltonian({TensorProd, Hermitian}[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[1]});
    ObsIdType tp = sim->TensorObservable({px, pz});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = sim->Observable(ObsId::Hermitian, mat2, {Qs[0]});
    ObsIdType hhtp = sim->HamiltonianObservable({0.5, 0.3}, {h, tp});

    CHECK(sim->Var(hhtp) == Approx(1.0).margin(1e-5));
}

#endif

TEST_CASE("State test with numWires=4", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    MemRefT<std::complex<double>, 1> buffer{state.data(), state.data(), 0, {state.size()}, {1}};
    MemRefView<std::complex<double>, 1> view(&buffer, state.size());
    sim->State(view);

    for (size_t i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 12 || i == 14) {
            CHECK(std::real(state[i]) == Approx(0.).margin(1e-5));
            CHECK(std::imag(state[i]) == Approx(0.5).margin(1e-5));
        }
        else {
            CHECK(std::real(state[i]) == Approx(0.).margin(1e-5));
            CHECK(std::imag(state[i]) == Approx(0.).margin(1e-5));
        }
    }
}

TEST_CASE("PartialProbs test with incorrect numWires and numAlloc", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    std::vector<double> probs_vec(1);
    MemRefT<double, 1> probs{probs_vec.data(), probs_vec.data(), 0, {probs_vec.size()}, {1}};
    MemRefView<double, 1> probs_view(&probs, probs_vec.size());

    REQUIRE_THROWS_WITH(sim->PartialProbs(probs_view, {Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]}),
                        Catch::Contains("Invalid number of wires"));

    REQUIRE_THROWS_WITH(
        sim->PartialProbs(probs_view, {Qs[0]}),
        Catch::Contains("Invalid size for the pre-allocated partial-probabilities"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(sim->PartialProbs(probs_view, {Qs[0]}),
                        Catch::Contains("Invalid given wires to measure"));
}

TEST_CASE("Probs and PartialProbs tests with numWires=0-4", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    std::vector<double> probs0(1);
    MemRefT<double, 1> buffer0{probs0.data(), probs0.data(), 0, {probs0.size()}, {1}};
    MemRefView<double, 1> view0(&buffer0, probs0.size());
    sim->PartialProbs(view0, std::vector<QubitIdType>{});

    std::vector<double> probs1(2);
    MemRefT<double, 1> buffer1{probs1.data(), probs1.data(), 0, {probs1.size()}, {1}};
    MemRefView<double, 1> view1(&buffer1, probs1.size());
    sim->PartialProbs(view1, std::vector<QubitIdType>{Qs[2]});

    std::vector<double> probs2(4);
    MemRefT<double, 1> buffer2{probs2.data(), probs2.data(), 0, {probs2.size()}, {1}};
    MemRefView<double, 1> view2(&buffer2, probs2.size());
    sim->PartialProbs(view2, std::vector<QubitIdType>{Qs[0], Qs[3]});

    std::vector<double> probs3(16);
    MemRefT<double, 1> buffer3{probs3.data(), probs3.data(), 0, {probs3.size()}, {1}};
    MemRefView<double, 1> view3(&buffer3, probs3.size());
    sim->PartialProbs(view3, Qs);

    std::vector<double> probs4(16);
    MemRefT<double, 1> buffer4{probs4.data(), probs4.data(), 0, {probs4.size()}, {1}};
    MemRefView<double, 1> view4(&buffer4, probs4.size());
    sim->Probs(view4);

    CHECK(probs0.size() == 1);
    CHECK(probs0[0] == Approx(1.0));
    CHECK(probs1[0] == Approx(0.5).margin(1e-5));
    CHECK(probs1[1] == Approx(0.5).margin(1e-5));
    for (size_t i = 0; i < 4; i++) {
        if (i == 0 || i == 2) {
            CHECK(probs2[i] == Approx(0.5).margin(1e-5));
        }
        else {
            CHECK(probs2[i] == Approx(0.).margin(1e-5));
        }
    }
    for (size_t i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 12 || i == 14) {
            CHECK(probs3[i] == Approx(0.25).margin(1e-5));
            CHECK(probs4[i] == Approx(0.25).margin(1e-5));
        }
        else {
            CHECK(probs3[i] == Approx(0.).margin(1e-5));
            CHECK(probs4[i] == Approx(0.).margin(1e-5));
        }
    }
}

TEST_CASE("PartialSample test with incorrect numWires and numAlloc", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    std::vector<double> samples_vec(1);
    MemRefT<double, 2> samples{
        samples_vec.data(), samples_vec.data(), 0, {samples_vec.size(), 1}, {1, 1}};
    MemRefView<double, 2> view(&samples, samples_vec.size());

    REQUIRE_THROWS_WITH(sim->PartialSample(view, {Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]}, 4),
                        Catch::Contains("Invalid number of wires"));

    REQUIRE_THROWS_WITH(sim->PartialSample(view, {Qs[0], Qs[1]}, 2),
                        Catch::Contains("Invalid size for the pre-allocated partial-samples"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(sim->PartialSample(view, {Qs[0]}, 4),
                        Catch::Contains("Invalid given wires to measure"));
}

TEST_CASE("PartialCounts test with incorrect numWires and numAlloc", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    std::vector<double> eigvals_vec(1);
    MemRefT<double, 1> eigvals{
        eigvals_vec.data(), eigvals_vec.data(), 0, {eigvals_vec.size()}, {1}};
    MemRefView<double, 1> eigvals_view(&eigvals, eigvals_vec.size());

    std::vector<int64_t> counts_vec(1);
    MemRefT<int64_t, 1> counts{counts_vec.data(), counts_vec.data(), 0, {counts_vec.size()}, {1}};
    MemRefView<int64_t, 1> counts_view(&counts, counts_vec.size());

    REQUIRE_THROWS_WITH(
        sim->PartialCounts(eigvals_view, counts_view, {Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]}, 4),
        Catch::Contains("Invalid number of wires"));

    REQUIRE_THROWS_WITH(sim->PartialCounts(eigvals_view, counts_view, {Qs[0]}, 1),
                        Catch::Contains("Invalid size for the pre-allocated partial-eigvals"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(sim->PartialCounts(eigvals_view, counts_view, {Qs[0]}, 4),
                        Catch::Contains("Invalid given wires to measure"));
}

TEST_CASE("Sample and PartialSample tests with numWires=0-4 shots=100", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    sim->NamedOperation("RX", {0.5}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    size_t shots = 100;

    std::vector<double> samples1(shots * 1);
    MemRefT<double, 2> buffer1{samples1.data(), samples1.data(), 0, {shots, 1}, {1, 1}};
    MemRefView<double, 2> view1(&buffer1, samples1.size());
    sim->PartialSample(view1, std::vector<QubitIdType>{Qs[2]}, shots);

    std::vector<double> samples2(shots * 2);
    MemRefT<double, 2> buffer2{samples2.data(), samples2.data(), 0, {shots, 2}, {1, 1}};
    MemRefView<double, 2> view2(&buffer2, samples2.size());
    sim->PartialSample(view2, std::vector<QubitIdType>{Qs[0], Qs[3]}, shots);

    std::vector<double> samples3(shots * 4);
    MemRefT<double, 2> buffer3{samples3.data(), samples3.data(), 0, {shots, 4}, {1, 1}};
    MemRefView<double, 2> view3(&buffer3, samples3.size());
    sim->PartialSample(view3, Qs, shots);

    std::vector<double> samples4(shots * 4);
    MemRefT<double, 2> buffer4{samples4.data(), samples4.data(), 0, {shots, 4}, {1, 1}};
    MemRefView<double, 2> view4(&buffer4, samples4.size());
    sim->Sample(view4, shots);

    for (size_t i = 0; i < shots * 1; i++)
        CHECK((samples1[i] == 0. || samples1[i] == 1.));
    for (size_t i = 0; i < shots * 2; i++)
        CHECK((samples2[i] == 0. || samples2[i] == 1.));
    for (size_t i = 0; i < shots * 4; i++)
        CHECK((samples3[i] == 0. || samples3[i] == 1.));
    for (size_t i = 0; i < shots * 4; i++)
        CHECK((samples4[i] == 0. || samples4[i] == 1.));
}

TEST_CASE("Counts and PartialCounts tests with numWires=0-4 shots=100", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs = sim->AllocateQubits(n);

    sim->NamedOperation("RX", {0.5}, {Qs[0]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    size_t shots = 100;

    std::vector<double> eigvals0(1);
    MemRefT<double, 1> ebuffer0{eigvals0.data(), eigvals0.data(), 0, {eigvals0.size()}, {1}};
    MemRefView<double, 1> eview0(&ebuffer0, eigvals0.size());
    std::vector<int64_t> counts0(1);
    MemRefT<int64_t, 1> cbuffer0{counts0.data(), counts0.data(), 0, {counts0.size()}, {1}};
    MemRefView<int64_t, 1> cview0(&cbuffer0, counts0.size());
    sim->PartialCounts(eview0, cview0, std::vector<QubitIdType>{}, shots);

    std::vector<double> eigvals1(2);
    MemRefT<double, 1> ebuffer1{eigvals1.data(), eigvals1.data(), 0, {eigvals1.size()}, {1}};
    MemRefView<double, 1> eview1(&ebuffer1, eigvals1.size());
    std::vector<int64_t> counts1(2);
    MemRefT<int64_t, 1> cbuffer1{counts1.data(), counts1.data(), 0, {counts1.size()}, {1}};
    MemRefView<int64_t, 1> cview1(&cbuffer1, counts1.size());
    sim->PartialCounts(eview1, cview1, std::vector<QubitIdType>{Qs[2]}, shots);

    std::vector<double> eigvals2(4);
    MemRefT<double, 1> ebuffer2{eigvals2.data(), eigvals2.data(), 0, {eigvals2.size()}, {1}};
    MemRefView<double, 1> eview2(&ebuffer2, eigvals2.size());
    std::vector<int64_t> counts2(4);
    MemRefT<int64_t, 1> cbuffer2{counts2.data(), counts2.data(), 0, {counts2.size()}, {1}};
    MemRefView<int64_t, 1> cview2(&cbuffer2, counts2.size());
    sim->PartialCounts(eview2, cview2, std::vector<QubitIdType>{Qs[0], Qs[3]}, shots);

    std::vector<double> eigvals3(16);
    MemRefT<double, 1> ebuffer3{eigvals3.data(), eigvals3.data(), 0, {eigvals3.size()}, {1}};
    MemRefView<double, 1> eview3(&ebuffer3, eigvals3.size());
    std::vector<int64_t> counts3(16);
    MemRefT<int64_t, 1> cbuffer3{counts3.data(), counts3.data(), 0, {counts3.size()}, {1}};
    MemRefView<int64_t, 1> cview3(&cbuffer3, counts3.size());
    sim->PartialCounts(eview3, cview3, Qs, shots);

    std::vector<double> eigvals4(16);
    MemRefT<double, 1> ebuffer4{eigvals4.data(), eigvals4.data(), 0, {eigvals4.size()}, {1}};
    MemRefView<double, 1> eview4(&ebuffer4, eigvals4.size());
    std::vector<int64_t> counts4(16);
    MemRefT<int64_t, 1> cbuffer4{counts4.data(), counts4.data(), 0, {counts4.size()}, {1}};
    MemRefView<int64_t, 1> cview4(&cbuffer4, counts4.size());
    sim->Counts(eview4, cview4, shots);

    CHECK(eigvals0.size() == 1);
    CHECK(eigvals0[0] == 0.0);
    CHECK(counts0.size() == 1);
    CHECK(counts0[0] == static_cast<int64_t>(shots));
    CHECK((eigvals1[0] == 0. && eigvals1[1] == 1.));
    CHECK((eigvals2[0] == 0. && eigvals2[1] == 1. && eigvals2[2] == 2. && eigvals2[3] == 3.));
    for (size_t i = 0; i < 16; i++) {
        CHECK(eigvals3[i] == static_cast<double>(i));
        CHECK(eigvals4[i] == static_cast<double>(i));
    }

    CHECK(counts1[0] + counts1[1] == static_cast<int64_t>(shots));
    CHECK(counts2[0] + counts2[1] + counts2[2] + counts2[3] == static_cast<int64_t>(shots));
    size_t sum3 = 0, sum4 = 0;
    for (size_t i = 0; i < 16; i++) {
        sum3 += counts3[i];
        sum4 += counts4[i];
    }
    CHECK(sum3 == shots);
    CHECK(sum4 == shots);
}
