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
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    REQUIRE_THROWS_WITH(qis->Observable(ObsId::PauliX, {}, {1}),
                        Catch::Contains("Invalid number of wires"));
}

TEST_CASE("NameObs test with invalid given wires for NamedObs", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());
    sim->AllocateQubit();

    REQUIRE_THROWS_WITH(qis->Observable(ObsId::PauliX, {}, {1}),
                        Catch::Contains("Invalid given wires"));
}

#ifndef _KOKKOS
TEST_CASE("HermitianObs test with invalid number of wires", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    REQUIRE_THROWS_WITH(qis->Observable(ObsId::Hermitian, {}, {1}),
                        Catch::Contains("Invalid number of wires"));
}

TEST_CASE("HermitianObs test with invalid given wires for HermitianObs", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());
    sim->AllocateQubit();

    REQUIRE_THROWS_WITH(qis->Observable(ObsId::Hermitian, {}, {1}),
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
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    constexpr size_t n = 2;

    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("Hadamard", {}, {Qs[0]}, false);

    auto m = qis->Measure(Qs[0]);

    auto &&state = qis->State();

    // LCOV_EXCL_START
    // This is conditional over the measurement result
    if (*m) {
        REQUIRE(pow(std::abs(std::real(state[2])), 2) + pow(std::abs(std::imag(state[2])), 2) ==
                Approx(1.0).margin(1e-5));
    }
    else {
        REQUIRE(pow(std::abs(std::real(state[0])), 2) + pow(std::abs(std::imag(state[0])), 2) ==
                Approx(1.0).margin(1e-5));
    }
    // LCOV_EXCL_STOP
}

TEST_CASE("Test Measurement", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    QubitIdType q;

    q = sim->AllocateQubit();

    qis->NamedOperation("PauliX", {}, {q}, false);

    auto m = qis->Measure(q);

    REQUIRE(*m);
}
#endif

TEST_CASE("Expval(ObsT) test with invalid key for cached observables", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    REQUIRE_THROWS_WITH(qis->Expval(0), Catch::Contains("Invalid key for cached observables"));
}

TEST_CASE("Expval(NamedObs) test with numWires=1", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("PauliX", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = qis->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[1]});

    REQUIRE(qis->Expval(px) == Approx(1.0).margin(1e-5));
    REQUIRE(qis->Expval(py) == Approx(.0).margin(1e-5));
    REQUIRE(qis->Expval(pz) == Approx(-1.0).margin(1e-5));
}

TEST_CASE("Expval(HermitianObs) test with numWires=1", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = qis->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = qis->Observable(ObsId::Hermitian, mat2, {Qs[0]});

    REQUIRE(qis->Expval(h1) == Approx(.0).margin(1e-5));
    REQUIRE(qis->Expval(h2) == Approx(.0).margin(1e-5));
}

TEST_CASE("Expval(TensorProd(NamedObs)) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("PauliX", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = qis->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[1]});

#ifndef _KOKKOS
    ObsIdType tpx = qis->TensorObservable({px});
    ObsIdType tpy = qis->TensorObservable({py});
    ObsIdType tpz = qis->TensorObservable({pz});

    REQUIRE(qis->Expval(tpx) == Approx(1.0).margin(1e-5));
    REQUIRE(qis->Expval(tpy) == Approx(.0).margin(1e-5));
    REQUIRE(qis->Expval(tpz) == Approx(-1.0).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(
        qis->TensorObservable({px}),
        Catch::Contains("Tensor observable not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Expval(TensorProd(NamedObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("PauliX", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = qis->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[1]});

#ifndef _KOKKOS
    ObsIdType tpxy = qis->TensorObservable({px, py});
    ObsIdType tpxz = qis->TensorObservable({px, pz});

    REQUIRE(qis->Expval(tpxy) == Approx(0.0).margin(1e-5));
    REQUIRE(qis->Expval(tpxz) == Approx(-1.0).margin(1e-5));

    REQUIRE_THROWS_WITH(qis->TensorObservable({px, py, pz}),
                        Catch::Contains("All wires in observables must be disjoint."));
#else
    REQUIRE_THROWS_WITH(
        qis->TensorObservable({px, py}),
        Catch::Contains("Tensor observable not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Expval(TensorProd(HermitianObs))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(16, {0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = qis->Observable(ObsId::Hermitian, mat1, {Qs[0], Qs[1]});
    ObsIdType h2 = qis->Observable(ObsId::Hermitian, mat2, {Qs[0]});

#ifndef _KOKKOS

    ObsIdType tph1 = qis->TensorObservable({h1});
    ObsIdType tph2 = qis->TensorObservable({h2});

    REQUIRE(qis->Expval(tph1) == Approx(.0).margin(1e-5));
    REQUIRE(qis->Expval(tph2) == Approx(.0).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(qis->TensorObservable({h1, h2}),
                        Catch::Contains("Tensor observable not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Expval(TensorProd(HermitianObs[]))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);

    std::vector<std::complex<double>> mat1(4, {1.0, 0});
    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}};

    ObsIdType h1 = qis->Observable(ObsId::Hermitian, mat1, {Qs[1]});
    ObsIdType h2 = qis->Observable(ObsId::Hermitian, mat2, {Qs[0]});

#ifndef _KOKKOS
    ObsIdType tp = qis->TensorObservable({h1, h2});

    REQUIRE(qis->Expval(tp) == Approx(.0).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(qis->TensorObservable({h1, h2}),
                        Catch::Contains("Tensor observable not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Expval(TensorProd(Obs[]))", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("PauliX", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};

    ObsIdType h = qis->Observable(ObsId::Hermitian, mat2, {Qs[0]});
#ifndef _KOKKOS
    ObsIdType tp = qis->TensorObservable({px, h, pz});

    REQUIRE(qis->Expval(tp) == Approx(-3.0).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(qis->TensorObservable({px, h, pz}),
                        Catch::Contains("Tensor observable not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Expval(Hamiltonian(NamedObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("PauliX", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = qis->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[1]});

#ifndef _KOKKOS
    ObsIdType hxyz = qis->HamiltonianObservable({0.4, 0.8, 0.2}, {px, py, pz});

    REQUIRE(qis->Expval(hxyz) == Approx(0.2).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(qis->HamiltonianObservable({0.4, 0.8, 0.2}, {px, py, pz}),
                        Catch::Contains("Hamiltonian observable not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Expval(Hamiltonian(TensorObs[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("PauliX", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = qis->Observable(ObsId::PauliY, {}, {Qs[1]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[1]});

#ifndef _KOKKOS
    ObsIdType tpxy = qis->TensorObservable({px, py});
    ObsIdType tpxz = qis->TensorObservable({px, pz});

    ObsIdType hxyz = qis->HamiltonianObservable({0.2, 0.6}, {tpxy, tpxz});

    REQUIRE(qis->Expval(hxyz) == Approx(-.6).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(
        qis->TensorObservable({px, py}),
        Catch::Contains("Tensor observable not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Expval(Hamiltonian(Hermitian[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("PauliX", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[1]});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};

    ObsIdType h = qis->Observable(ObsId::Hermitian, mat2, {Qs[0]});
#ifndef _KOKKOS
    ObsIdType hxhz = qis->HamiltonianObservable({0.2, 0.3, 0.6}, {px, h, pz});

    REQUIRE(qis->Expval(hxhz) == Approx(0.5).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(qis->HamiltonianObservable({0.2, 0.3, 0.6}, {px, h, pz}),
                        Catch::Contains("Hamiltonian observable not implemented in "
                                        "PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Expval(Hamiltonian({TensorProd, Hermitian}[])) test", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("PauliX", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[1]});

#ifndef _KOKKOS
    ObsIdType tp = qis->TensorObservable({px, pz});

    std::vector<std::complex<double>> mat2{{1.0, 0.0}, {2.0, 0.0}, {-1.0, 0.0}, {3.0, 0.0}};
    ObsIdType h = qis->Observable(ObsId::Hermitian, mat2, {Qs[0]});

    ObsIdType hhtp = qis->HamiltonianObservable({0.5, 0.3}, {h, tp});

    REQUIRE(qis->Expval(hhtp) == Approx(1.2).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(
        qis->TensorObservable({px, pz}),
        Catch::Contains("Tensor observable not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("Var test with numWires=4", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = qis->Observable(ObsId::PauliX, {}, {Qs[2]});
    ObsIdType py = qis->Observable(ObsId::PauliY, {}, {Qs[0]});
    ObsIdType pz = qis->Observable(ObsId::PauliZ, {}, {Qs[3]});

#ifndef _KOKKOS
    REQUIRE(qis->Var(px) == Approx(.0).margin(1e-5));
    REQUIRE(qis->Var(py) == Approx(1.0).margin(1e-5));
    REQUIRE(qis->Var(pz) == Approx(.0).margin(1e-5));
#else
    REQUIRE_THROWS_WITH(qis->Var(px),
                        Catch::Contains("Variance not implemented in PennyLane-Lightning-Kokkos"));
#endif
}

TEST_CASE("State test with numWires=4", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    auto &&state = qis->State();

    for (size_t i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 12 || i == 14) {
            REQUIRE(std::real(state[i]) == Approx(0.).margin(1e-5));
            REQUIRE(std::imag(state[i]) == Approx(0.5).margin(1e-5));
        }
        else {
            REQUIRE(std::real(state[i]) == Approx(0.).margin(1e-5));
            REQUIRE(std::imag(state[i]) == Approx(0.).margin(1e-5));
        }
    }
}

TEST_CASE("PartialProbs test with incorrect numWires and numAlloc", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    REQUIRE_THROWS_WITH(qis->PartialProbs({Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]}),
                        Catch::Contains("Invalid number of wires"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(qis->PartialProbs({Qs[0]}),
                        Catch::Contains("Invalid given wires to measure"));
}

TEST_CASE("Probs and PartialProbs tests with numWires=0-4", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("Hadamard", {}, {Qs[0]}, false);
    qis->NamedOperation("PauliY", {}, {Qs[1]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    qis->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    auto &&probs0 = qis->PartialProbs(std::vector<QubitIdType>{});
    auto &&probs1 = qis->PartialProbs(std::vector<QubitIdType>{Qs[2]});
    auto &&probs2 = qis->PartialProbs(std::vector<QubitIdType>{Qs[0], Qs[3]});
    auto &&probs3 = qis->PartialProbs(Qs);
    auto &&probs4 = qis->Probs();

    REQUIRE(probs0.size() == 1);
    REQUIRE(probs0[0] == Approx(1.0));
    REQUIRE(probs1[0] == Approx(0.5).margin(1e-5));
    REQUIRE(probs1[1] == Approx(0.5).margin(1e-5));
    for (size_t i = 0; i < 4; i++) {
        if (i == 0 || i == 2) {
            REQUIRE(probs2[i] == Approx(0.5).margin(1e-5));
        }
        else {
            REQUIRE(probs2[i] == Approx(0.).margin(1e-5));
        }
    }
    for (size_t i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 12 || i == 14) {
            REQUIRE(probs3[i] == Approx(0.25).margin(1e-5));
            REQUIRE(probs4[i] == Approx(0.25).margin(1e-5));
        }
        else {
            REQUIRE(probs3[i] == Approx(0.).margin(1e-5));
            REQUIRE(probs4[i] == Approx(0.).margin(1e-5));
        }
    }
}

TEST_CASE("PartialSample test with incorrect numWires and numAlloc", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    REQUIRE_THROWS_WITH(qis->PartialSample({Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]}, 4),
                        Catch::Contains("Invalid number of wires"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(qis->PartialSample({Qs[0]}, 4),
                        Catch::Contains("Invalid given wires to measure"));
}

TEST_CASE("PartialCounts test with incorrect numWires and numAlloc", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    REQUIRE_THROWS_WITH(qis->PartialCounts({Qs[0], Qs[1], Qs[2], Qs[3], Qs[0]}, 4),
                        Catch::Contains("Invalid number of wires"));

    sim->ReleaseQubit(Qs[0]);

    REQUIRE_THROWS_WITH(qis->PartialCounts({Qs[0]}, 4),
                        Catch::Contains("Invalid given wires to measure"));
}

#ifndef _KOKKOS
TEST_CASE("Sample and PartialSample tests with numWires=0-4 shots=100", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("RX", {0.5}, {Qs[0]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    qis->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    size_t shots = 100;

    auto &&samples0 = qis->PartialSample(std::vector<QubitIdType>{}, shots);
    auto &&samples1 = qis->PartialSample(std::vector<QubitIdType>{Qs[2]}, shots);
    auto &&samples2 = qis->PartialSample(std::vector<QubitIdType>{Qs[0], Qs[3]}, shots);
    auto &&samples3 = qis->PartialSample(Qs, shots);
    auto &&samples4 = qis->Sample(shots);

    REQUIRE(samples0.size() == 0);
    for (size_t i = 0; i < shots * 1; i++)
        REQUIRE((samples1[i] == 0. || samples1[i] == 1.));
    for (size_t i = 0; i < shots * 2; i++)
        REQUIRE((samples2[i] == 0. || samples2[i] == 1.));
    for (size_t i = 0; i < shots * 4; i++)
        REQUIRE((samples3[i] == 0. || samples3[i] == 1.));
    for (size_t i = 0; i < shots * 4; i++)
        REQUIRE((samples4[i] == 0. || samples4[i] == 1.));
}

TEST_CASE("Counts and PartialCounts tests with numWires=0-4 shots=100", "[lightning]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    QuantumDevice *qis = dynamic_cast<QuantumDevice *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs.push_back(sim->AllocateQubit());
    }

    qis->NamedOperation("RX", {0.5}, {Qs[0]}, false);
    qis->NamedOperation("Hadamard", {}, {Qs[1]}, false);
    qis->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    size_t shots = 100;

    auto &&[eigvals0, counts0] = qis->PartialCounts(std::vector<QubitIdType>{}, shots);
    auto &&[eigvals1, counts1] = qis->PartialCounts(std::vector<QubitIdType>{Qs[2]}, shots);
    auto &&[eigvals2, counts2] = qis->PartialCounts(std::vector<QubitIdType>{Qs[0], Qs[3]}, shots);
    auto &&[eigvals3, counts3] = qis->PartialCounts(Qs, shots);
    auto &&[eigvals4, counts4] = qis->Counts(shots);

    REQUIRE(eigvals0.size() == 1);
    REQUIRE(eigvals0[0] == 0.0);
    REQUIRE(counts0.size() == 1);
    REQUIRE(counts0[0] == static_cast<int64_t>(shots));
    REQUIRE((eigvals1[0] == 0. && eigvals1[1] == 1.));
    REQUIRE((eigvals2[0] == 0. && eigvals2[1] == 1. && eigvals2[2] == 2. && eigvals2[3] == 3.));
    for (size_t i = 0; i < 16; i++) {
        REQUIRE(eigvals3[i] == static_cast<double>(i));
        REQUIRE(eigvals4[i] == static_cast<double>(i));
    }

    REQUIRE(counts1[0] + counts1[1] == static_cast<int64_t>(shots));
    REQUIRE(counts2[0] + counts2[1] + counts2[2] + counts2[3] == static_cast<int64_t>(shots));
    size_t sum3 = 0, sum4 = 0;
    for (size_t i = 0; i < 16; i++) {
        sum3 += counts3[i];
        sum4 += counts4[i];
    }
    REQUIRE(sum3 == shots);
    REQUIRE(sum4 == shots);
}
#endif
