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

#include "TestHelpers.hpp"

#include "CacheManager.hpp"
#include "LightningUtils.hpp"
#include "QuantumDevice.hpp"
#include "RuntimeCAPI.h"

#if defined(_KOKKOS)
#include "LightningKokkosSimulator.hpp"
#else
#include "LightningSimulator.hpp"
#endif

#include <catch2/catch.hpp>

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Simulator;

TEST_CASE("Test the constructor with a naive example", "[CacheManager]")
{
    CacheManager cm = CacheManager();

    REQUIRE(cm.getNumOperations() == 0);
    REQUIRE(cm.getNumObservables() == 0);
}

TEST_CASE("Test addOperation with a naive example", "[CacheManager]")
{
    CacheManager cm = CacheManager();

    cm.addOperation("H", {0}, {0}, false);

    REQUIRE(cm.getNumOperations() == 1);
    REQUIRE(cm.getNumObservables() == 0);
}

TEST_CASE("Test addOperations with a naive example", "[CacheManager]")
{
    CacheManager cm = CacheManager();

    cm.addOperation("H", {0}, {0}, false);
    cm.addOperation("H", {0}, {0}, false);
    cm.addOperation("H", {0}, {0}, false);
    cm.addOperation("H", {0}, {0}, false);

    REQUIRE(cm.getNumOperations() == 4);
    REQUIRE(cm.getNumObservables() == 0);
}

TEST_CASE("Test a LightningSimulator circuit with num_qubits=2 ", "[CacheManager]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();

#if defined(_KOKKOS)
    LightningKokkosSimulator *qis = dynamic_cast<LightningKokkosSimulator *>(sim.get());
#else
    LightningSimulator *qis = dynamic_cast<LightningSimulator *>(sim.get());
#endif

    // state-vector with #qubits = n
    constexpr size_t n = 2;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->StartTapeRecording();
    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("CNOT", {}, {Qs[0], Qs[1]}, false);

    auto &&[num_ops, num_obs, num_params, op_names, _] = qis->CacheManagerInfo();
    REQUIRE((num_ops == 2 && num_obs == 0));
    REQUIRE(num_params == 0);
    REQUIRE(op_names[0] == "PauliX");
    REQUIRE(op_names[1] == "CNOT");
}

TEST_CASE("Test a LightningSimulator circuit with num_qubits=4", "[CacheManager]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
#if defined(_KOKKOS)
    LightningKokkosSimulator *qis = dynamic_cast<LightningKokkosSimulator *>(sim.get());
#else
    LightningSimulator *qis = dynamic_cast<LightningSimulator *>(sim.get());
#endif

    // state-vector with #qubit = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);

    sim->StartTapeRecording();
    Qs[0] = sim->AllocateQubit();
    sim->NamedOperation("Hadamard", {}, {Qs[0]}, false);

    Qs[1] = sim->AllocateQubit();
    sim->NamedOperation("CRX", {0.123}, {Qs[0], Qs[1]}, false);

    Qs[2] = sim->AllocateQubit();
    sim->NamedOperation("CRY", {0.456}, {Qs[0], Qs[2]}, false);

    Qs[3] = sim->AllocateQubit();
    sim->NamedOperation("CRZ", {0.789}, {Qs[0], Qs[3]}, false);
    sim->StopTapeRecording();

    auto &&[num_ops, num_obs, num_params, op_names, _] = qis->CacheManagerInfo();
    REQUIRE((num_ops == 4 && num_obs == 0));
    REQUIRE(num_params == 3);
    REQUIRE(op_names[0] == "Hadamard");
    REQUIRE(op_names[1] == "CRX");
    REQUIRE(op_names[2] == "CRY");
    REQUIRE(op_names[3] == "CRZ");
}

TEST_CASE("Test __quantum__qis__ circuit with observables", "[CacheManager]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.70357419).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[1].real == Approx(0.0).margin(1e-5) &&
             state[1].imag == Approx(-0.0705929).margin(1e-5)));
    REQUIRE((state[2].real == Approx(0.70357419).margin(1e-5) &&
             state[2].imag == Approx(0).margin(1e-5)));
    REQUIRE((state[3].real == Approx(0.0).margin(1e-5) &&
             state[3].imag == Approx(-0.0705929).margin(1e-5)));

    // qml.var(qml.PauliZ(wires=1))
    QUBIT **qubit = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);
    auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, *qubit);

    REQUIRE(__quantum__qis__Expval(obs) == Approx(0.9800665778).margin(1e-5));

    __quantum__rt__finalize();
}

TEST_CASE("Test __quantum__qis__ circuit with observables using deactiveCacheManager",
          "[CacheManager]")
{
    // initialize the simulator
    __quantum__rt__initialize();

    QUBIT *target = __quantum__rt__qubit_allocate();              // id = 0
    QirArray *ctrls_arr = __quantum__rt__qubit_allocate_array(1); // id = 1

    QUBIT **ctrls = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);

    __quantum__rt__toggle_recorder(/* activate_cm */ true);

    // qml.Hadamard(wires=0)
    __quantum__qis__Hadamard(target);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __quantum__qis__ControlledPhaseShift(0.6, target, *ctrls);
    // qml.IsingYY(0.2, wires=[0, 1])
    __quantum__qis__IsingYY(0.2, target, *ctrls);
    // qml.CRX(0.4, wires=[1,0])
    __quantum__qis__CRX(0.4, target, *ctrls);

    MemRefT_CplxT_double_1d *result = new MemRefT_CplxT_double_1d;
    __quantum__qis__State(result, 0);
    CplxT_double *state = result->data_allocated;

    REQUIRE((state[0].real == Approx(0.70357419).margin(1e-5) &&
             state[0].imag == Approx(0.0).margin(1e-5)));
    REQUIRE((state[1].real == Approx(0.0).margin(1e-5) &&
             state[1].imag == Approx(-0.0705929).margin(1e-5)));
    REQUIRE((state[2].real == Approx(0.70357419).margin(1e-5) &&
             state[2].imag == Approx(0).margin(1e-5)));
    REQUIRE((state[3].real == Approx(0.0).margin(1e-5) &&
             state[3].imag == Approx(-0.0705929).margin(1e-5)));

    // qml.var(qml.PauliZ(wires=1))
    QUBIT **qubit = (QUBIT **)__quantum__rt__array_get_element_ptr_1d(ctrls_arr, 0);
    auto obs = __quantum__qis__NamedObs(ObsId::PauliZ, *qubit);

    REQUIRE(__quantum__qis__Expval(obs) == Approx(0.9800665778).margin(1e-5));

    __quantum__rt__toggle_recorder(/* activate_cm */ false);

    __quantum__rt__finalize();
}

#ifndef _KOKKOS
TEST_CASE("Test a LightningSimulator circuit with num_qubits=4 and observables", "[CacheManager]")
{
    std::unique_ptr<QuantumDevice> sim = CreateQuantumDevice();
    LightningSimulator *qis = dynamic_cast<LightningSimulator *>(sim.get());

    // state-vector with #qubits = n
    constexpr size_t n = 4;
    std::vector<QubitIdType> Qs;
    Qs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Qs[i] = sim->AllocateQubit();
    }

    sim->StartTapeRecording();
    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);
    sim->NamedOperation("PauliY", {}, {Qs[1]}, false);
    sim->NamedOperation("Hadamard", {}, {Qs[2]}, false);
    sim->NamedOperation("PauliZ", {}, {Qs[3]}, false);

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[1]});
    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[0]});

    sim->StopTapeRecording();

    ObsIdType h = sim->Observable(ObsId::Hadamard, {}, {Qs[0]});

    sim->Var(h);
    sim->Var(px);
    sim->Expval(pz);

    auto &&[num_ops, num_obs, num_params, op_names, obs_keys] = qis->CacheManagerInfo();
    REQUIRE(num_ops == 4);
    REQUIRE(num_params == 0);
    REQUIRE(op_names[0] == "PauliX");
    REQUIRE(op_names[1] == "PauliY");
    REQUIRE(op_names[2] == "Hadamard");
    REQUIRE(op_names[3] == "PauliZ");

    REQUIRE(num_obs == 2);
    REQUIRE(obs_keys[0] == px);
    REQUIRE(obs_keys[1] == pz);
}
#endif
