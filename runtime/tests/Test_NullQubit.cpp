
// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ExecutionContext.hpp"
#include "NullQubit.hpp"
#include "QuantumDevice.hpp"
#include "QubitManager.hpp"
#include "RuntimeCAPI.h"

#include "TestUtils.hpp"

using namespace Catalyst::Runtime;
using namespace Catalyst::Runtime::Devices;

TEST_CASE("Test success of loading a device", "[NullQubit]")
{
    std::unique_ptr<ExecutionContext> driver = std::make_unique<ExecutionContext>();
    CHECK(loadDevice("NullQubit", "librtd_null_qubit" + get_dylib_ext()));
}

TEST_CASE("Test __catalyst__rt__device_init registering device=null.qubit", "[NullQubit]")
{
    __catalyst__rt__initialize(nullptr);

    char rtd_name[11] = "null.qubit";
    __catalyst__rt__device_init((int8_t *)rtd_name, nullptr, nullptr, 0);

    __catalyst__rt__device_release();

    __catalyst__rt__finalize();
}

TEST_CASE("Test NullQubit qubit allocation is successful.", "[NullQubit]")
{
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();
    sim->AllocateQubit();
}

TEST_CASE("Test a NullQubit circuit with num_qubits=2 ", "[NullQubit]")
{
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();

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

    auto &&[num_ops, num_obs, num_params, op_names, obs_keys] = sim->CacheManagerInfo();
    CHECK((num_ops == 0 && num_obs == 0));
    CHECK(num_params == 0);
    CHECK(op_names.empty());
    CHECK(obs_keys.empty());
}

TEST_CASE("Test a NullQubit circuit with num_qubits=4", "[NullQubit]")
{
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();

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

    auto &&[num_ops, num_obs, num_params, op_names, obs_keys] = sim->CacheManagerInfo();
    CHECK((num_ops == 0 && num_obs == 0));
    CHECK(num_params == 0);
    CHECK(op_names.empty());
    CHECK(obs_keys.empty());
}

TEST_CASE("Test a NullQubit circuit with num_qubits=4 and observables", "[NullQubit]")
{
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();

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

    ObsIdType pz = sim->Observable(ObsId::PauliZ, {}, {Qs[0]});

    ObsIdType px = sim->Observable(ObsId::PauliX, {}, {Qs[1]});
    ObsIdType h = sim->Observable(ObsId::Hadamard, {}, {Qs[0]});

    sim->Var(h);
    sim->Var(px);
    sim->Expval(pz);

    auto &&[num_ops, num_obs, num_params, op_names, obs_keys] = sim->CacheManagerInfo();
    CHECK(num_ops == 0);
    CHECK(num_obs == 0);
    CHECK(num_params == 0);
    CHECK(op_names.empty());
    CHECK(obs_keys.empty());
}

TEST_CASE("Test __catalyst__qis__Sample with num_qubits=2 and PartialSample calling Hadamard, "
          "ControlledPhaseShift, IsingYY, and CRX quantum operations",
          "[CoreQIS]")
{
    const auto [rtd_lib, rtd_name, rtd_kwargs] =
        std::array<std::string, 3>{"null.qubit", "null_qubit", ""};
    __catalyst__rt__initialize(nullptr);
    __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                (int8_t *)rtd_kwargs.c_str(), 1000);

    QirArray *qs = __catalyst__rt__qubit_allocate_array(2);

    QUBIT **target = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 0);
    QUBIT **ctrls = (QUBIT **)__catalyst__rt__array_get_element_ptr_1d(qs, 1);

    // qml.Hadamard(wires=0)
    __catalyst__qis__Hadamard(*target, NO_MODIFIERS);
    // qml.ControlledPhaseShift(0.6, wires=[0,1])
    __catalyst__qis__ControlledPhaseShift(0.6, *target, *ctrls, NO_MODIFIERS);
    // qml.IsingYY(0.2, wires=[0, 1])
    __catalyst__qis__IsingYY(0.2, *target, *ctrls, NO_MODIFIERS);
    // qml.CRX(0.4, wires=[1,0])
    __catalyst__qis__CRX(0.4, *target, *ctrls, NO_MODIFIERS);

    constexpr size_t n = 1;
    constexpr size_t shots = 1000;

    double *buffer = new double[shots * n];
    MemRefT_double_2d result = {buffer, buffer, 0, {shots, n}, {n, 1}};
    __catalyst__qis__Sample(&result, 1, ctrls[0]);

    CHECK(shots == 1000);

    auto obs = __catalyst__qis__NamedObs(ObsId::PauliZ, *ctrls);

    CHECK(__catalyst__qis__Variance(obs) == Approx(0.0).margin(1e-5));

    delete[] buffer;

    __catalyst__rt__qubit_release_array(qs);
    __catalyst__rt__device_release();
    __catalyst__rt__finalize();
}

TEST_CASE("NullQubit (no) Basis vector", "[NullQubit]")
{
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();

    QubitIdType q = sim->AllocateQubit();
    q = sim->AllocateQubit();
    q = sim->AllocateQubit();

    sim->ReleaseQubit(q);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(view.size() == 8);
    CHECK(view(0).real() == Approx(0.0).epsilon(1e-5));
    CHECK(view(0).imag() == Approx(0.0).epsilon(1e-5));
}

TEST_CASE("test AllocateQubits", "[NullQubit]")
{
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();

    CHECK(sim->AllocateQubits(0).size() == 0);

    auto &&q = sim->AllocateQubits(2);

    sim->ReleaseQubit(q[0]);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(state.size() == 4);
    CHECK(state[0].real() == Approx(0.0).epsilon(1e-5));
    CHECK(state[0].imag() == Approx(0.0).epsilon(1e-5));
}

TEST_CASE("test AllocateQubits generates a proper std::vector<QubitIdType>", "[NullQubit]")
{
    std::size_t num_qubits = 4;
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();

    CHECK(sim->AllocateQubits(0).size() == 0);

    auto q_vec = sim->AllocateQubits(num_qubits);

    QubitManager qm = QubitManager();
    std::vector<QubitIdType> result(num_qubits);
    std::generate_n(result.begin(), num_qubits, [&, n = 0]() mutable { return qm.Allocate(n++); });

    for (std::size_t nn = 0; nn < num_qubits; nn++) {
        CHECK(q_vec[nn] == result[nn]);
    }
}

TEST_CASE("Mix Gate test R(X,Y,Z) num_qubits=4", "[NullQubit]")
{
    std::unique_ptr<NullQubit> sim = std::make_unique<NullQubit>();

    std::vector<QubitIdType> Qs = sim->AllocateQubits(4);

    sim->NamedOperation("PauliX", {}, {Qs[0]}, false);

    sim->NamedOperation("RX", {0.123}, {Qs[1]}, false);
    sim->NamedOperation("RY", {0.456}, {Qs[2]}, false);
    sim->NamedOperation("RZ", {0.789}, {Qs[3]}, false);

    std::vector<std::complex<double>> state(1U << sim->GetNumQubits());
    DataView<std::complex<double>, 1> view(state);
    sim->State(view);

    CHECK(view.size() == 16);
    CHECK(view(0).real() == Approx(0.0).epsilon(1e-5));
    CHECK(view(0).imag() == Approx(0.0).epsilon(1e-5));
}

TEST_CASE("Test __catalyst__qis__Gradient_params Op=[Hadamard,RZ,RY,RZ,S,T,ParamShift], "
          "Obs=[X]",
          "[Gradient]")
{
    const std::vector<double> param{0.3, 0.7, 0.4};

    std::vector<int64_t> trainParams{0, 1, 2};
    size_t J = trainParams.size();
    double *buffer = new double[J];
    MemRefT_double_1d result = {buffer, buffer, 0, {J}, {1}};
    double *buffer_tp = new double[J];
    MemRefT_double_1d result_tp = {buffer_tp, buffer_tp, 0, {J}, {1}};
    int64_t *buffer_tp_memref = trainParams.data();
    MemRefT_int64_1d tp_memref = {buffer_tp_memref, buffer_tp_memref, 0, {trainParams.size()}, {1}};

    __catalyst__rt__initialize(nullptr);

    const auto [rtd_lib, rtd_name, rtd_kwargs] =
        std::array<std::string, 3>{"null.qubit", "null_qubit", ""};

    __catalyst__rt__device_init((int8_t *)rtd_lib.c_str(), (int8_t *)rtd_name.c_str(),
                                (int8_t *)rtd_kwargs.c_str(), 0);

    QUBIT *q0 = __catalyst__rt__qubit_allocate();
    QUBIT *q1 = __catalyst__rt__qubit_allocate();

    __catalyst__rt__toggle_recorder(/* activate_cm */ true);

    __catalyst__qis__Hadamard(q0, NO_MODIFIERS);
    __catalyst__qis__RZ(param[0], q0, NO_MODIFIERS);
    __catalyst__qis__RY(param[1], q0, NO_MODIFIERS);
    __catalyst__qis__RZ(param[2], q0, NO_MODIFIERS);
    __catalyst__qis__S(q0, NO_MODIFIERS);
    __catalyst__qis__T(q0, NO_MODIFIERS);

    auto obs_idx_0 = __catalyst__qis__NamedObs(ObsId::PauliX, q0);

    __catalyst__qis__Expval(obs_idx_0);

    __catalyst__qis__Gradient_params(&tp_memref, 1, &result_tp);

    __catalyst__qis__Gradient(1, &result);

    __catalyst__rt__toggle_recorder(/* activate_cm */ false);

    CHECK(result_tp.data_aligned[0] == Approx(0.0).margin(1e-5));
    CHECK(result_tp.data_aligned[1] == Approx(0.0).margin(1e-5));
    CHECK(result_tp.data_aligned[2] == Approx(0.0).margin(1e-5));

    __catalyst__rt__qubit_release(q1);
    __catalyst__rt__qubit_release(q0);
    __catalyst__rt__device_release();
    __catalyst__rt__finalize();

    delete[] buffer;
    delete[] buffer_tp;
}
