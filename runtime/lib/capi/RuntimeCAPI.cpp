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

#include <cassert>
#include <cstdarg>
#include <cstdlib>
#include <ctime>

#include <bitset>
#include <stdexcept>

#include <iostream>
#include <ostream>
#include <string>

#include "QuantumDevice.hpp"

#include "RuntimeCAPI.h"

namespace Catalyst::Runtime::CAPI {

/**
 * @brief Global quantum device unique pointer.
 */
static std::unique_ptr<Catalyst::Runtime::QuantumDevice> GLOBAL_DEVICE_PTR = nullptr;

/**
 * @brief Get the global device unique pointer.
 *
 * @return std::unique_ptr<Catalyst::Runtime::QuantumDevice> &
 */
static auto get_device() -> std::unique_ptr<Catalyst::Runtime::QuantumDevice> &
{
    return GLOBAL_DEVICE_PTR;
}

} // namespace Catalyst::Runtime::CAPI

extern "C" {

void __quantum__rt__fail_cstr(const char *cstr) { throw std::runtime_error(cstr); }

void __quantum__rt__initialize()
{
    if (Catalyst::Runtime::CAPI::get_device()) {
        __quantum__rt__fail_cstr("Invalid initialization of the global simulator");
    }

    Catalyst::Runtime::CAPI::GLOBAL_DEVICE_PTR = Catalyst::Runtime::CreateQuantumDevice();
    assert(Catalyst::Runtime::CAPI::get_device() != nullptr);
}

void __quantum__rt__finalize()
{
    Catalyst::Runtime::CAPI::GLOBAL_DEVICE_PTR.reset(nullptr);
    assert(Catalyst::Runtime::CAPI::get_device() == nullptr);
}

void __quantum__rt__toggle_recorder(bool activate_cm)
{
    if (activate_cm) {
        Catalyst::Runtime::CAPI::get_device()->StartTapeRecording();
    }
    else {
        Catalyst::Runtime::CAPI::get_device()->StopTapeRecording();
    }
}

void __quantum__rt__print_state() { Catalyst::Runtime::CAPI::get_device()->PrintState(); }

QUBIT *__quantum__rt__qubit_allocate()
{
    return reinterpret_cast<QUBIT *>(Catalyst::Runtime::CAPI::get_device()->AllocateQubit());
}

QirArray *__quantum__rt__qubit_allocate_array(int64_t num_qubits)
{
    assert(num_qubits >= 0);

    QirArray *qubit_array = __quantum__rt__array_create_1d(sizeof(QubitIdType), num_qubits);
    const auto &&qubit_vector = Catalyst::Runtime::CAPI::get_device()->AllocateQubits(num_qubits);
    for (int64_t idx = 0; idx < num_qubits; idx++) {
        *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(qubit_array, idx)) =
            reinterpret_cast<QUBIT *>(qubit_vector[idx]);
    }
    return qubit_array;
}

void __quantum__rt__qubit_release(QUBIT *qubit)
{
    return Catalyst::Runtime::CAPI::get_device()->ReleaseQubit(
        reinterpret_cast<QubitIdType>(qubit));
}

void __quantum__rt__qubit_release_array(QirArray *qubit_array)
{
    // Update the reference count of qubit_array by -1
    // It will deallocates it iff the reference count becomes 0
    // The behavior is undefined if the reference count becomes < 0
    __quantum__rt__array_update_reference_count(qubit_array, -1);

    Catalyst::Runtime::CAPI::get_device()->ReleaseAllQubits();
}

int64_t __quantum__rt__num_qubits()
{
    return static_cast<int64_t>(Catalyst::Runtime::CAPI::get_device()->GetNumQubits());
}

QirString *__quantum__rt__qubit_to_string(QUBIT *qubit)
{
    return __quantum__rt__string_create(
        std::to_string(reinterpret_cast<QubitIdType>(qubit)).c_str());
}

bool __quantum__rt__result_equal(RESULT *r0, RESULT *r1) { return (r0 == r1) || (*r0 == *r1); }

RESULT *__quantum__rt__result_get_one() { return Catalyst::Runtime::CAPI::get_device()->One(); }

RESULT *__quantum__rt__result_get_zero() { return Catalyst::Runtime::CAPI::get_device()->Zero(); }

QirString *__quantum__rt__result_to_string(RESULT *result)
{
    return __quantum__rt__result_equal(result, __quantum__rt__result_get_one())
               ? __quantum__rt__string_create("true")   // one
               : __quantum__rt__string_create("false"); // zero
}

void __quantum__qis__Gradient(int64_t numResults, /* results = */...)
{
    assert(numResults >= 0);
    using ResultType = MemRefT_double_1d;

    // num_observables * num_train_params
    auto &&jacobian = Catalyst::Runtime::CAPI::get_device()->Gradient({});

    const size_t num_observables = jacobian.size();
    if (num_observables != static_cast<size_t>(numResults)) {
        __quantum__rt__fail_cstr("Invalid number of results; "
                                 "The number of results must be equal to the "
                                 "number of cached observables");
    }

    // for zero number of observables
    if (jacobian.empty()) {
        return;
    }

    const size_t num_train_params = jacobian[0].size();

    // extract variadic results of size num_observables
    va_list args;
    va_start(args, numResults);
    for (int64_t i = 0; i < numResults; i++) {
        auto *mrp = va_arg(args, ResultType *);
        assert(mrp && "the result type cannot be a null pointer");

        double *jac_data = (double *)mrp->data_aligned;
        for (size_t j = 0; j < num_train_params; j++) {
            jac_data[j] = jacobian[i][j];
        }
    }
    va_end(args);
}

void __quantum__qis__Gradient_params(MemRefT_int64_1d *params, int64_t numResults,
                                     /* results = */...)
{
    assert(numResults >= 0);
    using ResultType = MemRefT_double_1d;

    if (params == nullptr || !params->sizes[0]) {
        __quantum__rt__fail_cstr("Invalid number of trainable parameters");
    }

    const size_t tp_size = params->sizes[0];

    // create a vector of custom trainable parameters
    std::vector<size_t> train_params;
    auto *params_data = params->data_aligned;
    train_params.reserve(tp_size);
    for (size_t i = 0; i < tp_size; i++) {
        auto p = params_data[i];
        assert(p >= 0 && "trainable parameter cannot be a negative integer");
        train_params.push_back(p);
    }

    // num_observables * num_train_params
    auto &&jacobian = Catalyst::Runtime::CAPI::get_device()->Gradient(train_params);

    const size_t num_observables = jacobian.size();
    if (num_observables != static_cast<size_t>(numResults)) {
        __quantum__rt__fail_cstr("Invalid number of results; "
                                 "The number of results must be equal to the "
                                 "number of cached observables");
    }

    // for zero number of observables
    if (jacobian.empty()) {
        return;
    }

    const size_t num_train_params = jacobian[0].size();

    // extract variadic results of size num_observables
    va_list args;
    va_start(args, numResults);
    for (int64_t i = 0; i < numResults; i++) {
        auto *mrp = va_arg(args, ResultType *);
        assert(mrp && "the result type cannot be a null pointer");

        double *jac_data =
            (double *)aligned_alloc(sizeof(double), num_train_params * sizeof(double));
        for (size_t j = 0; j < num_train_params; j++) {
            jac_data[j] = jacobian[i][j];
        }

        mrp->data_aligned = jac_data;
        mrp->data_aligned = jac_data;
        mrp->offset = 0;
        mrp->sizes[0] = num_train_params;
        mrp->strides[0] = 1;
    }
    va_end(args);
}

void __quantum__qis__Identity(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("Identity", {},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__PauliX(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("PauliX", {},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__PauliY(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("PauliY", {},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__PauliZ(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("PauliZ", {},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__Hadamard(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("Hadamard", {},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__S(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("S", {},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__T(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("T", {},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__PhaseShift(double theta, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("PhaseShift", {theta},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__RX(double theta, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("RX", {theta},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__RY(double theta, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("RY", {theta},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__RZ(double theta, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("RZ", {theta},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__Rot(double phi, double theta, double omega, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("Rot", {phi, theta, omega},
                                                          {reinterpret_cast<QubitIdType>(qubit)},
                                                          /* inverse = */ false);
}

void __quantum__qis__CNOT(QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "CNOT", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CY(QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "CY", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CZ(QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "CZ", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__SWAP(QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "SWAP", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__IsingXX(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "IsingXX", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__IsingYY(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "IsingYY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__IsingXY(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "IsingXY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__IsingZZ(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "IsingZZ", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__ControlledPhaseShift(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "ControlledPhaseShift", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CRX(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "CRX", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CRY(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "CRY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CRZ(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "CRZ", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CRot(double phi, double theta, double omega, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation(
        "CRot", {phi, theta, omega},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CSWAP(QUBIT *control, QUBIT *aswap, QUBIT *bswap)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("CSWAP", {},
                                                          {reinterpret_cast<QubitIdType>(control),
                                                           reinterpret_cast<QubitIdType>(aswap),
                                                           reinterpret_cast<QubitIdType>(bswap)},
                                                          /* inverse = */ false);
}

void __quantum__qis__Toffoli(QUBIT *wire0, QUBIT *wire1, QUBIT *wire2)
{
    Catalyst::Runtime::CAPI::get_device()->NamedOperation("Toffoli", {},
                                                          {reinterpret_cast<QubitIdType>(wire0),
                                                           reinterpret_cast<QubitIdType>(wire1),
                                                           reinterpret_cast<QubitIdType>(wire2)},
                                                          /* inverse = */ false);
}

void __quantum__qis__MultiRZ(double theta, int64_t numQubits, ...)
{
    assert(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    Catalyst::Runtime::CAPI::get_device()->NamedOperation("MultiRZ", {theta}, wires,
                                                          /* inverse = */ false);
}

void __quantum__qis__QubitUnitary(MemRefT_CplxT_double_2d *matrix, int64_t numQubits,
                                  /*qubits*/...)
{
    assert(numQubits >= 0);

    if (matrix == nullptr) {
        __quantum__rt__fail_cstr("The QubitUnitary matrix must be initialized");
    }

    if (numQubits > __quantum__rt__num_qubits()) {
        __quantum__rt__fail_cstr("Invalid number of wires");
    }

    const size_t num_rows = matrix->sizes[0];
    const size_t num_col = matrix->sizes[1];
    const size_t expected_size = std::pow(2, numQubits);

    if (num_rows != expected_size || num_col != expected_size) {
        __quantum__rt__fail_cstr(
            "Invalid given QubitUnitary matrix; "
            "The size of the matrix must be pow(2, numWires) * pow(2, numWires)");
    }

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires;
    wires.reserve(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires.push_back(va_arg(args, QubitIdType));
    }
    va_end(args);

    const size_t matrix_size = num_rows * num_col;
    std::vector<std::complex<double>> coeffs;
    coeffs.reserve(matrix_size);
    for (size_t i = 0; i < matrix_size; i++) {
        coeffs.emplace_back(matrix->data_aligned[i].real, matrix->data_aligned[i].imag);
    }

    return Catalyst::Runtime::CAPI::get_device()->MatrixOperation(coeffs, wires,
                                                                  /*inverse*/ false);
}

ObsIdType __quantum__qis__NamedObs(int64_t obsId, QUBIT *wire)
{
    return Catalyst::Runtime::CAPI::get_device()->Observable(static_cast<ObsId>(obsId), {},
                                                             {reinterpret_cast<QubitIdType>(wire)});
}

ObsIdType __quantum__qis__HermitianObs(MemRefT_CplxT_double_2d *matrix, int64_t numQubits, ...)
{
    assert(numQubits >= 0);

    if (matrix == nullptr) {
        __quantum__rt__fail_cstr("The Hermitian matrix must be initialized");
    }

    const size_t num_rows = matrix->sizes[0];
    const size_t num_col = matrix->sizes[1];
    const size_t expected_size = std::pow(2, numQubits);

    if (num_rows != expected_size || num_col != expected_size) {
        __quantum__rt__fail_cstr(
            "Invalid given Hermitian matrix; "
            "The size of the matrix must be pow(2, numWires) * pow(2, numWires)");
    }

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    if (numQubits > __quantum__rt__num_qubits()) {
        __quantum__rt__fail_cstr("Invalid number of wires");
    }

    const size_t matrix_size = num_rows * num_col;
    std::vector<std::complex<double>> coeffs;
    coeffs.reserve(matrix_size);
    for (size_t i = 0; i < matrix_size; i++) {
        coeffs.emplace_back(matrix->data_aligned[i].real, matrix->data_aligned[i].imag);
    }

    return Catalyst::Runtime::CAPI::get_device()->Observable(ObsId::Hermitian, coeffs, wires);
}

ObsIdType __quantum__qis__TensorObs(int64_t numObs, /*obsKeys*/...)
{
    if (numObs < 1) {
        __quantum__rt__fail_cstr("Invalid number of observables to create TensorProdObs");
    }

    va_list args;
    va_start(args, numObs);
    std::vector<ObsIdType> obsKeys;
    obsKeys.reserve(numObs);
    for (int64_t i = 0; i < numObs; i++) {
        obsKeys.push_back(va_arg(args, ObsIdType));
    }
    va_end(args);

    return Catalyst::Runtime::CAPI::get_device()->TensorObservable(obsKeys);
}

ObsIdType __quantum__qis__HamiltonianObs(MemRefT_double_1d *coeffs, int64_t numObs,
                                         /*obsKeys*/...)
{
    assert(numObs >= 0);

    if (coeffs == nullptr) {
        __quantum__rt__fail_cstr("Invalid coefficients for computing Hamiltonian; "
                                 "The coefficients list must be initialized");
    }

    const size_t coeffs_size = coeffs->sizes[0];

    if (static_cast<size_t>(numObs) != coeffs_size) {
        __quantum__rt__fail_cstr("Invalid coefficients for computing Hamiltonian; "
                                 "The number of coefficients and observables must be equal");
    }

    va_list args;
    va_start(args, numObs);
    std::vector<ObsIdType> obsKeys;
    obsKeys.reserve(numObs);
    for (int64_t i = 0; i < numObs; i++) {
        obsKeys.push_back(va_arg(args, ObsIdType));
    }
    va_end(args);

    std::vector<double> coeffs_vec(coeffs->data_aligned, coeffs->data_aligned + coeffs_size);
    return Catalyst::Runtime::CAPI::get_device()->HamiltonianObservable(coeffs_vec, obsKeys);
}

RESULT *__quantum__qis__Measure(QUBIT *wire)
{
    return Catalyst::Runtime::CAPI::get_device()->Measure(reinterpret_cast<QubitIdType>(wire));
}

double __quantum__qis__Expval(ObsIdType obsKey)
{
    return Catalyst::Runtime::CAPI::get_device()->Expval(obsKey);
}

double __quantum__qis__Variance(ObsIdType obsKey)
{
    return Catalyst::Runtime::CAPI::get_device()->Var(obsKey);
}

void __quantum__qis__Probs(MemRefT_double_1d *result, int64_t numQubits, ...)
{
    assert(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    if (wires.empty()) {
        numQubits = __quantum__rt__num_qubits();
    }

    std::vector<double> sv_probs;

    if (wires.empty()) {
        sv_probs = Catalyst::Runtime::CAPI::get_device()->Probs();
    }
    else {
        sv_probs = Catalyst::Runtime::CAPI::get_device()->PartialProbs(wires);
    }

    const size_t numElements = 1U << numQubits;

    double *probs = result->data_aligned;
    double *curr = probs;
    for (size_t idx = 0; idx < numElements; idx++) {
        *(curr++) = sv_probs[idx];
    }
}

void __quantum__qis__State(MemRefT_CplxT_double_1d *result, int64_t numQubits, ...)
{
    assert(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    if (wires.empty()) {
        numQubits = __quantum__rt__num_qubits();
    }

    std::vector<std::complex<double>> sv_state;

    if (wires.empty()) {
        sv_state = Catalyst::Runtime::CAPI::get_device()->State();
    }
    else {
        __quantum__rt__fail_cstr("Partial State-Vector not supported yet");
        // Catalyst::Runtime::CAPI::get_device()->PartialState(stateVec,
        // numElements, wires);
    }

    const size_t numElements = sv_state.size();
    assert(numElements == (1U << numQubits));

    for (size_t idx = 0; idx < numElements; idx++) {
        result->data_aligned[idx].real = std::real(sv_state[idx]);
        result->data_aligned[idx].imag = std::imag(sv_state[idx]);
    }
}

void __quantum__qis__Sample(MemRefT_double_2d *result, int64_t shots, int64_t numQubits, ...)
{
    assert(shots >= 0);
    assert(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    if (wires.empty()) {
        numQubits = __quantum__rt__num_qubits();
    }

    std::vector<double> sv_samples;
    if (wires.empty()) {
        sv_samples = Catalyst::Runtime::CAPI::get_device()->Sample(shots);
    }
    else {
        sv_samples = Catalyst::Runtime::CAPI::get_device()->PartialSample(wires, shots);
    }

    const size_t numElements = sv_samples.size();
    assert(numElements == static_cast<size_t>(shots * numQubits));

    double *samples = result->data_aligned;
    double *curr = samples;
    for (size_t idx = 0; idx < numElements; idx++) {
        *(curr++) = sv_samples[idx];
    }
}

void __quantum__qis__Counts(PairT_MemRefT_double_int64_1d *result, int64_t shots, int64_t numQubits,
                            ...)
{
    assert(shots >= 0);
    assert(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    if (wires.empty()) {
        numQubits = __quantum__rt__num_qubits();
    }

    std::tuple<std::vector<double>, std::vector<int64_t>> sv_counts;

    if (wires.empty()) {
        sv_counts = Catalyst::Runtime::CAPI::get_device()->Counts(shots);
    }
    else {
        sv_counts = Catalyst::Runtime::CAPI::get_device()->PartialCounts(wires, shots);
    }

    auto &&sv_eigvals = std::get<0>(sv_counts);
    auto &&sv_cts = std::get<1>(sv_counts);

    const size_t numElements = 1U << numQubits;
    assert(numElements == sv_eigvals.size());
    assert(numElements == sv_cts.size());

    // eigvals
    // TODO: memory management
    double *eigvals = (double *)result->first.data_aligned;
    double *curr = eigvals;
    for (size_t idx = 0; idx < numElements; idx++) {
        *(curr++) = sv_eigvals[idx];
    }

    // counts
    // TODO: memory management
    int64_t *counts = (int64_t *)result->second.data_aligned;
    int64_t *icurr = counts;
    for (size_t idx = 0; idx < numElements; idx++) {
        *(icurr++) = sv_cts[idx];
    }
}
}
