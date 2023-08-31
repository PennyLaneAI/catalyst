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

#include <cstdarg>
#include <cstdlib>
#include <ctime>

#include <bitset>
#include <stdexcept>

#include <iostream>
#include <memory>
#include <ostream>

#include "Exception.hpp"
#include "QuantumDevice.hpp"

#include "ExecutionContext.hpp"
#include "MemRefUtils.hpp"

#include "RuntimeCAPI.h"

namespace Catalyst::Runtime {

/**
 * @brief Global quantum device unique pointer.
 */
thread_local static std::unique_ptr<ExecutionContext> CTX = nullptr;

} // namespace Catalyst::Runtime

extern "C" {

void *_mlir_memref_to_llvm_alloc(size_t size)
{
    void *ptr = malloc(size);
    Catalyst::Runtime::CTX->getMemoryManager()->insert(ptr);
    return ptr;
}

void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size)
{
    void *ptr = aligned_alloc(alignment, size);
    Catalyst::Runtime::CTX->getMemoryManager()->insert(ptr);
    return ptr;
}

bool _mlir_memory_transfer(void *ptr)
{
    if (!Catalyst::Runtime::CTX->getMemoryManager()->contains(ptr)) {
        return false;
    }
    Catalyst::Runtime::CTX->getMemoryManager()->erase(ptr);
    return true;
}

void _mlir_memref_to_llvm_free(void *ptr)
{
    Catalyst::Runtime::CTX->getMemoryManager()->erase(ptr);
    free(ptr);
}

void __quantum__rt__fail_cstr(const char *cstr) { RT_FAIL(cstr); }

void __quantum__rt__initialize()
{
    Catalyst::Runtime::CTX = std::make_unique<Catalyst::Runtime::ExecutionContext>();
}

void __quantum__rt__finalize() { Catalyst::Runtime::CTX.reset(nullptr); }

void __quantum__rt__device(int8_t *spec, int8_t *value)
{
    RT_FAIL_IF((!spec || !value), "Invalid device specification");
    RT_FAIL_IF(!Catalyst::Runtime::CTX, "Invalid use of the global driver before initialization");

    const std::vector<std::string_view> args{reinterpret_cast<char *>(spec),
                                             reinterpret_cast<char *>(value)};
    if (args[0] == "kwargs") {
        Catalyst::Runtime::CTX->setDeviceKwArgs(args[1]);
        return;
    }
    else if (args[0] == "backend") {
        RT_FAIL_IF(!Catalyst::Runtime::CTX->initDevice(args[1]),
                   "Failed initialization of the backend device");
        return;
    }

    RT_FAIL("Invalid device specification; Supported keys: ['kwargs', 'backend']");
}

void __quantum__rt__print_state() { Catalyst::Runtime::CTX->getDevice()->PrintState(); }

void __quantum__rt__toggle_recorder(bool status)
{
    Catalyst::Runtime::CTX->setDeviceRecorder(status);
    if (!Catalyst::Runtime::CTX->getDevice()) {
        return;
    }

    if (status) {
        Catalyst::Runtime::CTX->getDevice()->StartTapeRecording();
    }
    else {
        Catalyst::Runtime::CTX->getDevice()->StopTapeRecording();
    }
}

QUBIT *__quantum__rt__qubit_allocate()
{
    RT_ASSERT(Catalyst::Runtime::CTX->getDevice() != nullptr);
    RT_ASSERT(Catalyst::Runtime::CTX->getMemoryManager() != nullptr);

    return reinterpret_cast<QUBIT *>(Catalyst::Runtime::CTX->getDevice()->AllocateQubit());
}

QirArray *__quantum__rt__qubit_allocate_array(int64_t num_qubits)
{
    RT_ASSERT(Catalyst::Runtime::CTX->getDevice() != nullptr);
    RT_ASSERT(Catalyst::Runtime::CTX->getMemoryManager() != nullptr);
    RT_ASSERT(num_qubits >= 0);

    QirArray *qubit_array = __quantum__rt__array_create_1d(sizeof(QubitIdType), num_qubits);
    const auto &&qubit_vector = Catalyst::Runtime::CTX->getDevice()->AllocateQubits(num_qubits);
    for (int64_t idx = 0; idx < num_qubits; idx++) {
        *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(qubit_array, idx)) =
            reinterpret_cast<QUBIT *>(qubit_vector[idx]);
    }
    return qubit_array;
}

void __quantum__rt__qubit_release(QUBIT *qubit)
{
    return Catalyst::Runtime::CTX->getDevice()->ReleaseQubit(reinterpret_cast<QubitIdType>(qubit));
}

void __quantum__rt__qubit_release_array(QirArray *qubit_array)
{
    // Update the reference count of qubit_array by -1
    // It will deallocates it iff the reference count becomes 0
    // The behavior is undefined if the reference count becomes < 0
    __quantum__rt__array_update_reference_count(qubit_array, -1);

    Catalyst::Runtime::CTX->getDevice()->ReleaseAllQubits();
}

int64_t __quantum__rt__num_qubits()
{
    return static_cast<int64_t>(Catalyst::Runtime::CTX->getDevice()->GetNumQubits());
}

QirString *__quantum__rt__qubit_to_string(QUBIT *qubit)
{
    return __quantum__rt__string_create(
        std::to_string(reinterpret_cast<QubitIdType>(qubit)).c_str());
}

bool __quantum__rt__result_equal(RESULT *r0, RESULT *r1) { return (r0 == r1) || (*r0 == *r1); }

RESULT *__quantum__rt__result_get_one() { return Catalyst::Runtime::CTX->getDevice()->One(); }

RESULT *__quantum__rt__result_get_zero() { return Catalyst::Runtime::CTX->getDevice()->Zero(); }

QirString *__quantum__rt__result_to_string(RESULT *result)
{
    return __quantum__rt__result_equal(result, __quantum__rt__result_get_one())
               ? __quantum__rt__string_create("true")   // one
               : __quantum__rt__string_create("false"); // zero
}

void __quantum__qis__Gradient(int64_t numResults, /* results = */...)
{
    RT_ASSERT(numResults >= 0);
    using ResultType = MemRefT<double, 1>;

    std::vector<ResultType *> mem_ptrs;
    mem_ptrs.reserve(numResults);
    va_list args;
    va_start(args, numResults);
    for (int64_t i = 0; i < numResults; i++) {
        mem_ptrs.push_back(va_arg(args, ResultType *));
    }
    va_end(args);

    std::vector<DataView<double, 1>> mem_views;
    mem_views.reserve(numResults);
    for (auto *mr : mem_ptrs) {
        mem_views.emplace_back(mr->data_aligned, mr->offset, mr->sizes, mr->strides);
    }

    // num_observables * num_train_params
    Catalyst::Runtime::CTX->getDevice()->Gradient(mem_views, {});
}

void __quantum__qis__Gradient_params([[maybe_unused]] MemRefT_int64_1d *params,
                                     [[maybe_unused]] int64_t numResults,
                                     /* results = */...)
{
    RT_ASSERT(numResults >= 0);
    using ResultType = MemRefT<double, 1>;

    if (params == nullptr || !params->sizes[0]) {
        RT_FAIL("Invalid number of trainable parameters");
    }

    const size_t tp_size = params->sizes[0];

    // create a vector of custom trainable parameters
    std::vector<size_t> train_params;
    auto *params_data = params->data_aligned;
    train_params.reserve(tp_size);
    for (size_t i = 0; i < tp_size; i++) {
        auto p = params_data[i];
        RT_FAIL_IF(p < 0, "trainable parameter cannot be a negative integer");
        train_params.push_back(p);
    }

    std::vector<ResultType *> mem_ptrs;
    mem_ptrs.reserve(numResults);
    va_list args;
    va_start(args, numResults);
    for (int64_t i = 0; i < numResults; i++) {
        mem_ptrs.push_back(va_arg(args, ResultType *));
    }
    va_end(args);

    std::vector<DataView<double, 1>> mem_views;
    mem_views.reserve(numResults);
    for (auto *mr : mem_ptrs) {
        mem_views.emplace_back(mr->data_aligned, mr->offset, mr->sizes, mr->strides);
    }

    // num_observables * num_train_params
    Catalyst::Runtime::CTX->getDevice()->Gradient(mem_views, train_params);
}

void __quantum__qis__Identity(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("Identity", {},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__PauliX(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("PauliX", {},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__PauliY(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("PauliY", {},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__PauliZ(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("PauliZ", {},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__Hadamard(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("Hadamard", {},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__S(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("S", {},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__T(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("T", {},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__PhaseShift(double theta, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("PhaseShift", {theta},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__RX(double theta, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("RX", {theta},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__RY(double theta, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("RY", {theta},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__RZ(double theta, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("RZ", {theta},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__Rot(double phi, double theta, double omega, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("Rot", {phi, theta, omega},
                                                        {reinterpret_cast<QubitIdType>(qubit)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__CNOT(QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "CNOT", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__CY(QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "CY", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__CZ(QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "CZ", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__SWAP(QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "SWAP", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__IsingXX(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "IsingXX", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__IsingYY(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "IsingYY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__IsingXY(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "IsingXY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__IsingZZ(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "IsingZZ", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__ControlledPhaseShift(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "ControlledPhaseShift", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__CRX(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "CRX", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__CRY(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "CRY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__CRZ(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "CRZ", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__CRot(double phi, double theta, double omega, QUBIT *control, QUBIT *target,
                          bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation(
        "CRot", {phi, theta, omega},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __quantum__qis__CSWAP(QUBIT *control, QUBIT *aswap, QUBIT *bswap, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("CSWAP", {},
                                                        {reinterpret_cast<QubitIdType>(control),
                                                         reinterpret_cast<QubitIdType>(aswap),
                                                         reinterpret_cast<QubitIdType>(bswap)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__Toffoli(QUBIT *wire0, QUBIT *wire1, QUBIT *wire2, bool adjoint)
{
    Catalyst::Runtime::CTX->getDevice()->NamedOperation("Toffoli", {},
                                                        {reinterpret_cast<QubitIdType>(wire0),
                                                         reinterpret_cast<QubitIdType>(wire1),
                                                         reinterpret_cast<QubitIdType>(wire2)},
                                                        /* inverse = */ adjoint);
}

void __quantum__qis__MultiRZ(double theta, bool adjoint, int64_t numQubits, ...)
{
    RT_ASSERT(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    Catalyst::Runtime::CTX->getDevice()->NamedOperation("MultiRZ", {theta}, wires,
                                                        /* inverse = */ adjoint);
}

static void _qubitUnitary_impl(MemRefT_CplxT_double_2d *matrix, int64_t numQubits,
                               std::vector<std::complex<double>> &coeffs,
                               std::vector<QubitIdType> &wires, va_list *args)
{
    const size_t num_rows = matrix->sizes[0];
    const size_t num_col = matrix->sizes[1];
    const size_t expected_size = std::pow(2, numQubits);

    if (num_rows != expected_size || num_col != expected_size) {
        RT_FAIL("Invalid given QubitUnitary matrix; "
                "The size of the matrix must be pow(2, numWires) * pow(2, numWires).");
    }

    wires.reserve(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires.push_back(va_arg(*args, QubitIdType));
    }

    const size_t matrix_size = num_rows * num_col;
    coeffs.reserve(matrix_size);
    for (size_t i = 0; i < matrix_size; i++) {
        coeffs.emplace_back(matrix->data_aligned[i].real, matrix->data_aligned[i].imag);
    }
}

void __quantum__qis__QubitUnitary(MemRefT_CplxT_double_2d *matrix, bool adjoint, int64_t numQubits,
                                  /*qubits*/...)
{
    RT_ASSERT(numQubits >= 0);

    if (matrix == nullptr) {
        RT_FAIL("The QubitUnitary matrix must be initialized");
    }

    if (numQubits > __quantum__rt__num_qubits()) {
        RT_FAIL("Invalid number of wires");
    }

    va_list args;
    std::vector<std::complex<double>> coeffs;
    std::vector<QubitIdType> wires;
    va_start(args, numQubits);
    _qubitUnitary_impl(matrix, numQubits, coeffs, wires, &args);
    va_end(args);
    return Catalyst::Runtime::CTX->getDevice()->MatrixOperation(coeffs, wires,
                                                                /*inverse = */ adjoint);
}

ObsIdType __quantum__qis__NamedObs(int64_t obsId, QUBIT *wire)
{
    return Catalyst::Runtime::CTX->getDevice()->Observable(static_cast<ObsId>(obsId), {},
                                                           {reinterpret_cast<QubitIdType>(wire)});
}

ObsIdType __quantum__qis__HermitianObs(MemRefT_CplxT_double_2d *matrix, int64_t numQubits, ...)
{
    RT_ASSERT(numQubits >= 0);

    if (matrix == nullptr) {
        RT_FAIL("The Hermitian matrix must be initialized");
    }

    const size_t num_rows = matrix->sizes[0];
    const size_t num_col = matrix->sizes[1];
    const size_t expected_size = std::pow(2, numQubits);

    if (num_rows != expected_size || num_col != expected_size) {
        RT_FAIL("Invalid given Hermitian matrix; "
                "The size of the matrix must be pow(2, numWires) * pow(2, numWires).");
    }

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    if (numQubits > __quantum__rt__num_qubits()) {
        RT_FAIL("Invalid number of wires");
    }

    const size_t matrix_size = num_rows * num_col;
    std::vector<std::complex<double>> coeffs;
    coeffs.reserve(matrix_size);
    for (size_t i = 0; i < matrix_size; i++) {
        coeffs.emplace_back(matrix->data_aligned[i].real, matrix->data_aligned[i].imag);
    }

    return Catalyst::Runtime::CTX->getDevice()->Observable(ObsId::Hermitian, coeffs, wires);
}

ObsIdType __quantum__qis__TensorObs(int64_t numObs, /*obsKeys*/...)
{
    if (numObs < 1) {
        RT_FAIL("Invalid number of observables to create TensorProdObs");
    }

    va_list args;
    va_start(args, numObs);
    std::vector<ObsIdType> obsKeys;
    obsKeys.reserve(numObs);
    for (int64_t i = 0; i < numObs; i++) {
        obsKeys.push_back(va_arg(args, ObsIdType));
    }
    va_end(args);

    return Catalyst::Runtime::CTX->getDevice()->TensorObservable(obsKeys);
}

ObsIdType __quantum__qis__HamiltonianObs(MemRefT_double_1d *coeffs, int64_t numObs,
                                         /*obsKeys*/...)
{
    RT_ASSERT(numObs >= 0);

    if (coeffs == nullptr) {
        RT_FAIL("Invalid coefficients for computing Hamiltonian; "
                "The coefficients list must be initialized.");
    }

    const size_t coeffs_size = coeffs->sizes[0];

    if (static_cast<size_t>(numObs) != coeffs_size) {
        RT_FAIL("Invalid coefficients for computing Hamiltonian; "
                "The number of coefficients and observables must be equal.");
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
    return Catalyst::Runtime::CTX->getDevice()->HamiltonianObservable(coeffs_vec, obsKeys);
}

RESULT *__quantum__qis__Measure(QUBIT *wire)
{
    return Catalyst::Runtime::CTX->getDevice()->Measure(reinterpret_cast<QubitIdType>(wire));
}

double __quantum__qis__Expval(ObsIdType obsKey)
{
    return Catalyst::Runtime::CTX->getDevice()->Expval(obsKey);
}

double __quantum__qis__Variance(ObsIdType obsKey)
{
    return Catalyst::Runtime::CTX->getDevice()->Var(obsKey);
}

void __quantum__qis__State(MemRefT_CplxT_double_1d *result, int64_t numQubits, ...)
{
    RT_ASSERT(numQubits >= 0);
    MemRefT<std::complex<double>, 1> *result_p = (MemRefT<std::complex<double>, 1> *)result;

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    DataView<std::complex<double>, 1> view(result_p->data_aligned, result_p->offset,
                                           result_p->sizes, result_p->strides);

    if (wires.empty()) {
        Catalyst::Runtime::CTX->getDevice()->State(view);
    }
    else {
        RT_FAIL("Partial State-Vector not supported yet");
        // Catalyst::Runtime::CTX->getDevice()->PartialState(stateVec,
        // numElements, wires);
    }
}

void __quantum__qis__Probs(MemRefT_double_1d *result, int64_t numQubits, ...)
{
    RT_ASSERT(numQubits >= 0);
    MemRefT<double, 1> *result_p = (MemRefT<double, 1> *)result;

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    DataView<double, 1> view(result_p->data_aligned, result_p->offset, result_p->sizes,
                             result_p->strides);

    if (wires.empty()) {
        Catalyst::Runtime::CTX->getDevice()->Probs(view);
    }
    else {
        Catalyst::Runtime::CTX->getDevice()->PartialProbs(view, wires);
    }
}

void __quantum__qis__Sample(MemRefT_double_2d *result, int64_t shots, int64_t numQubits, ...)
{
    RT_ASSERT(shots >= 0);
    RT_ASSERT(numQubits >= 0);
    MemRefT<double, 2> *result_p = (MemRefT<double, 2> *)result;

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    DataView<double, 2> view(result_p->data_aligned, result_p->offset, result_p->sizes,
                             result_p->strides);

    if (wires.empty()) {
        Catalyst::Runtime::CTX->getDevice()->Sample(view, shots);
    }
    else {
        Catalyst::Runtime::CTX->getDevice()->PartialSample(view, wires, shots);
    }
}

void __quantum__qis__Counts(PairT_MemRefT_double_int64_1d *result, int64_t shots, int64_t numQubits,
                            ...)
{
    RT_ASSERT(shots >= 0);
    RT_ASSERT(numQubits >= 0);
    MemRefT<double, 1> *result_eigvals_p = (MemRefT<double, 1> *)&result->first;
    MemRefT<int64_t, 1> *result_counts_p = (MemRefT<int64_t, 1> *)&result->second;

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    DataView<double, 1> eigvals_view(result_eigvals_p->data_aligned, result_eigvals_p->offset,
                                     result_eigvals_p->sizes, result_eigvals_p->strides);
    DataView<int64_t, 1> counts_view(result_counts_p->data_aligned, result_counts_p->offset,
                                     result_counts_p->sizes, result_counts_p->strides);

    if (wires.empty()) {
        Catalyst::Runtime::CTX->getDevice()->Counts(eigvals_view, counts_view, shots);
    }
    else {
        Catalyst::Runtime::CTX->getDevice()->PartialCounts(eigvals_view, counts_view, wires, shots);
    }
}
}
