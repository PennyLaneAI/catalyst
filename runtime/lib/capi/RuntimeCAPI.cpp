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

#include "Driver.hpp"
#include "MemRefUtils.hpp"

#include "RuntimeCAPI.h"

namespace Catalyst::Runtime::CAPI {

/**
 * @brief Global quantum device unique pointer.
 */
static std::unique_ptr<Driver> DRIVER = nullptr;

} // namespace Catalyst::Runtime::CAPI

extern "C" {

void *_mlir_memref_to_llvm_alloc(size_t size)
{
    void *ptr = malloc(size);
    Catalyst::Runtime::CAPI::DRIVER->get_memory_manager()->insert(ptr);
    return ptr;
}

void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size)
{
    void *ptr = aligned_alloc(alignment, size);
    Catalyst::Runtime::CAPI::DRIVER->get_memory_manager()->insert(ptr);
    return ptr;
}

void _mlir_memref_to_llvm_free(void *ptr)
{
    Catalyst::Runtime::CAPI::DRIVER->get_memory_manager()->erase(ptr);
    free(ptr);
}

void __quantum__rt__fail_cstr(const char *cstr) { RT_FAIL(cstr); }

void __quantum__rt__initialize()
{
    if (!Catalyst::Runtime::CAPI::DRIVER) {
        RT_FAIL("Initialization before defining the device");
    }

    if (Catalyst::Runtime::CAPI::DRIVER->get_device()) {
        RT_FAIL("Invalid initialization of the global device");
    }

    if (!Catalyst::Runtime::CAPI::DRIVER->init_device()) {
        // TODO: remove this after fixing the issue with propagating runtime error messages
        std::cerr << "Failed initialization of the global device, "
                  << Catalyst::Runtime::CAPI::DRIVER->get_device_name() << std::endl;

        RT_FAIL("Failed initialization of the global device");
    }

    RT_ASSERT(Catalyst::Runtime::CAPI::DRIVER->get_device() != nullptr);
    RT_ASSERT(Catalyst::Runtime::CAPI::DRIVER->get_memory_manager() != nullptr);
}

void __quantum__rt__finalize() { Catalyst::Runtime::CAPI::DRIVER.reset(nullptr); }

void __quantum__rt__device(int8_t *spec, int8_t *value)
{
    if (!Catalyst::Runtime::CAPI::DRIVER) {
        Catalyst::Runtime::CAPI::DRIVER = std::make_unique<Catalyst::Runtime::CAPI::Driver>();
    }

    if (!spec || !value) {
        // default simulator
        return;
    }

    const std::vector<std::string_view> args{reinterpret_cast<char *>(spec),
                                             reinterpret_cast<char *>(value)};

    if (args[0] == "backend") {
        Catalyst::Runtime::CAPI::DRIVER->set_device_name(args[1]);
    }
    else if (args[0] == "shots") {
        try {
            Catalyst::Runtime::CAPI::DRIVER->set_device_shots(
                static_cast<size_t>(std::stoul(std::string(args[1]))));
        }
        catch (std::exception &) {
            RT_FAIL("Invalid argument for the device specification (shots)");
        }
    }
    else {
        RT_FAIL("Invalid device specification; 'backend' and 'shots' are only supported.");
    }
}

void __quantum__rt__toggle_recorder(bool activate_cm)
{
    if (activate_cm) {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->StartTapeRecording();
    }
    else {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->StopTapeRecording();
    }
}

void __quantum__rt__print_state() { Catalyst::Runtime::CAPI::DRIVER->get_device()->PrintState(); }

QUBIT *__quantum__rt__qubit_allocate()
{
    return reinterpret_cast<QUBIT *>(
        Catalyst::Runtime::CAPI::DRIVER->get_device()->AllocateQubit());
}

QirArray *__quantum__rt__qubit_allocate_array(int64_t num_qubits)
{
    RT_ASSERT(num_qubits >= 0);

    QirArray *qubit_array = __quantum__rt__array_create_1d(sizeof(QubitIdType), num_qubits);
    const auto &&qubit_vector =
        Catalyst::Runtime::CAPI::DRIVER->get_device()->AllocateQubits(num_qubits);
    for (int64_t idx = 0; idx < num_qubits; idx++) {
        *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(qubit_array, idx)) =
            reinterpret_cast<QUBIT *>(qubit_vector[idx]);
    }
    return qubit_array;
}

void __quantum__rt__qubit_release(QUBIT *qubit)
{
    return Catalyst::Runtime::CAPI::DRIVER->get_device()->ReleaseQubit(
        reinterpret_cast<QubitIdType>(qubit));
}

void __quantum__rt__qubit_release_array(QirArray *qubit_array)
{
    // Update the reference count of qubit_array by -1
    // It will deallocates it iff the reference count becomes 0
    // The behavior is undefined if the reference count becomes < 0
    __quantum__rt__array_update_reference_count(qubit_array, -1);

    Catalyst::Runtime::CAPI::DRIVER->get_device()->ReleaseAllQubits();
}

int64_t __quantum__rt__num_qubits()
{
    return static_cast<int64_t>(Catalyst::Runtime::CAPI::DRIVER->get_device()->GetNumQubits());
}

QirString *__quantum__rt__qubit_to_string(QUBIT *qubit)
{
    return __quantum__rt__string_create(
        std::to_string(reinterpret_cast<QubitIdType>(qubit)).c_str());
}

bool __quantum__rt__result_equal(RESULT *r0, RESULT *r1) { return (r0 == r1) || (*r0 == *r1); }

RESULT *__quantum__rt__result_get_one()
{
    return Catalyst::Runtime::CAPI::DRIVER->get_device()->One();
}

RESULT *__quantum__rt__result_get_zero()
{
    return Catalyst::Runtime::CAPI::DRIVER->get_device()->Zero();
}

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

    std::vector<MemRefView<double, 1>> mem_views;
    mem_views.reserve(numResults);
    for (auto *mr : mem_ptrs) {
        mem_views.emplace_back(mr, mr->sizes[0]);
    }

    // num_observables * num_train_params
    Catalyst::Runtime::CAPI::DRIVER->get_device()->Gradient(mem_views, {});
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

    std::vector<MemRefView<double, 1>> mem_views;
    mem_views.reserve(numResults);
    for (auto *mr : mem_ptrs) {
        mem_views.emplace_back(mr, mr->sizes[0]);
    }

    // num_observables * num_train_params
    Catalyst::Runtime::CAPI::DRIVER->get_device()->Gradient(mem_views, train_params);
}

void __quantum__qis__Identity(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "Identity", {}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__PauliX(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "PauliX", {}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__PauliY(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "PauliY", {}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__PauliZ(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "PauliZ", {}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__Hadamard(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "Hadamard", {}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__S(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "S", {}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__T(QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "T", {}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__PhaseShift(double theta, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "PhaseShift", {theta}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__RX(double theta, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "RX", {theta}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__RY(double theta, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "RY", {theta}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__RZ(double theta, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "RZ", {theta}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__Rot(double phi, double theta, double omega, QUBIT *qubit)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "Rot", {phi, theta, omega}, {reinterpret_cast<QubitIdType>(qubit)},
        /* inverse = */ false);
}

void __quantum__qis__CNOT(QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "CNOT", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CY(QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "CY", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CZ(QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "CZ", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__SWAP(QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "SWAP", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__IsingXX(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "IsingXX", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__IsingYY(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "IsingYY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__IsingXY(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "IsingXY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__IsingZZ(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "IsingZZ", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__ControlledPhaseShift(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "ControlledPhaseShift", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CRX(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "CRX", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CRY(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "CRY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CRZ(double theta, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "CRZ", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CRot(double phi, double theta, double omega, QUBIT *control, QUBIT *target)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "CRot", {phi, theta, omega},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ false);
}

void __quantum__qis__CSWAP(QUBIT *control, QUBIT *aswap, QUBIT *bswap)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "CSWAP", {},
        {reinterpret_cast<QubitIdType>(control), reinterpret_cast<QubitIdType>(aswap),
         reinterpret_cast<QubitIdType>(bswap)},
        /* inverse = */ false);
}

void __quantum__qis__Toffoli(QUBIT *wire0, QUBIT *wire1, QUBIT *wire2)
{
    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation(
        "Toffoli", {},
        {reinterpret_cast<QubitIdType>(wire0), reinterpret_cast<QubitIdType>(wire1),
         reinterpret_cast<QubitIdType>(wire2)},
        /* inverse = */ false);
}

void __quantum__qis__MultiRZ(double theta, int64_t numQubits, ...)
{
    RT_ASSERT(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    Catalyst::Runtime::CAPI::DRIVER->get_device()->NamedOperation("MultiRZ", {theta}, wires,
                                                                  /* inverse = */ false);
}

void __quantum__qis__QubitUnitary(MemRefT_CplxT_double_2d *matrix, int64_t numQubits,
                                  /*qubits*/...)
{
    RT_ASSERT(numQubits >= 0);

    if (matrix == nullptr) {
        RT_FAIL("The QubitUnitary matrix must be initialized");
    }

    if (numQubits > __quantum__rt__num_qubits()) {
        RT_FAIL("Invalid number of wires");
    }

    const size_t num_rows = matrix->sizes[0];
    const size_t num_col = matrix->sizes[1];
    const size_t expected_size = std::pow(2, numQubits);

    if (num_rows != expected_size || num_col != expected_size) {
        RT_FAIL("Invalid given QubitUnitary matrix; "
                "The size of the matrix must be pow(2, numWires) * pow(2, numWires).");
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

    return Catalyst::Runtime::CAPI::DRIVER->get_device()->MatrixOperation(coeffs, wires,
                                                                          /*inverse*/ false);
}

ObsIdType __quantum__qis__NamedObs(int64_t obsId, QUBIT *wire)
{
    return Catalyst::Runtime::CAPI::DRIVER->get_device()->Observable(
        static_cast<ObsId>(obsId), {}, {reinterpret_cast<QubitIdType>(wire)});
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

    return Catalyst::Runtime::CAPI::DRIVER->get_device()->Observable(ObsId::Hermitian, coeffs,
                                                                     wires);
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

    return Catalyst::Runtime::CAPI::DRIVER->get_device()->TensorObservable(obsKeys);
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
    return Catalyst::Runtime::CAPI::DRIVER->get_device()->HamiltonianObservable(coeffs_vec,
                                                                                obsKeys);
}

RESULT *__quantum__qis__Measure(QUBIT *wire)
{
    return Catalyst::Runtime::CAPI::DRIVER->get_device()->Measure(
        reinterpret_cast<QubitIdType>(wire));
}

double __quantum__qis__Expval(ObsIdType obsKey)
{
    return Catalyst::Runtime::CAPI::DRIVER->get_device()->Expval(obsKey);
}

double __quantum__qis__Variance(ObsIdType obsKey)
{
    return Catalyst::Runtime::CAPI::DRIVER->get_device()->Var(obsKey);
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

    MemRefView<std::complex<double>, 1> view(result_p, result->sizes[0]);

    if (wires.empty()) {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->State(view);
    }
    else {
        RT_FAIL("Partial State-Vector not supported yet");
        // Catalyst::Runtime::CAPI::DRIVER->get_device()->PartialState(stateVec,
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

    MemRefView<double, 1> view(result_p, result->sizes[0]);

    if (wires.empty()) {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->Probs(view);
    }
    else {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->PartialProbs(view, wires);
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

    MemRefView<double, 2> view(result_p, result->sizes[0] * result->sizes[1]);

    if (wires.empty()) {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->Sample(view, shots);
    }
    else {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->PartialSample(view, wires, shots);
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

    MemRefView<double, 1> eigvals_view(result_eigvals_p, result_eigvals_p->sizes[0]);
    MemRefView<int64_t, 1> counts_view(result_counts_p, result_counts_p->sizes[0]);

    if (wires.empty()) {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->Counts(eigvals_view, counts_view, shots);
    }
    else {
        Catalyst::Runtime::CAPI::DRIVER->get_device()->PartialCounts(eigvals_view, counts_view,
                                                                     wires, shots);
    }
}
}
