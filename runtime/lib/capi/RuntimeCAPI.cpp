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
#include <string_view>

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "Exception.hpp"
#include "QuantumDevice.hpp"

#include "ExecutionContext.hpp"
#include "MemRefUtils.hpp"

#include "RuntimeCAPI.h"

namespace Catalyst::Runtime {

/**
 * @brief Global quantum device unique pointer.
 */
static std::unique_ptr<ExecutionContext> CTX = nullptr;

/**
 * @brief Thread local device pointer with internal linkage.
 */
thread_local static RTDevice *RTD_PTR = nullptr;

/**
 * @brief Initialize the device instance and update the value of RTD_PTR
 * to the new initialized device pointer.
 */
[[nodiscard]] bool initRTDevicePtr(std::string_view rtd_lib, std::string_view rtd_name,
                                   std::string_view rtd_kwargs)
{
    auto &&device = CTX->getOrCreateDevice(rtd_lib, rtd_name, rtd_kwargs);
    if (device) {
        RTD_PTR = device.get();
        return RTD_PTR ? true : false;
    }
    return false;
}

/**
 * @brief get the active device.
 */
auto getQuantumDevicePtr() -> const std::unique_ptr<QuantumDevice> &
{
    return RTD_PTR->getQuantumDevicePtr();
}

/**
 * @brief Inactivate the active device instance.
 */
void deactivateDevice()
{
    CTX->deactivateDevice(RTD_PTR);
    RTD_PTR = nullptr;
}
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

void __catalyst__rt__print_string(char *string)
{
    if (!string) {
        std::cout << "None" << std::endl;
        return;
    }
    std::cout << string << std::endl;
}

void __catalyst__rt__print_tensor(OpaqueMemRefT *c_memref, bool printDescriptor)
{
    if (c_memref->datatype == NumericType::idx) {
        printMemref<impl::index_type>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::i1) {
        printMemref<bool>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::i8) {
        printMemref<int8_t>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::i16) {
        printMemref<int16_t>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::i32) {
        printMemref<int32_t>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::i64) {
        printMemref<int64_t>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::f32) {
        printMemref<float>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::f64) {
        printMemref<double>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::c64) {
        printMemref<impl::complex32>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else if (c_memref->datatype == NumericType::c128) {
        printMemref<impl::complex64>({c_memref->rank, c_memref->descriptor}, printDescriptor);
    }
    else {
        RT_FAIL("Unkown numeric type encoding for array printing.");
    }

    std::cout << std::endl;
}

void __catalyst__rt__fail_cstr(const char *cstr) { RT_FAIL(cstr); }

void __catalyst__rt__initialize()
{
    Catalyst::Runtime::CTX = std::make_unique<Catalyst::Runtime::ExecutionContext>();
}

void __catalyst__rt__finalize()
{
    Catalyst::Runtime::RTD_PTR = nullptr;
    Catalyst::Runtime::CTX.reset(nullptr);
}

void __catalyst__rt__device_init(int8_t *rtd_lib, int8_t *rtd_name, int8_t *rtd_kwargs)
{
    // Device library cannot be a nullptr
    RT_FAIL_IF(!rtd_lib, "Invalid device library");
    RT_FAIL_IF(!Catalyst::Runtime::CTX, "Invalid use of the global driver before initialization");
    RT_FAIL_IF(Catalyst::Runtime::RTD_PTR,
               "Cannot re-initialize an ACTIVE device: Consider using "
               "__catalyst__rt__device_release before __catalyst__rt__device_init");

    const std::vector<std::string_view> args{
        reinterpret_cast<char *>(rtd_lib), (rtd_name ? reinterpret_cast<char *>(rtd_name) : ""),
        (rtd_kwargs ? reinterpret_cast<char *>(rtd_kwargs) : "")};
    RT_FAIL_IF(!Catalyst::Runtime::initRTDevicePtr(args[0], args[1], args[2]),
               "Failed initialization of the backend device");
    if (Catalyst::Runtime::CTX->getDeviceRecorderStatus()) {
        Catalyst::Runtime::getQuantumDevicePtr()->StartTapeRecording();
    }
}

void __catalyst__rt__device_release()
{
    RT_FAIL_IF(!Catalyst::Runtime::CTX,
               "Cannot release an ACTIVE device out of scope of the global driver");
    // TODO: This will be used for the async support
    Catalyst::Runtime::deactivateDevice();
}

void __catalyst__rt__print_state() { Catalyst::Runtime::getQuantumDevicePtr()->PrintState(); }

void __catalyst__rt__toggle_recorder(bool status)
{
    Catalyst::Runtime::CTX->setDeviceRecorderStatus(status);
    if (!Catalyst::Runtime::RTD_PTR) {
        return;
    }

    if (status) {
        Catalyst::Runtime::getQuantumDevicePtr()->StartTapeRecording();
    }
    else {
        Catalyst::Runtime::getQuantumDevicePtr()->StopTapeRecording();
    }
}

QUBIT *__catalyst__rt__qubit_allocate()
{
    RT_ASSERT(Catalyst::Runtime::getQuantumDevicePtr() != nullptr);
    RT_ASSERT(Catalyst::Runtime::CTX->getMemoryManager() != nullptr);

    return reinterpret_cast<QUBIT *>(Catalyst::Runtime::getQuantumDevicePtr()->AllocateQubit());
}

QirArray *__catalyst__rt__qubit_allocate_array(int64_t num_qubits)
{
    RT_ASSERT(Catalyst::Runtime::getQuantumDevicePtr() != nullptr);
    RT_ASSERT(Catalyst::Runtime::CTX->getMemoryManager() != nullptr);
    RT_ASSERT(num_qubits >= 0);

    QirArray *qubit_array = __quantum__rt__array_create_1d(sizeof(QubitIdType), num_qubits);
    const auto &&qubit_vector =
        Catalyst::Runtime::getQuantumDevicePtr()->AllocateQubits(num_qubits);
    for (int64_t idx = 0; idx < num_qubits; idx++) {
        *reinterpret_cast<QUBIT **>(__quantum__rt__array_get_element_ptr_1d(qubit_array, idx)) =
            reinterpret_cast<QUBIT *>(qubit_vector[idx]);
    }
    return qubit_array;
}

void __catalyst__rt__qubit_release(QUBIT *qubit)
{
    return Catalyst::Runtime::getQuantumDevicePtr()->ReleaseQubit(
        reinterpret_cast<QubitIdType>(qubit));
}

void __catalyst__rt__qubit_release_array(QirArray *qubit_array)
{
    // Update the reference count of qubit_array by -1
    // It will deallocates it iff the reference count becomes 0
    // The behavior is undefined if the reference count becomes < 0
    __quantum__rt__array_update_reference_count(qubit_array, -1);

    Catalyst::Runtime::getQuantumDevicePtr()->ReleaseAllQubits();
}

int64_t __catalyst__rt__num_qubits()
{
    return static_cast<int64_t>(Catalyst::Runtime::getQuantumDevicePtr()->GetNumQubits());
}

QirString *__catalyst__rt__qubit_to_string(QUBIT *qubit)
{
    return __quantum__rt__string_create(
        std::to_string(reinterpret_cast<QubitIdType>(qubit)).c_str());
}

bool __catalyst__rt__result_equal(RESULT *r0, RESULT *r1) { return (r0 == r1) || (*r0 == *r1); }

RESULT *__catalyst__rt__result_get_one() { return Catalyst::Runtime::getQuantumDevicePtr()->One(); }

RESULT *__catalyst__rt__result_get_zero()
{
    return Catalyst::Runtime::getQuantumDevicePtr()->Zero();
}

QirString *__catalyst__rt__result_to_string(RESULT *result)
{
    return __catalyst__rt__result_equal(result, __catalyst__rt__result_get_one())
               ? __quantum__rt__string_create("true")   // one
               : __quantum__rt__string_create("false"); // zero
}

void __catalyst__qis__Gradient(int64_t numResults, /* results = */...)
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
    Catalyst::Runtime::getQuantumDevicePtr()->Gradient(mem_views, {});
}

void __catalyst__qis__Gradient_params(MemRefT_int64_1d *params, int64_t numResults,
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
    Catalyst::Runtime::getQuantumDevicePtr()->Gradient(mem_views, train_params);
}

void __catalyst__qis__Identity(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("Identity", {},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__PauliX(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("PauliX", {},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__PauliY(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("PauliY", {},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__PauliZ(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("PauliZ", {},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__Hadamard(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("Hadamard", {},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__S(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("S", {},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__T(QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("T", {},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__PhaseShift(double theta, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("PhaseShift", {theta},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__RX(double theta, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("RX", {theta},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__RY(double theta, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("RY", {theta},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__RZ(double theta, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("RZ", {theta},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__Rot(double phi, double theta, double omega, QUBIT *qubit, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("Rot", {phi, theta, omega},
                                                             {reinterpret_cast<QubitIdType>(qubit)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__CNOT(QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "CNOT", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__CY(QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "CY", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__CZ(QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "CZ", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__SWAP(QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "SWAP", {},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__IsingXX(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "IsingXX", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__IsingYY(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "IsingYY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__IsingXY(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "IsingXY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__IsingZZ(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "IsingZZ", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__ControlledPhaseShift(double theta, QUBIT *control, QUBIT *target,
                                           bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "ControlledPhaseShift", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__CRX(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "CRX", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__CRY(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "CRY", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__CRZ(double theta, QUBIT *control, QUBIT *target, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "CRZ", {theta},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__CRot(double phi, double theta, double omega, QUBIT *control, QUBIT *target,
                           bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "CRot", {phi, theta, omega},
        {/* control = */ reinterpret_cast<QubitIdType>(control),
         /* target = */ reinterpret_cast<QubitIdType>(target)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__CSWAP(QUBIT *control, QUBIT *aswap, QUBIT *bswap, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "CSWAP", {},
        {reinterpret_cast<QubitIdType>(control), reinterpret_cast<QubitIdType>(aswap),
         reinterpret_cast<QubitIdType>(bswap)},
        /* inverse = */ adjoint);
}

void __catalyst__qis__Toffoli(QUBIT *wire0, QUBIT *wire1, QUBIT *wire2, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("Toffoli", {},
                                                             {reinterpret_cast<QubitIdType>(wire0),
                                                              reinterpret_cast<QubitIdType>(wire1),
                                                              reinterpret_cast<QubitIdType>(wire2)},
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__MultiRZ(double theta, bool adjoint, int64_t numQubits, ...)
{
    RT_ASSERT(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation("MultiRZ", {theta}, wires,
                                                             /* inverse = */ adjoint);
}

void __catalyst__qis__ISWAP(QUBIT *wire0, QUBIT *wire1, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "ISWAP", {}, {reinterpret_cast<QubitIdType>(wire0), reinterpret_cast<QubitIdType>(wire1)},
        adjoint);
}

void __catalyst__qis__PSWAP(double phi, QUBIT *wire0, QUBIT *wire1, bool adjoint)
{
    Catalyst::Runtime::getQuantumDevicePtr()->NamedOperation(
        "PSWAP", {phi},
        {reinterpret_cast<QubitIdType>(wire0), reinterpret_cast<QubitIdType>(wire1)}, adjoint);
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

void __catalyst__qis__QubitUnitary(MemRefT_CplxT_double_2d *matrix, bool adjoint, int64_t numQubits,
                                   /*qubits*/...)
{
    RT_ASSERT(numQubits >= 0);

    if (matrix == nullptr) {
        RT_FAIL("The QubitUnitary matrix must be initialized");
    }

    if (numQubits > __catalyst__rt__num_qubits()) {
        RT_FAIL("Invalid number of wires");
    }

    va_list args;
    std::vector<std::complex<double>> coeffs;
    std::vector<QubitIdType> wires;
    va_start(args, numQubits);
    _qubitUnitary_impl(matrix, numQubits, coeffs, wires, &args);
    va_end(args);
    return Catalyst::Runtime::getQuantumDevicePtr()->MatrixOperation(coeffs, wires,
                                                                     /*inverse = */ adjoint);
}

ObsIdType __catalyst__qis__NamedObs(int64_t obsId, QUBIT *wire)
{
    return Catalyst::Runtime::getQuantumDevicePtr()->Observable(
        static_cast<ObsId>(obsId), {}, {reinterpret_cast<QubitIdType>(wire)});
}

ObsIdType __catalyst__qis__HermitianObs(MemRefT_CplxT_double_2d *matrix, int64_t numQubits, ...)
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

    if (numQubits > __catalyst__rt__num_qubits()) {
        RT_FAIL("Invalid number of wires");
    }

    const size_t matrix_size = num_rows * num_col;
    std::vector<std::complex<double>> coeffs;
    coeffs.reserve(matrix_size);
    for (size_t i = 0; i < matrix_size; i++) {
        coeffs.emplace_back(matrix->data_aligned[i].real, matrix->data_aligned[i].imag);
    }

    return Catalyst::Runtime::getQuantumDevicePtr()->Observable(ObsId::Hermitian, coeffs, wires);
}

ObsIdType __catalyst__qis__TensorObs(int64_t numObs, /*obsKeys*/...)
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

    return Catalyst::Runtime::getQuantumDevicePtr()->TensorObservable(obsKeys);
}

ObsIdType __catalyst__qis__HamiltonianObs(MemRefT_double_1d *coeffs, int64_t numObs,
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
    return Catalyst::Runtime::getQuantumDevicePtr()->HamiltonianObservable(coeffs_vec, obsKeys);
}

RESULT *__catalyst__qis__Measure(QUBIT *wire)
{
    return Catalyst::Runtime::getQuantumDevicePtr()->Measure(reinterpret_cast<QubitIdType>(wire));
}

double __catalyst__qis__Expval(ObsIdType obsKey)
{
    return Catalyst::Runtime::getQuantumDevicePtr()->Expval(obsKey);
}

double __catalyst__qis__Variance(ObsIdType obsKey)
{
    return Catalyst::Runtime::getQuantumDevicePtr()->Var(obsKey);
}

void __catalyst__qis__State(MemRefT_CplxT_double_1d *result, int64_t numQubits, ...)
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
        Catalyst::Runtime::getQuantumDevicePtr()->State(view);
    }
    else {
        RT_FAIL("Partial State-Vector not supported yet");
        // Catalyst::Runtime::getQuantumDevicePtr()->PartialState(stateVec,
        // numElements, wires);
    }
}

void __catalyst__qis__Probs(MemRefT_double_1d *result, int64_t numQubits, ...)
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
        Catalyst::Runtime::getQuantumDevicePtr()->Probs(view);
    }
    else {
        Catalyst::Runtime::getQuantumDevicePtr()->PartialProbs(view, wires);
    }
}

void __catalyst__qis__Sample(MemRefT_double_2d *result, int64_t shots, int64_t numQubits, ...)
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
        Catalyst::Runtime::getQuantumDevicePtr()->Sample(view, shots);
    }
    else {
        Catalyst::Runtime::getQuantumDevicePtr()->PartialSample(view, wires, shots);
    }
}

void __catalyst__qis__Counts(PairT_MemRefT_double_int64_1d *result, int64_t shots,
                             int64_t numQubits, ...)
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
        Catalyst::Runtime::getQuantumDevicePtr()->Counts(eigvals_view, counts_view, shots);
    }
    else {
        Catalyst::Runtime::getQuantumDevicePtr()->PartialCounts(eigvals_view, counts_view, wires,
                                                                shots);
    }
}
}
