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

#include <memory>
#include <ostream>
#include <string_view>

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "Exception.hpp"
#include "QuantumDevice.hpp"

#include "ExecutionContext.hpp"
#include "MemRefUtils.hpp"
#include "Timer.hpp"

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

bool getModifiersAdjoint(const Modifiers *modifiers)
{
    return !modifiers ? false : modifiers->adjoint;
}

std::vector<QubitIdType> getModifiersControlledWires(const Modifiers *modifiers)
{
    return !modifiers ? std::vector<QubitIdType>()
                      : std::vector<QubitIdType>(
                            reinterpret_cast<QubitIdType *>(modifiers->controlled_wires),
                            reinterpret_cast<QubitIdType *>(modifiers->controlled_wires) +
                                modifiers->num_controlled);
}

std::vector<bool> getModifiersControlledValues(const Modifiers *modifiers)
{
    return !modifiers ? std::vector<bool>()
                      : std::vector<bool>(modifiers->controlled_values,
                                          modifiers->controlled_values + modifiers->num_controlled);
}

#define MODIFIERS_ARGS(mod)                                                                        \
    getModifiersAdjoint(mod), getModifiersControlledWires(mod), getModifiersControlledValues(mod)

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

using namespace Catalyst::Runtime;
using timer = catalyst::utils::Timer;

void __catalyst_inactive_callback(int64_t identifier, int64_t argc, int64_t retc, ...)
{
    // LIBREGISTRY is a compile time macro. It is defined based on the output
    // name of the callback library. And since it is stored in the same location
    // as this library, it shares the ORIGIN variable. Do a `git grep LIBREGISTRY`
    // to find its definition in the CMakeFiles.
    // It is the name of the library that contains the callbackCall implementation.
    // The reason why this is using dlopen is because we have historically wanted
    // to avoid a dependency of python in the runtime.
    // With dlopen, we leave the possibility of linking against the runtime without
    // linking with LIBREGISTRY which is implemented as a pybind11 module.
    //
    // The only restriction is that there should be no calls to pyregsitry.
    //
    // This function cannot be tested from the runtime tests because there would be no valid python
    // function to callback...
    void *handle = dlopen(LIBREGISTRY, RTLD_LAZY);
    if (!handle) {
        char *err_msg = dlerror();
        RT_FAIL(err_msg);
    }

    void (*callbackCall)(int64_t, int64_t, int64_t, va_list);
    typedef void (*func_ptr_t)(int64_t, int64_t, int64_t, va_list);
    callbackCall = (func_ptr_t)dlsym(handle, "callbackCall");
    if (!callbackCall) {
        char *err_msg = dlerror();
        RT_FAIL(err_msg);
    }

    va_list args;
    va_start(args, retc);
    callbackCall(identifier, argc, retc, args);
    va_end(args);
    dlclose(handle);
}

void __catalyst__host__rt__unrecoverable_error()
{
    RT_FAIL("Unrecoverable error from asynchronous execution of multiple quantum programs.");
}

void *_mlir_memref_to_llvm_alloc(size_t size)
{
    void *ptr = malloc(size);
    CTX->getMemoryManager()->insert(ptr);
    return ptr;
}

void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size)
{
    void *ptr = aligned_alloc(alignment, size);
    CTX->getMemoryManager()->insert(ptr);
    return ptr;
}

bool _mlir_memory_transfer(void *ptr)
{
    if (!CTX->getMemoryManager()->contains(ptr)) {
        return false;
    }
    CTX->getMemoryManager()->erase(ptr);
    return true;
}

void _mlir_memref_to_llvm_free(void *ptr)
{
    CTX->getMemoryManager()->erase(ptr);
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

void __catalyst__rt__assert_bool(bool p, char *s) { RT_FAIL_IF(!p, s); }

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

void __catalyst__rt__initialize(uint32_t *seed) { CTX = std::make_unique<ExecutionContext>(seed); }

void __catalyst__rt__finalize()
{
    RTD_PTR = nullptr;
    CTX.reset(nullptr);
}

static int __catalyst__rt__device_init__impl(int8_t *rtd_lib, int8_t *rtd_name, int8_t *rtd_kwargs,
                                             int64_t shots)
{
    // Device library cannot be a nullptr
    RT_FAIL_IF(!rtd_lib, "Invalid device library");
    RT_FAIL_IF(!CTX, "Invalid use of the global driver before initialization");
    RT_FAIL_IF(RTD_PTR, "Cannot re-initialize an ACTIVE device: Consider using "
                        "__catalyst__rt__device_release before __catalyst__rt__device_init");

    const std::vector<std::string_view> args{
        reinterpret_cast<char *>(rtd_lib), (rtd_name ? reinterpret_cast<char *>(rtd_name) : ""),
        (rtd_kwargs ? reinterpret_cast<char *>(rtd_kwargs) : "")};
    RT_FAIL_IF(!initRTDevicePtr(args[0], args[1], args[2]),
               "Failed initialization of the backend device");
    getQuantumDevicePtr()->SetDeviceShots(shots);
    if (CTX->getDeviceRecorderStatus()) {
        getQuantumDevicePtr()->StartTapeRecording();
    }
    return 0;
}

void __catalyst__rt__device_init(int8_t *rtd_lib, int8_t *rtd_name, int8_t *rtd_kwargs,
                                 int64_t shots)
{
    timer::timer(__catalyst__rt__device_init__impl, "device_init", /* add_endl */ true, rtd_lib,
                 rtd_name, rtd_kwargs, shots);
}

static int __catalyst__rt__device_release__impl()
{
    RT_FAIL_IF(!CTX, "Cannot release an ACTIVE device out of scope of the global driver");
    // TODO: This will be used for the async support
    deactivateDevice();
    return 0;
}

void __catalyst__rt__device_release()
{
    timer::timer(__catalyst__rt__device_release__impl, "device_release", /* add_endl */ true);
}

void __catalyst__rt__print_state() { getQuantumDevicePtr()->PrintState(); }

void __catalyst__rt__toggle_recorder(bool status)
{
    CTX->setDeviceRecorderStatus(status);
    if (!RTD_PTR) {
        return;
    }

    if (status) {
        getQuantumDevicePtr()->StartTapeRecording();
    }
    else {
        getQuantumDevicePtr()->StopTapeRecording();
    }
}

static QUBIT *__catalyst__rt__qubit_allocate__impl()
{
    RT_ASSERT(getQuantumDevicePtr() != nullptr);
    RT_ASSERT(CTX->getMemoryManager() != nullptr);

    return reinterpret_cast<QUBIT *>(getQuantumDevicePtr()->AllocateQubit());
}

QUBIT *__catalyst__rt__qubit_allocate()
{
    return timer::timer(__catalyst__rt__qubit_allocate__impl, "qubit_allocate",
                        /* add_endl */ true);
}

static QirArray *__catalyst__rt__qubit_allocate_array__impl(int64_t num_qubits)
{
    RT_ASSERT(getQuantumDevicePtr() != nullptr);
    RT_ASSERT(CTX->getMemoryManager() != nullptr);
    RT_ASSERT(num_qubits >= 0);

    // For first prototype, we just want to make this work.
    // But ideally, I think the device should determine the representation.
    // Essentially just forward this to the device library.
    // And the device library can choose how to handle everything.
    std::vector<QubitIdType> qubit_vector = getQuantumDevicePtr()->AllocateQubits(num_qubits);

    // I don't like this copying.
    std::vector<QubitIdType> *qubit_vector_ptr =
        new std::vector<QubitIdType>(qubit_vector.begin(), qubit_vector.end());

    // Because this function is interfacing with C
    // I think we should return a trivial-type
    //     https://en.cppreference.com/w/cpp/named_req/TrivialType
    // Why should we return a trivial type?
    //
    // Paraphrasing from stackoverflow: https://stackoverflow.com/a/72409589
    //     extern "C" will avoid name mangling from happening.
    //     It doesn't prevent a function from returning or accepting a C++ type.
    //     But the calling language needs to understand the data-layout for the
    //     type being returned.
    //     For non-trivial types, this will be difficult to impossible.
    return (QirArray *)qubit_vector_ptr;
}

QirArray *__catalyst__rt__qubit_allocate_array(int64_t num_qubits)
{
    return timer::timer(__catalyst__rt__qubit_allocate_array__impl, "qubit_allocate_array",
                        /* add_endl */ true, num_qubits);
}

static int __catalyst__rt__qubit_release__impl(QUBIT *qubit)
{
    getQuantumDevicePtr()->ReleaseQubit(reinterpret_cast<QubitIdType>(qubit));
    return 0;
}

void __catalyst__rt__qubit_release(QUBIT *qubit)
{
    timer::timer(__catalyst__rt__qubit_release__impl, "qubit_release",
                 /* add_endl */ true, qubit);
}

static int __catalyst__rt__qubit_release_array__impl(QirArray *qubit_array)
{
    getQuantumDevicePtr()->ReleaseAllQubits();
    std::vector<QubitIdType> *qubit_array_ptr =
        reinterpret_cast<std::vector<QubitIdType> *>(qubit_array);
    delete qubit_array_ptr;
    return 0;
}

void __catalyst__rt__qubit_release_array(QirArray *qubit_array)
{
    timer::timer(__catalyst__rt__qubit_release_array__impl, "qubit_release_array",
                 /* add_endl */ true, qubit_array);
}

int64_t __catalyst__rt__num_qubits()
{
    return static_cast<int64_t>(getQuantumDevicePtr()->GetNumQubits());
}

bool __catalyst__rt__result_equal(RESULT *r0, RESULT *r1) { return (r0 == r1) || (*r0 == *r1); }

RESULT *__catalyst__rt__result_get_one() { return getQuantumDevicePtr()->One(); }

RESULT *__catalyst__rt__result_get_zero() { return getQuantumDevicePtr()->Zero(); }

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
    getQuantumDevicePtr()->Gradient(mem_views, {});
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
    getQuantumDevicePtr()->Gradient(mem_views, train_params);
}

void __catalyst__qis__GlobalPhase(double phi, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("GlobalPhase", {phi}, {}, MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__SetState(MemRefT_CplxT_double_1d *data, uint64_t numQubits, ...)
{
    RT_ASSERT(numQubits > 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (uint64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    MemRefT<std::complex<double>, 1> *data_p = (MemRefT<std::complex<double>, 1> *)data;
    DataView<std::complex<double>, 1> data_view(data_p->data_aligned, data_p->offset, data_p->sizes,
                                                data_p->strides);
    getQuantumDevicePtr()->SetState(data_view, wires);
}

void __catalyst__qis__SetBasisState(MemRefT_int8_1d *data, uint64_t numQubits, ...)
{
    RT_ASSERT(numQubits > 0);

    DataView<int8_t, 1> data_view(data->data_aligned, data->offset, data->sizes, data->strides);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (uint64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);
    std::unordered_set<QubitIdType> wire_set(wires.begin(), wires.end());
    RT_FAIL_IF(wire_set.size() != numQubits, "Wires must be unique");
    RT_FAIL_IF(data->sizes[0] != numQubits,
               "BasisState parameter and wires must be of equal length.");

    getQuantumDevicePtr()->SetBasisState(data_view, wires);
}

void __catalyst__qis__Identity(QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("Identity", {}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__PauliX(QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("PauliX", {}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__PauliY(QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("PauliY", {}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__PauliZ(QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("PauliZ", {}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__Hadamard(QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("Hadamard", {}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__S(QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("S", {}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__T(QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("T", {}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__PhaseShift(double theta, QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation(
        "PhaseShift", {theta}, {reinterpret_cast<QubitIdType>(qubit)}, MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__RX(double theta, QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("RX", {theta}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__RY(double theta, QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("RY", {theta}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__RZ(double theta, QUBIT *qubit, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("RZ", {theta}, {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__Rot(double phi, double theta, double omega, QUBIT *qubit,
                          const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("Rot", {phi, theta, omega},
                                          {reinterpret_cast<QubitIdType>(qubit)},
                                          MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__CNOT(QUBIT *control, QUBIT *target, const Modifiers *modifiers)
{
    RT_FAIL_IF(control == target,
               "Invalid input for CNOT gate. Control and target qubit operands must be distinct.");
    getQuantumDevicePtr()->NamedOperation("CNOT", {},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__CY(QUBIT *control, QUBIT *target, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("CY", {},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__CZ(QUBIT *control, QUBIT *target, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("CZ", {},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__SWAP(QUBIT *control, QUBIT *target, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("SWAP", {},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__IsingXX(double theta, QUBIT *control, QUBIT *target,
                              const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("IsingXX", {theta},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__IsingYY(double theta, QUBIT *control, QUBIT *target,
                              const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("IsingYY", {theta},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__IsingXY(double theta, QUBIT *control, QUBIT *target,
                              const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("IsingXY", {theta},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__IsingZZ(double theta, QUBIT *control, QUBIT *target,
                              const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("IsingZZ", {theta},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__ControlledPhaseShift(double theta, QUBIT *control, QUBIT *target,
                                           const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("ControlledPhaseShift", {theta},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__CRX(double theta, QUBIT *control, QUBIT *target, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("CRX", {theta},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__CRY(double theta, QUBIT *control, QUBIT *target, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("CRY", {theta},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__CRZ(double theta, QUBIT *control, QUBIT *target, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("CRZ", {theta},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__CRot(double phi, double theta, double omega, QUBIT *control, QUBIT *target,
                           const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("CRot", {phi, theta, omega},
                                          {/* control = */ reinterpret_cast<QubitIdType>(control),
                                           /* target = */ reinterpret_cast<QubitIdType>(target)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__CSWAP(QUBIT *control, QUBIT *aswap, QUBIT *bswap, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("CSWAP", {},
                                          {reinterpret_cast<QubitIdType>(control),
                                           reinterpret_cast<QubitIdType>(aswap),
                                           reinterpret_cast<QubitIdType>(bswap)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__Toffoli(QUBIT *wire0, QUBIT *wire1, QUBIT *wire2, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation("Toffoli", {},
                                          {reinterpret_cast<QubitIdType>(wire0),
                                           reinterpret_cast<QubitIdType>(wire1),
                                           reinterpret_cast<QubitIdType>(wire2)},
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__MultiRZ(double theta, const Modifiers *modifiers, int64_t numQubits, ...)
{
    RT_ASSERT(numQubits >= 0);

    va_list args;
    va_start(args, numQubits);
    std::vector<QubitIdType> wires(numQubits);
    for (int64_t i = 0; i < numQubits; i++) {
        wires[i] = va_arg(args, QubitIdType);
    }
    va_end(args);

    getQuantumDevicePtr()->NamedOperation("MultiRZ", {theta}, wires,
                                          /* modifiers */ MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__ISWAP(QUBIT *wire0, QUBIT *wire1, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation(
        "ISWAP", {}, {reinterpret_cast<QubitIdType>(wire0), reinterpret_cast<QubitIdType>(wire1)},
        MODIFIERS_ARGS(modifiers));
}

void __catalyst__qis__PSWAP(double phi, QUBIT *wire0, QUBIT *wire1, const Modifiers *modifiers)
{
    getQuantumDevicePtr()->NamedOperation(
        "PSWAP", {phi},
        {reinterpret_cast<QubitIdType>(wire0), reinterpret_cast<QubitIdType>(wire1)},
        MODIFIERS_ARGS(modifiers));
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

void __catalyst__qis__QubitUnitary(MemRefT_CplxT_double_2d *matrix, const Modifiers *modifiers,
                                   int64_t numQubits, /*qubits*/...)
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
    return getQuantumDevicePtr()->MatrixOperation(coeffs, wires, MODIFIERS_ARGS(modifiers));
}

ObsIdType __catalyst__qis__NamedObs(int64_t obsId, QUBIT *wire)
{
    return getQuantumDevicePtr()->Observable(static_cast<ObsId>(obsId), {},
                                             {reinterpret_cast<QubitIdType>(wire)});
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

    return getQuantumDevicePtr()->Observable(ObsId::Hermitian, coeffs, wires);
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

    return getQuantumDevicePtr()->TensorObservable(obsKeys);
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
    return getQuantumDevicePtr()->HamiltonianObservable(coeffs_vec, obsKeys);
}

RESULT *__catalyst__qis__Measure(QUBIT *wire, int32_t postselect)
{
    std::optional<int32_t> postselectOpt{postselect};

    // Any value different to 0 or 1 denotes absence of postselect, and it is hence turned into
    // std::nullopt at the C++ interface
    if (postselect != 0 && postselect != 1) {
        postselectOpt = std::nullopt;
    }

    return getQuantumDevicePtr()->Measure(reinterpret_cast<QubitIdType>(wire), postselectOpt);
}

double __catalyst__qis__Expval(ObsIdType obsKey) { return getQuantumDevicePtr()->Expval(obsKey); }

double __catalyst__qis__Variance(ObsIdType obsKey) { return getQuantumDevicePtr()->Var(obsKey); }

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
        getQuantumDevicePtr()->State(view);
    }
    else {
        RT_FAIL("Partial State-Vector not supported yet");
        // getQuantumDevicePtr()->PartialState(stateVec,
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
        getQuantumDevicePtr()->Probs(view);
    }
    else {
        getQuantumDevicePtr()->PartialProbs(view, wires);
    }
}

void __catalyst__qis__Sample(MemRefT_double_2d *result, int64_t numQubits, ...)
{
    int64_t shots = getQuantumDevicePtr()->GetDeviceShots();
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
        getQuantumDevicePtr()->Sample(view, shots);
    }
    else {
        getQuantumDevicePtr()->PartialSample(view, wires, shots);
    }
}

void __catalyst__qis__Counts(PairT_MemRefT_double_int64_1d *result, int64_t numQubits, ...)
{
    int64_t shots = getQuantumDevicePtr()->GetDeviceShots();
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
        getQuantumDevicePtr()->Counts(eigvals_view, counts_view, shots);
    }
    else {
        getQuantumDevicePtr()->PartialCounts(eigvals_view, counts_view, wires, shots);
    }
}

int64_t __catalyst__rt__array_get_size_1d(QirArray *ptr)
{
    std::vector<QubitIdType> *qubit_vector_ptr = reinterpret_cast<std::vector<QubitIdType> *>(ptr);
    return qubit_vector_ptr->size();
}

int8_t *__catalyst__rt__array_get_element_ptr_1d(QirArray *ptr, int64_t idx)
{
    std::vector<QubitIdType> *qubit_vector_ptr = reinterpret_cast<std::vector<QubitIdType> *>(ptr);

    RT_ASSERT(idx >= 0);
    std::string error_msg = "The qubit register does not contain the requested wire: ";
    error_msg += std::to_string(idx);
    RT_FAIL_IF(static_cast<size_t>(idx) >= qubit_vector_ptr->size(), error_msg.c_str());

    QubitIdType *data = qubit_vector_ptr->data();
    return (int8_t *)&data[idx];
}
}
