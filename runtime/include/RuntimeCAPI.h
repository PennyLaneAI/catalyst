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

#pragma once
#ifndef RUNTIMECAPI_H
#define RUNTIMECAPI_H

#include "Types.h"

#include "qir_stdlib.h"

#ifdef __cplusplus

template <typename T, size_t R> struct MemRefT {
    T *data_allocated;
    T *data_aligned;
    size_t offset;
    size_t sizes[R];
    size_t strides[R];
};

template <typename T, size_t R> void memref_copy(MemRefT<T, R> *memref, T *buffer)
{
    size_t how_many_elements = 1;
    for (size_t i = 0; i < R; i++) {
        how_many_elements *= memref->sizes[i];
    }
    memcpy(memref->data_aligned, buffer, sizeof(T) * how_many_elements);
}

extern "C" {
#endif

// Quantum Runtime Instructions
void __quantum__rt__fail_cstr(const char *cstr);
void __quantum__rt__initialize();
void __quantum__rt__finalize();
void __quantum__rt__toggle_recorder(bool);
void __quantum__rt__print_state();

QUBIT *__quantum__rt__qubit_allocate();
QirArray *__quantum__rt__qubit_allocate_array(int64_t);
void __quantum__rt__qubit_release(QUBIT *);
void __quantum__rt__qubit_release_array(QirArray *);

int64_t __quantum__rt__num_qubits();
QirString *__quantum__rt__qubit_to_string(QUBIT *);

bool __quantum__rt__result_equal(RESULT *, RESULT *);
RESULT *__quantum__rt__result_get_one();
RESULT *__quantum__rt__result_get_zero();
QirString *__quantum__rt__result_to_string(RESULT *);

// Quantum Gate Set Instructions
void __quantum__qis__Identity(QUBIT *);
void __quantum__qis__PauliX(QUBIT *);
void __quantum__qis__PauliY(QUBIT *);
void __quantum__qis__PauliZ(QUBIT *);
void __quantum__qis__Hadamard(QUBIT *);
void __quantum__qis__S(QUBIT *);
void __quantum__qis__T(QUBIT *);
void __quantum__qis__PhaseShift(double, QUBIT *);
void __quantum__qis__RX(double, QUBIT *);
void __quantum__qis__RY(double, QUBIT *);
void __quantum__qis__RZ(double, QUBIT *);
void __quantum__qis__Rot(double, double, double, QUBIT *);

void __quantum__qis__CNOT(QUBIT *, QUBIT *);
void __quantum__qis__CY(QUBIT *, QUBIT *);
void __quantum__qis__CZ(QUBIT *, QUBIT *);
void __quantum__qis__SWAP(QUBIT *, QUBIT *);
void __quantum__qis__IsingXX(double, QUBIT *, QUBIT *);
void __quantum__qis__IsingYY(double, QUBIT *, QUBIT *);
void __quantum__qis__IsingXY(double, QUBIT *, QUBIT *);
void __quantum__qis__IsingZZ(double, QUBIT *, QUBIT *);
void __quantum__qis__ControlledPhaseShift(double, QUBIT *, QUBIT *);
void __quantum__qis__CRX(double, QUBIT *, QUBIT *);
void __quantum__qis__CRY(double, QUBIT *, QUBIT *);
void __quantum__qis__CRZ(double, QUBIT *, QUBIT *);
void __quantum__qis__CRot(double, double, double, QUBIT *, QUBIT *);

void __quantum__qis__CSWAP(QUBIT *, QUBIT *, QUBIT *);
void __quantum__qis__Toffoli(QUBIT *, QUBIT *, QUBIT *);

void __quantum__qis__MultiRZ(double, int64_t numQubits, /*qubits*/...);

// Struct pointer arguments for these instructions represent real arguments,
// as passing structs by value is too unreliable / compiler dependant.
void __quantum__qis__QubitUnitary(MemRefT_CplxT_double_2d *, int64_t numQubits, /*qubits*/...);

ObsIdType __quantum__qis__NamedObs(int64_t, QUBIT *);
ObsIdType __quantum__qis__HermitianObs(MemRefT_CplxT_double_2d *, int64_t numQubits,
                                       /*qubits*/...);
ObsIdType __quantum__qis__TensorObs(int64_t numObs, /*obsKeys*/...);
ObsIdType __quantum__qis__HamiltonianObs(MemRefT_double_1d *, int64_t numObs,
                                         /*obsKeys*/...);

// Struct pointers arguments here represent return values.
RESULT *__quantum__qis__Measure(QUBIT *);
double __quantum__qis__Expval(ObsIdType);
double __quantum__qis__Variance(ObsIdType);
void __quantum__qis__Probs(MemRefT_double_1d *, int64_t numQubits,
                           /*qubits*/...);
void __quantum__qis__Sample(MemRefT_double_2d *, int64_t shots, int64_t numQubits,
                            /*qubits*/...);
void __quantum__qis__Counts(PairT_MemRefT_double_int64_1d *, int64_t shots, int64_t numQubits,
                            /*qubits*/...);
void __quantum__qis__State(MemRefT_CplxT_double_1d *, int64_t numQubits,
                           /*qubits*/...);
void __quantum__qis__Gradient(int64_t numResults, /*results*/...);
void __quantum__qis__Gradient_params(MemRefT_int64_1d *, int64_t numResults,
                                     /*results*/...);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
