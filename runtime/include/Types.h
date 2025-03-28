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
#ifndef TYPES_H
#define TYPES_H

#include <cmath>
#include <cstdint>
#include <limits>

#ifdef __cplusplus
extern "C" {
#endif

// Qubit, Result and Observable types
struct QUBIT;
using QubitIdType = intptr_t;

using RESULT = bool;
using Result = RESULT *;
using QirArray = void *;

using ObsIdType = intptr_t;

enum ObsId : int8_t {
    Identity = 0,
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    Hermitian,
};

enum ObsType : int8_t {
    Basic = 0,
    TensorProd,
    Hamiltonian,
};

// complex<float> type
struct CplxT_float {
    float real;
    float imag;
};

// complex<double> type
struct CplxT_double {
    double real;
    double imag;
};

enum NumericType : int8_t {
    idx = 0,
    i1,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
    c64,
    c128,
};

// MemRefT<datatype, dimension=rank> type
struct OpaqueMemRefT {
    int64_t rank;
    void *descriptor;
    NumericType datatype;
};

// MemRefT<complex<double>, dimension=1> type
struct MemRefT_CplxT_double_1d {
    CplxT_double *data_allocated;
    CplxT_double *data_aligned;
    size_t offset;
    size_t sizes[1];
    size_t strides[1];
};

// MemRefT<complex<double>, dimension=2> type
struct MemRefT_CplxT_double_2d {
    CplxT_double *data_allocated;
    CplxT_double *data_aligned;
    size_t offset;
    size_t sizes[2];
    size_t strides[2];
};

// MemRefT<double, dimension=1> type
struct MemRefT_double_1d {
    double *data_allocated;
    double *data_aligned;
    size_t offset;
    size_t sizes[1];
    size_t strides[1];
};

// MemRefT<double, dimension=2> type
struct MemRefT_double_2d {
    double *data_allocated;
    double *data_aligned;
    size_t offset;
    size_t sizes[2];
    size_t strides[2];
};

// MemRefT<int64_t, dimension=1> type
struct MemRefT_int64_1d {
    int64_t *data_allocated;
    int64_t *data_aligned;
    size_t offset;
    size_t sizes[1];
    size_t strides[1];
};

// MemRefT<int64_t, dimension=1> type
struct MemRefT_int8_1d {
    int8_t *data_allocated;
    int8_t *data_aligned;
    size_t offset;
    size_t sizes[1];
    size_t strides[1];
};

// PairT<MemRefT<double, dimension=1>, MemRefT<int64, dimension=2>> type
struct PairT_MemRefT_double_int64_1d {
    struct MemRefT_double_1d first;
    struct MemRefT_int64_1d second;
};

// Quantum operation modifiers
struct Modifiers {
    bool adjoint;
    size_t num_controlled;
    QUBIT *controlled_wires;
    bool *controlled_values;
};

using CplxT_double = struct CplxT_double;
using MemRefT_CplxT_double_1d = struct MemRefT_CplxT_double_1d;
using MemRefT_CplxT_double_2d = struct MemRefT_CplxT_double_2d;
using MemRefT_double_1d = struct MemRefT_double_1d;
using MemRefT_double_2d = struct MemRefT_double_2d;
using MemRefT_int64_1d = struct MemRefT_int64_1d;
using PairT_MemRefT_double_int64_1d = struct PairT_MemRefT_double_int64_1d;
using Modifiers = struct Modifiers;

#ifdef __cplusplus
} // extern "C"
#endif

#endif
