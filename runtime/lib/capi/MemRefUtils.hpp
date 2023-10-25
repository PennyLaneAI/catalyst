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

#pragma once

#include <cstddef>
#include <sstream>
#include <type_traits>
#include <variant>
#include <vector>

#include "DataView.hpp"
#include "Exception.hpp"
#include "Types.h"

extern "C" {
void *_mlir_memref_to_llvm_alloc(size_t size);
void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size);
bool _mlir_memory_transfer(void *);
void _mlir_memref_to_llvm_free(void *ptr);
}

// MemRef type definition
template <typename T, size_t R> struct MemRefT {
    T *data_allocated;
    T *data_aligned;
    size_t offset;
    size_t sizes[R];
    size_t strides[R];
};

typedef std::variant<bool *, int16_t *, int32_t *, int64_t *, float *, double *, CplxT_float *,
                     CplxT_double *>
    NumericPtr;

struct DynamicMemRefT {
    NumericPtr data_allocated;
    NumericPtr data_aligned;
    size_t offset;
    std::vector<size_t> sizes;
    std::vector<size_t> strides;
    std::ostringstream in_str;
};

template <typename T, typename TPtr>
static inline void _gen_oss_dyn_memref(DynamicMemRefT &newMemref, int64_t rank)
{
    const auto data = std::get<TPtr>(newMemref.data_aligned);
    const auto sizes = newMemref.sizes;

    if (rank > 0) {
        newMemref.in_str << "[ ";
    }

    if (rank == 0) {
        if constexpr (std::is_same_v<T, CplxT_float> || std::is_same_v<T, CplxT_double>) {
            newMemref.in_str << data[0].real << "+" << data[0].imag << "j";
        }
        else {
            newMemref.in_str << data[0];
        }
    }
    else if (rank == 1) {
        DataView<T, 1> view(data, newMemref.offset, newMemref.sizes.data(),
                            newMemref.strides.data());
        for (const auto elem : view) {
            if constexpr (std::is_same_v<T, CplxT_float> || std::is_same_v<T, CplxT_double>) {
                newMemref.in_str << elem.real << "+" << elem.imag << "j";
            }
            else {
                newMemref.in_str << elem << " ";
            }
        }
    }
    else if (rank == 2) {
        DataView<T, 2> view(data, newMemref.offset, newMemref.sizes.data(),
                            newMemref.strides.data());
        size_t col_index = 0;
        newMemref.in_str << "[ ";
        for (const auto elem : view) {
            if (col_index == sizes[1]) {
                newMemref.in_str << "]";
                col_index = 0;
                newMemref.in_str << ", [ ";
            }
            if constexpr (std::is_same_v<T, CplxT_float> || std::is_same_v<T, CplxT_double>) {
                newMemref.in_str << elem.real << "+" << elem.imag << "j";
            }
            else {
                newMemref.in_str << elem << " ";
            }
            col_index++;
        }
        newMemref.in_str << "] ";
    }
    else {
        newMemref.in_str << "MemRef of Rank > 2 ";
    }

    if (rank > 0) {
        newMemref.in_str << "]";
    }
}

inline DynamicMemRefT get_dynamic_memref(OpaqueMemRefT memref)
{
    DynamicMemRefT newMemref;
    const auto rank = memref.rank;

    newMemref.offset = *(size_t *)((char *)memref.descriptor + 2 * sizeof(void *));
    size_t *sizesPtr = (size_t *)((char *)memref.descriptor + 2 * sizeof(void *) + sizeof(size_t));
    size_t *stridesPtr = (size_t *)((char *)memref.descriptor + 2 * sizeof(void *) +
                                    sizeof(size_t) + rank * sizeof(size_t));

    newMemref.sizes.assign(sizesPtr, sizesPtr + rank);
    newMemref.strides.assign(stridesPtr, stridesPtr + rank);

    switch (memref.datatype) {
    case NumericType::i1:
        newMemref.data_allocated = *((bool **)memref.descriptor);
        newMemref.data_aligned = *((bool **)memref.descriptor + 1);
        _gen_oss_dyn_memref<bool, bool *>(newMemref, rank);
        break;
    case NumericType::i16:
        newMemref.data_allocated = *((int16_t **)memref.descriptor);
        newMemref.data_aligned = *((int16_t **)memref.descriptor + 1);
        _gen_oss_dyn_memref<int16_t, int16_t *>(newMemref, rank);
        break;
    case NumericType::i32:
        newMemref.data_allocated = *((int32_t **)memref.descriptor);
        newMemref.data_aligned = *((int32_t **)memref.descriptor + 1);
        _gen_oss_dyn_memref<int32_t, int32_t *>(newMemref, rank);
        break;
    case NumericType::i64:
        newMemref.data_allocated = *((int64_t **)memref.descriptor);
        newMemref.data_aligned = *((int64_t **)memref.descriptor + 1);
        _gen_oss_dyn_memref<int64_t, int64_t *>(newMemref, rank);
        break;
    case NumericType::f32:
        newMemref.data_allocated = *((float **)memref.descriptor);
        newMemref.data_aligned = *((float **)memref.descriptor + 1);
        _gen_oss_dyn_memref<float, float *>(newMemref, rank);
        break;
    case NumericType::f64:
        newMemref.data_allocated = *((double **)memref.descriptor);
        newMemref.data_aligned = *((double **)memref.descriptor + 1);
        _gen_oss_dyn_memref<double, double *>(newMemref, rank);
        break;
    case NumericType::c64:
        newMemref.data_allocated = *((CplxT_float **)memref.descriptor);
        newMemref.data_aligned = *((CplxT_float **)memref.descriptor + 1);
        _gen_oss_dyn_memref<CplxT_float, CplxT_float *>(newMemref, rank);
        break;
    case NumericType::c128:
        newMemref.data_allocated = *((CplxT_double **)memref.descriptor);
        newMemref.data_aligned = *((CplxT_double **)memref.descriptor + 1);
        _gen_oss_dyn_memref<CplxT_double, CplxT_double *>(newMemref, rank);
        break;
    default:
        RT_FAIL("Unkown numeric type encoding for array printing.");
    }

    return newMemref;
}
