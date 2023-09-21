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
#include <variant>
#include <vector>

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
};

DynamicMemRefT get_dynamic_memref(OpaqueMemRefT memref)
{
    DynamicMemRefT newMemref;

    switch (memref.datatype) {
    case NumericType::i1:
        newMemref.data_aligned = *((bool **)memref.descriptor);
        newMemref.data_allocated = *((bool **)memref.descriptor + 1);
        break;
    case NumericType::i16:
        newMemref.data_aligned = *((int16_t **)memref.descriptor);
        newMemref.data_allocated = *((int16_t **)memref.descriptor + 1);
        break;
    case NumericType::i32:
        newMemref.data_aligned = *((int32_t **)memref.descriptor);
        newMemref.data_allocated = *((int32_t **)memref.descriptor + 1);
        break;
    case NumericType::i64:
        newMemref.data_aligned = *((int64_t **)memref.descriptor);
        newMemref.data_allocated = *((int64_t **)memref.descriptor + 1);
        break;
    case NumericType::f32:
        newMemref.data_aligned = *((float **)memref.descriptor);
        newMemref.data_allocated = *((float **)memref.descriptor + 1);
        break;
    case NumericType::f64:
        newMemref.data_aligned = *((double **)memref.descriptor);
        newMemref.data_allocated = *((double **)memref.descriptor + 1);
        break;
    case NumericType::c64:
        newMemref.data_aligned = *((CplxT_float **)memref.descriptor);
        newMemref.data_allocated = *((CplxT_float **)memref.descriptor + 1);
        break;
    case NumericType::c128:
        newMemref.data_aligned = *((CplxT_double **)memref.descriptor);
        newMemref.data_allocated = *((CplxT_double **)memref.descriptor + 1);
        break;
    default:
        RT_FAIL("Unkown numeric type encoding for array printing.");
    }

    newMemref.offset = *(size_t *)((char *)memref.descriptor + 2 * sizeof(void *));
    size_t *sizesPtr = (size_t *)((char *)memref.descriptor + 2 * sizeof(void *) + sizeof(size_t));
    size_t *stridesPtr = (size_t *)((char *)memref.descriptor + 2 * sizeof(void *) +
                                    sizeof(size_t) + memref.rank * sizeof(size_t));

    newMemref.sizes.assign(sizesPtr, sizesPtr + memref.rank);
    newMemref.strides.assign(stridesPtr, stridesPtr + memref.rank);

    return newMemref;
}
