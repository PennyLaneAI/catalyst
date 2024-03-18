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

#include "mlir/ExecutionEngine/RunnerUtils.h"

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

template <typename T>
inline void printMemref(const UnrankedMemRefType<T> &memref, bool printDescriptor = false)
{
    auto m = DynamicMemRefType<T>(memref);
    if (printDescriptor) {
        std::cout << "MemRef: ";
        printMemRefMetaData(std::cout, m);
        std::cout << " data =" << std::endl;
    }
    impl::MemRefDataPrinter<T>::print(std::cout, m.data, m.rank, m.rank, m.offset, m.sizes,
                                      m.strides);
}
