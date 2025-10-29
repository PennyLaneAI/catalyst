// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

/**
 * This is a dummy implementation of the rs decomposition
 */

// 1D memref descriptor
struct MemRef1D {
    int64_t *allocated;
    int64_t *aligned;
    int64_t offset;
    int64_t size;
    int64_t stride;
};

extern "C" {

// Decomposition body
MemRef1D rs_decomposition_0(double param)
{
    std::cout << "Calling rs_decomposition runtime function!\n";
    // This returns a dummy gate sequence for testing
    std::vector<int64_t> gates_data = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
    int64_t num_gates = gates_data.size();
    int64_t *heap_data = new int64_t[num_gates];
    memcpy(heap_data, gates_data.data(), num_gates * sizeof(int64_t));

    std::cout << "param received = " << param << "\n";
    std::cout << "heap_data address: " << static_cast<void *>(heap_data) << "\n";

    MemRef1D result;
    result.allocated = heap_data;
    result.aligned = heap_data;
    result.offset = 0;
    result.size = num_gates;
    result.stride = 1;
    return result;
}

// Will be used to return global phase
double some_func_get_val1_0() { return 1.23; }

// Free memref
void free_memref_0(int64_t *allocated, int64_t *aligned, int64_t offset, int64_t size,
                   int64_t stride)
{
    // Mark other args as unused to prevent compiler warnings
    (void)aligned;
    (void)offset;
    (void)size;
    (void)stride;

    std::cout << "free_memref_0 called\n";
    std::cout << "deleting heap_data at: " << static_cast<void *>(allocated) << "\n";
    delete[] allocated;
}

} // extern "C"
