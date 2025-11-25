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

#include "DataView.hpp"

/**
 * This is a dummy implementation of the rs decomposition
 */
extern "C" {
int64_t rs_decomposition_get_size_0(double theta, double epsilon, bool ppr_basis)
{
    // std::cout << "Calling rs_decomposition_get_size runtime function!\n";
    // This is a dummy implementation
    (void)theta;
    (void)epsilon;
    (void)ppr_basis;
    // The dummy sequence {0, 2, 4, 6, 8, 1, 3, 5, 7, 9} has 10 elements
    return 10;
}

/**
 * @brief Fills a pre-allocated memref with the gate sequence.
 *
 * This function signature matches the standard MLIR calling convention for
 * a 1D memref, which passes the struct fields as individual arguments.
 * Note: I have tried to use `MemRefT_int64_1d` directly from Types.h, but ran into
 * C++ ABI errors (on macOS) leading to segmentation faults. Thus, we manually unpack the memref
 * here.
 *
 * @param data_allocated Pointer to allocated data
 * @param data_aligned Pointer to aligned data
 * @param offset Data offset
 * @param size0 Size of dimension 0
 * @param stride0 Stride of dimension 0
 * @param theta Angle
 * @param epsilon Error
 * @param ppr_basis Whether to use PPR basis
 */
void rs_decomposition_get_gates_0([[maybe_unused]] int64_t *data_allocated, int64_t *data_aligned,
                                  size_t offset, size_t size0, size_t stride0, double theta,
                                  double epsilon, bool ppr_basis)
{
    // --- VERIFICATION LOGS ---
    std::cout << "[rs_decomposition_get_gates] VERIFICATION LOGS:" << std::endl;
    std::cout << "  data_allocated: " << static_cast<void *>(data_allocated) << std::endl;
    std::cout << "  data_aligned:   " << static_cast<void *>(data_aligned) << std::endl;
    std::cout << "  offset:         " << offset << std::endl;
    std::cout << "  size0:          " << size0 << std::endl;
    std::cout << "  stride0:        " << stride0 << std::endl;
    std::cout << "  theta:          " << theta << std::endl;
    std::cout << "  epsilon:        " << epsilon << std::endl;
    std::cout << "  ppr_basis:      " << (ppr_basis ? "true" : "false") << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    // This is the dummy gate sequence for testing
    std::vector<int64_t> gates_data = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};

    // Re-construct the sizes and strides arrays for the DataView constructor
    const size_t sizes[1] = {size0};
    const size_t strides[1] = {stride0};

    // Wrap the memref descriptor in a DataView for access
    DataView<int64_t, 1> gates_view(data_aligned, offset, sizes, strides);

    // Ensure the MLIR-allocated buffer is at least as large as the data we're writing
    if (static_cast<size_t>(gates_view.size()) < gates_data.size()) {
        std::cerr << "Error: memref allocated for rs_decomposition is too small." << std::endl;
        return;
    }

    // Fill the memref data buffer
    for (size_t i = 0; i < gates_data.size(); ++i) {
        gates_view(i) = gates_data[i];
    }
}

/**
 * @brief Returns the global phase component of the decomposition.
 *
 * @param theta Angle
 * @param epsilon Error
 * @param ppr_basis Whether to use PPR basis
 * @return double The global phase
 */
double rs_decomposition_get_phase_0(double theta, double epsilon, bool ppr_basis)
{
    std::cout << "Calling rs_decomposition_get_phase runtime function!\n";
    std::cout << "phase got ppr_basis " << ppr_basis << std::endl;
    std::cout << "phase got theta " << theta << std::endl;
    std::cout << "phase got epsilon " << epsilon << std::endl;
    (void)theta;
    (void)epsilon;
    (void)ppr_basis;
    return 1.23;
}

} // extern "C"
