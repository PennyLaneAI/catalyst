// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <vector>

#include "DataView.hpp"
#include "lut_decoder.hpp"

namespace Catalyst::Runtime::QEC {
int64_t *__catalyst__qecp__decode_steane_lut(MemRefT_int64_1d *row_idx, MemRefT_int64_1d *col_ptr,
                                             MemRefT_int8_1d *syndrome_results)
{
    // 1. Recover the parity check matrix from the CSC sparse representation.
    // Get shapes of the dense parity check matrix (H) from the input sparse representation.
    const size_t nnz = row_idx->sizes[0]; // number of non-zero elements
    std::vector<size_t> row_idx_vec(row_idx->data_aligned, row_idx->data_aligned + nnz);

    const size_t n = col_ptr->sizes[0] - 1; // number of columns
    std::vector<size_t> col_ptr_vec(col_ptr->data_aligned, col_ptr->data_aligned + n + 1);
    const size_t m =
        *std::max_element(row_idx_vec.begin(), row_idx_vec.end()) + 1; // number of rows

    // Construct the parity check matrix (H) from the sparse representation.
    std::vector<std::vector<uint8_t>> H(m, std::vector<uint8_t>(n, 0));
    for (size_t col = 0; col < n; col++) {
        for (size_t idx = col_ptr_vec[col]; idx < col_ptr_vec[col + 1]; idx++) {
            size_t row = row_idx_vec[idx];
            H[row][col] = 1;
        }
    }
    // 2. Create a look up table using a singleton pattern, which would only be initialized once
    // during the first call to the function, and would be reused for subsequent calls.

    // NOTE: We need to use std::next_permutation to iterate over all possible weight-w errors,
    // which is not straightforward to implement in a constexpr function. Hence we use a static
    // variable to store the look up table, which is initialized at runtime during the first call to
    // the function.

    // NOTE: The look up table is an unordered_map<str, std::vector<int64_t>>, whose key is the str
    // representation of sydrome results, and value is the corresponding the indices of qubits to
    // correct.

    // NOTE: We need two separate look up tables for X and Z errors, since they are decoded using
    // different parity check matrices (H_X and H_Z). This implies that we need to implement
    // separate singleton functions to initialize the look up tables for X and Z errors? I believe
    // so.

    std::vector<std::vector<uint8_t>> lut(1 << m, std::vector<uint8_t>(n, 0));

    // 3. Convert the given syndrome results (represented as int8_t) into a string representation,
    // which can be used as the key to look up the look up table.

    // 4. Return the look up table entry corresponding to the given syndrome results

    return nullptr;
}
} // namespace Catalyst::Runtime::QEC