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

#pragma once
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "Exception.hpp"

namespace Catalyst::Runtime::QEC {
/**
 * @brief Get the syndrome from errors object
 * NOTE: The current implementation is experimental. Based on the design choice of the first Steane
 * code prototyping,
 *
 *
 * @param row_idx_vec
 * @param col_ptr_vec
 * @param num_rows
 * @param num_cols
 * @return std::string
 */
std::string map_errors_to_syndrome(const std::vector<size_t> &row_idx_vec,
                                   const std::vector<size_t> &col_ptr_vec, const size_t num_rows,
                                   const size_t num_cols)
{

    std::vector<int8_t> syndrome_res(num_rows, 0);

    for (size_t row = 0; col < num_cols; col++) {
        for (size_t idx = col_ptr_vec[col]; idx < col_ptr_vec[col + 1]; idx++) {
            size_t row = row_idx_vec[idx];
            H[row][col] = 1;
        }

        // convert syndrome to string representation

        // return the syndrome bitstr
    }
}

/**
 * @brief Generates a look up table with the given parity check matrix represented in CSC format and
 * QEC code distance.
 *
 * NOTE: The current implementation of LUT generation is not limited to the [[7,1,3]] Steane code.
 * The runtime cost would be expensive if the code distance/size are large. NOTE: With the current
 * design of Tanner graph operation in MLIR, the first `n` columns represent the physical data
 * qubits, while the last `n-1` columns represent auxillary qubits (more details here
 * https://github.com/PennyLaneAI/catalyst/blob/ab97f982539b31ab802a63020292595476f22d15/mlir/include/QecPhysical/IR/QecPhysicalTypes.td).
 *
 * NOTE: However, for the Steane code prototyping, the shape of dense matrix representation of the
 * Tanner graph is `(10, 10)`, instead of `(13, 13)`. It makes sense as the Steane code is a
 * self-dual CSS code, which means the connectivities of Z-check auxillary qubits are identical to
 * the X-check auxillary qubits.
 *
 * TODOs: We need to come back to update the error to syndrome mapping code when we need to support
 * non self-dual codes.
 * @param row_idx_vec The the row vector of length nnz that contains row indices of the
 * corresponding elements.
 * @param col_ptr_vec The column offsets vector of length number of row + 1 that represents the
 * starting position of each row.
 * @param code_distance The code distance, which represents the number of quantum errors can be
 * corrected.
 * @return std::unordered_map<std::string, std::vector<int64_t>>&
 */
std::unordered_map<std::string, std::vector<int64_t>> &
generate_lookup_table(const std::vector<size_t> &row_idx_vec,
                      const std::vector<size_t> &col_ptr_vec, const size_t code_size,
                      const size_t code_distance)
{
    // The key here is the bitstr representation of the syndrome results, e.g., "0101"
    // The value is the corresponding indices of qubits to correct, e.g., {0, 2}.
    std::unordered_map<std::string, std::vector<int64_t>> lut;

    const size_t nnz = row_idx_vec.size();
    const size_t num_cols = col_ptr_vec.size() - 1; // number of cols
    const size_t num_rows =
        *std::max_element(row_idx_vec.begin(), row_idx_vec.end()) + 1; // number of rows
    RT_ASSERT(num_cols > 0);
    RT_ASSERT(nnz > 0);
    RT_ASSERT(num_rows > 0);

    // Get number of errors can be detected from code distance
    const size_t num_errors = (code_distance - 1) / 2;

    // Traverse all possible quantum error combinations
    for (int i = 0; i <= num_errors; i++) {
        // create a base error vector
        std::vector<int8_t> err_vector(code_size, 0);
        std::fill_n(err_vector.begin(), i, 1);

        do {
            // Compute the sydrome corresponding to the current error vector
            // by multiply the parity check matrix H with current_error vector

            // Get the indices of non-zero elements in the current vector

            // Convert the syndrome to str
        } while (std::next_permutation(err_vector.begin(), err_vector.end()))
    }

    return lut;
}

} // namespace Catalyst::Runtime::QEC