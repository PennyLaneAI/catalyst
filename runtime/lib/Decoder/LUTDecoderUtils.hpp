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

#pragma once
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Exception.hpp"

namespace Catalyst::Runtime::QEC {
/**
 * @brief Convert a vector of syndrome results to a bit string representation.
 *
 * @tparam IntegerType
 * @param syndrome_res A vector of syndrome results
 * @return std::string A bit string representation of the given syndrome results.
 */
template <class IntegerType = std::size_t>
std::string convert_syndrome_res_to_bitstr(std::vector<IntegerType> &syndrome_res)
{
    std::string syndrom_str;
    for (const auto &bit : syndrome_res) {
        RT_ASSERT(bit == 0 || bit == 1)
        syndrom_str += (bit ? '1' : '0');
    }

    // Return results
    return syndrom_str;
}

/**
 * @brief Get a parity check matrix from the Tanner graph data.
 * The syndrome $s$ is calculated using the parity check matrix $H$ and the
 * error vector $e$ according to the linear relation:
 *
 * $$s = He \pmod 2$$
 *
 * NOTE: With the current design of Tanner graph operation in MLIR, the first $n$ or $code_size$
 * columns represent the physical data qubits, while the last $n-1$ columns represent auxillary
 * qubits (more details here
 * https://github.com/PennyLaneAI/catalyst/blob/ab97f982539b31ab802a63020292595476f22d15/mlir/include/QecPhysical/IR/QecPhysicalTypes.td).
 *
 * @param tanner_row_idx The vector of row indices of non-zero elements with a length of $nnz$.
 * @param tanner_col_ptr The column offsets vector of a length number of $num_col + 1$ that
 * represents the starting position of each column.
 * @param aux_cols A vector of column indices for the corresponding type of auxillary qubits.
 *
 * @return std::pair<std::vector<size_t>, std::vector<size_t>> The corresponding parity check matrix
 * in the CSS format. Each column represents an auxillary qubit.
 */

std::pair<std::vector<size_t>, std::vector<size_t>>
get_parity_check_matrix(const std::vector<size_t> &tanner_row_idx,
                        const std::vector<size_t> &tanner_col_ptr,
                        const std::vector<size_t> &aux_cols)
{
    std::vector<size_t> row_idx_parity;
    std::vector<size_t> col_ptr_parity{0};

    for (const auto &col : aux_cols) {
        size_t offset_start = tanner_col_ptr[col];
        size_t offset_end = tanner_col_ptr[col + 1];

        row_idx_parity.insert(row_idx_parity.end(), tanner_row_idx.begin() + offset_start,
                              tanner_row_idx.begin() + offset_end);
        size_t new_offset = col_ptr_parity.back() + offset_end - offset_start;
        col_ptr_parity.push_back(new_offset);
    }
    return {row_idx_parity, col_ptr_parity};
}

/**
 * @brief Get the bit representation of a syndrome from errors object
 * NOTE: The current implementation is experimental. Based on the design choice of the first Steane
 * code prototyping,
 *
 * @param row_idx_vec
 * @param col_ptr_vec
 * @param num_rows
 * @param num_cols
 * @return std::string
 */
std::string get_syndrome_from_errors(const std::vector<size_t> &row_idx,
                                     const std::vector<size_t> &col_ptr, const size_t num_rows,
                                     const size_t num_cols, std::vector<uint8_t> &err_vec)
{

    std::vector<size_t> syndrome_res(num_cols, 0);

    for (size_t col = 0; col < num_cols; col++) {
        for (size_t idx = col_ptr[col]; idx < col_ptr[col + 1]; idx++) {
            size_t row = row_idx[idx];
            syndrome_res[col] += err_vec[row];
        }
        syndrome_res[col] = syndrome_res[col] % 2;
    }

    // Return results
    return convert_syndrome_res_to_bitstr(syndrome_res);
}

std::vector<size_t> get_error_indices(std::vector<uint8_t> &err_vec)
{
    // Get indices of errors
    std::vector<size_t> error_indices;

    error_indices.reserve(err_vec.size());

    for (size_t i = 0; i < err_vec.size(); ++i) {
        if (err_vec[i] != 0) {
            error_indices.push_back(i);
        }
    }

    return error_indices;
}

/**
 * @brief Generates a look up table with the given parity check matrix represented in CSC format and
 * QEC code distance.
 *
 * NOTE: The current implementation of LUT generation is not limited to the $[[7,1,3]]$ Steane code.
 * The runtime cost would be expensive if the code distance/size are large.
 *
 *
 * NOTE: However, for the Steane code prototyping, the shape of dense matrix representation of the
 * Tanner graph is $(10, 10)$, instead of $(13, 13)$. It makes sense as the Steane code is a
 * self-dual CSS code, which means the connectivities of $Z-check$ auxillary qubits are identical to
 * the $X-check$ auxillary qubits.
 *
 * NOTE: This subroutine only requires the columns of either check type of auxillary qubits
 * connectivity information. Therefore, we add a function to sanitize the tanner graph to get the
 * corresponding connectivities.
 *
 * TODOs: We need to come back to update the error to syndrome mapping code when we need to support
 * non self-dual codes.
 * @param parity_mat_row_idx .The the row vector of length nnz that contains row indices of the
 * corresponding elements. Each column corresponds to an auxillary qubit.
 * @param parity_mat_col_ptr The column offsets vector of length number of num_col + 1 that
 * represents the starting position of each row.
 * @param code_size The number of data qubits in the QEC code. This param is for safe guard only
 * purpose.
 * @param code_distance The code distance, which represents the number of quantum errors can be
 * corrected.
 * @return std::unordered_map<std::string, std::vector<size_t>>&
 */
std::unordered_map<std::string, std::vector<size_t>>
generate_lookup_table(const std::vector<size_t> &parity_mat_row_idx,
                      const std::vector<size_t> &parity_mat_col_ptr, const size_t code_size,
                      const size_t code_distance)
{
    // The key here is the bitstr representation of the syndrome results, e.g., "0101"
    // The value is the corresponding indices of qubits to correct, e.g., {0, 2}.
    std::unordered_map<std::string, std::vector<size_t>> lut;

    const size_t nnz = parity_mat_row_idx.size();
    const size_t num_aux_qubits =
        parity_mat_col_ptr.size() - 1; // number of cols or number of auxillary qubits
    const size_t num_data_qubits =
        *std::max_element(parity_mat_row_idx.begin(), parity_mat_row_idx.end()) +
        1; // number of rows or number of data qubits

    RT_ASSERT(num_aux_qubits == (code_size - 1) >> 1);
    RT_ASSERT(nnz > 0);
    RT_ASSERT(num_data_qubits == code_size);

    // Get number of errors can be detected from code distance
    const size_t num_errors = (code_distance - 1) / 2;

    // Traverse all possible quantum error combinations
    for (int i = 0; i <= num_errors; i++) {
        // create a base error vector
        std::vector<uint8_t> err_vector(num_data_qubits, 0);
        std::fill(err_vector.end() - i, err_vector.end(), 1);

        do {
            std::string syndrome_str =
                get_syndrome_from_errors(parity_mat_row_idx, parity_mat_col_ptr, num_data_qubits,
                                         num_aux_qubits, err_vector);
            std::vector<size_t> error_indices = get_error_indices(err_vector);
            // We assume that 1:1 mapping for the syndrome and err_vector
            lut[syndrome_str] = error_indices;
        } while (std::next_permutation(err_vector.begin(), err_vector.end()));
    }

    return lut;
}

} // namespace Catalyst::Runtime::QEC