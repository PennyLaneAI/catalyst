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
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataView.hpp"
#include "Exception.hpp"

namespace Catalyst::Runtime::QEC {
/**
 * @brief Convert a vector of syndrome results to a bit string representation.
 *
 * @tparam IntegerType
 * @param syndrome_res A dataview of syndrome results.
 * @return std::string A bit string representation of the given syndrome results.
 */
template <typename T = int8_t>
std::string convert_syndrome_res_to_bitstr(DataView<T, 1> &syndrome_res)
{
    std::string syndrome_str;
    for (const auto &bit : syndrome_res) {
        RT_ASSERT(bit == 0 || bit == 1);
        syndrome_str += (bit ? '1' : '0');
    }

    // Return results
    return syndrome_str;
}

/**
 * @brief Get a parity check matrix from Tanner graph data.
 *
 * NOTE: With the current design of Tanner graph type in MLIR, the first $n$ or $code_size$
 * columns represent the physical data qubits, while the last $n-1$ columns represent auxiliary
 * qubits (more details in the definition of `TannerGraphType` in
 * mlir/include/QecPhysical/IR/QecPhysicalTypes.td).
 *
 * @tparam TANNER_GRAPH_INT MLIR int type
 * @param tanner_row_idx The dataview of row indices of the Tanner graph data.
 * @param tanner_col_ptr The column offsets dataview of the Tanner graph data.
 * @param aux_cols A vector of column indices for the corresponding type of auxiliary qubits.
 * @return std::pair<std::vector<int64_t>, std::vector<int64_t>> The corresponding parity check
 * matrix in the CSS format. Each column represents an auxiliary qubit.
 */
template <typename TANNER_GRAPH_INT = int32_t>
std::pair<std::vector<TANNER_GRAPH_INT>, std::vector<TANNER_GRAPH_INT>>
get_parity_check_matrix(DataView<TANNER_GRAPH_INT, 1> &tanner_row_idx,
                        DataView<TANNER_GRAPH_INT, 1> &tanner_col_ptr,
                        const std::vector<size_t> &aux_cols)
{
    std::vector<TANNER_GRAPH_INT> row_idx_parity;
    std::vector<TANNER_GRAPH_INT> col_ptr_parity{0};

    for (const auto &col : aux_cols) {
        auto offset_start = tanner_col_ptr(col);
        auto offset_end = tanner_col_ptr(col + 1);

        for (int i = offset_start; i < offset_end; i++) {
            row_idx_parity.push_back(tanner_row_idx(i));
        }
        size_t new_offset = col_ptr_parity.back() + offset_end - offset_start;
        col_ptr_parity.push_back(new_offset);
    }
    return {row_idx_parity, col_ptr_parity};
}

/**
 * @brief Get the bit representation of a syndrome from errors object
 * The syndrome $s$ is calculated using a CSC parity check matrix $H$ and the
 * error vector $e$ according to the linear relation:
 *
 * $$s = He \pmod 2$$
 *
 * @tparam TANNER_GRAPH_INT MLIR int type
 * @param row_idx The row_idx vector of $H$.
 * @param col_ptr The col_ptr vector of $H$.
 * @param num_rows Number of rows of $H$.
 * @param num_cols Number of columns of $H$.
 * @param err_vec A vector of qubit errors.
 * @return std::string The syndrome string corresponds to the err_vec.
 */
template <typename TANNER_GRAPH_INT = int32_t>
std::string get_syndrome_from_errors(const std::vector<TANNER_GRAPH_INT> &row_idx,
                                     const std::vector<TANNER_GRAPH_INT> &col_ptr,
                                     const size_t num_rows, const size_t num_cols,
                                     std::vector<int8_t> &err_vec)
{
    std::vector<size_t> syndrome_res(num_cols, 0);

    for (size_t col = 0; col < num_cols; col++) {
        for (size_t idx = static_cast<size_t>(col_ptr[col]);
             idx < static_cast<size_t>(col_ptr[col + 1]); idx++) {
            size_t row = row_idx[idx];
            syndrome_res[col] += err_vec[row];
        }
        syndrome_res[col] = syndrome_res[col] % 2;
    }
    DataView<size_t, 1> syndrome_res_data_view(syndrome_res);

    return convert_syndrome_res_to_bitstr<size_t>(syndrome_res_data_view);
}

/**
 * @brief Get the error indices of a vector of qubit errors.
 *
 * @tparam ERR_IDX_INT MLIR integer type
 * @param err_vec  A vector of qubit errors.
 * @param num_errors Number of errors that the QEC code can detect
 * @return std::vector<int64_t> Indices of qubit errors.
 */
template <typename ERR_IDX_INT = int64_t>
std::vector<ERR_IDX_INT> get_error_indices(std::vector<int8_t> &err_vec, const size_t num_errors)
{
    std::vector<ERR_IDX_INT> error_indices;

    error_indices.reserve(err_vec.size());

    for (size_t i = 0; i < err_vec.size(); ++i) {
        if (err_vec[i] != 0) {
            error_indices.push_back(i);
        }
    }

    // Case for number of errors less than those can be detected.
    // We insert -1 instead.
    size_t num_detected_errors = error_indices.size();
    if (num_detected_errors < num_errors) {
        error_indices.insert(error_indices.end(), num_errors - num_detected_errors, -1);
    }

    return error_indices;
}

/**
 * @brief Generates a look up table with a CSC parity check matrix $H$ and QEC code information.
 *
 * NOTE: Note that this function has a combinatorial time complexity of $O(n^k)$, where $n$
 * represents the number of data qubits and $k$ represents the maximum error weight. Consequently,
 * it is computationally intractable for large-scale codes.
 *
 * @tparam TANNER_GRAPH_INT MLIR int type
 * @tparam ERR_IDX_INT MLIR int type
 * @param parity_mat_row_idx The row vector of length nnz that contains row indices of the
 * corresponding elements. Each column corresponds to an X- or Z- check type auxiliary qubit.
 * @param parity_mat_col_ptr The column offsets vector of length number of num_col + 1 that
 * represents the starting position of each row.
 * @param code_size The number of data qubits in the QEC code. This param is for safe guard only
 * purpose.
 * @param code_distance The code distance, which represents the number of quantum errors can be
 * corrected.
 * @return std::unordered_map<std::string, std::vector<size_t>>& The result lookup table.
 */
template <typename TANNER_GRAPH_INT = int32_t, typename ERR_IDX_INT = int64_t>
std::unordered_map<std::string, std::vector<ERR_IDX_INT>>
generate_lookup_table(const std::vector<TANNER_GRAPH_INT> &parity_mat_row_idx,
                      const std::vector<TANNER_GRAPH_INT> &parity_mat_col_ptr,
                      const size_t code_size, const size_t code_distance)
{
    // The key here is the bitstr representation of the syndrome results, e.g., "0101"
    // The value is the corresponding indices of qubits to correct, e.g., {0, 2}.
    std::unordered_map<std::string, std::vector<ERR_IDX_INT>> lut;

    const size_t nnz = parity_mat_row_idx.size();
    const size_t num_aux_qubits = parity_mat_col_ptr.size() - 1; // number of auxiliary qubits
    const size_t num_data_qubits =
        *std::max_element(parity_mat_row_idx.begin(), parity_mat_row_idx.end()) +
        1; // number of data qubits

    RT_ASSERT(num_aux_qubits == (code_size - 1) >> 1);
    RT_ASSERT(nnz > 0);
    RT_ASSERT(num_data_qubits == code_size);

    // Get number of errors can be detected from code distance
    const size_t num_errors = (code_distance - 1) / 2;

    RT_ASSERT(num_data_qubits >= num_errors + 1)

    // Traverse all possible quantum error combinations
    for (size_t i = 0; i <= num_errors; i++) {
        // create a base error vector
        std::vector<int8_t> err_vector(num_data_qubits, 0);
        std::fill(err_vector.end() - i, err_vector.end(), 1);

        do {
            std::string syndrome_str = get_syndrome_from_errors<TANNER_GRAPH_INT>(
                parity_mat_row_idx, parity_mat_col_ptr, num_data_qubits, num_aux_qubits,
                err_vector);
            std::vector<ERR_IDX_INT> error_indices =
                get_error_indices<ERR_IDX_INT>(err_vector, num_errors);
            // We assume that 1:1 mapping for the syndrome and err_vector
            lut.try_emplace(syndrome_str, error_indices);
        } while (std::next_permutation(err_vector.begin(), err_vector.end()));
    }

    return lut;
}

/**
 * @brief Singleton design of the LUT decoder.
 *
 * @tparam TANNER_GRAPH_INT MLIR int type for Tanner graph types
 * @tparam ERR_IDX_INT MLIR int type for err_idx
 */
template <typename TANNER_GRAPH_INT = int32_t, typename ERR_IDX_INT = int64_t> class LUTs final {
  private:
    std::unordered_map<size_t, std::unordered_map<std::string, std::vector<ERR_IDX_INT>>> luts_;

    mutable std::mutex mutex_;

    explicit LUTs() = default;

  public:
    LUTs(const LUTs &) = delete;
    LUTs &operator=(const LUTs &) = delete;
    LUTs(LUTs &&) = delete;
    LUTs &operator=(LUTs &&) = delete;

    static auto getInstance() -> LUTs &
    {
        static LUTs instance;
        return instance;
    }

    /**
     * @brief Get a lookup table.
     *
     * @param aux_col_offset The offset of the first X-check or Z-check column in a Tanner graph.
     * @param code_size Number of data qubits in a QEC code.
     * @param code_distance Code distance of a QEC code.
     * @param row_idx Dataview of the row_idx of a Tanner graph.
     * @param col_ptr Dataview of the col_ptr of a Tanner graph.
     * @return const std::unordered_map<std::string, std::vector<MLIR_I64>>& The corresponding
     * lookup table.
     */
    auto get_lut(size_t aux_col_offset, size_t code_size, size_t code_distance,
                 DataView<TANNER_GRAPH_INT, 1> &row_idx, DataView<TANNER_GRAPH_INT, 1> &col_ptr)
        -> const std::unordered_map<std::string, std::vector<ERR_IDX_INT>> &
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = luts_.find(aux_col_offset);

        if (it == luts_.end()) {
            std::vector<size_t> aux_cols((code_size - 1) / 2);
            std::iota(aux_cols.begin(), aux_cols.end(), aux_col_offset);

            auto csc_parity_matrix =
                get_parity_check_matrix<TANNER_GRAPH_INT>(row_idx, col_ptr, aux_cols);

            auto lut = generate_lookup_table<TANNER_GRAPH_INT, ERR_IDX_INT>(
                csc_parity_matrix.first, csc_parity_matrix.second, code_size, code_distance);

            luts_.try_emplace(aux_col_offset, lut);
            return luts_.at(aux_col_offset);
        }

        return it->second;
    }
};

} // namespace Catalyst::Runtime::QEC
