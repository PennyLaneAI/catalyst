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

#include "LUTDecoder.hpp"

#include <algorithm>
#include <ranges>
#include <vector>

#include "DataView.hpp"
#include "LUTDecoderUtils.hpp"

namespace Catalyst::Runtime::QEC {
/**
 * @brief
 *
 * @param row_idx_tanner
 * @param col_ptr_tanner
 * @param syndrome_results
 * @return size_t
 */
void __catalyst__qecp__lut_decoder(MemRefT_int64_1d *row_idx_tanner,
                                   MemRefT_int64_1d *col_ptr_tanner,
                                   MemRefT_int8_1d *current_syndromes, MemRefT_int64_1d *err_idx)
{
    // 1. Recover the parity check matrix from a tanner graph
    const size_t nnz = row_idx_tanner->sizes[0];
    std::vector<size_t> row_idx_tanner_vec(row_idx_tanner->data_aligned,
                                           row_idx_tanner->data_aligned + nnz);

    const size_t n = col_ptr_tanner->sizes[0] - 1; // number of columns
    std::vector<size_t> col_ptr_tanner_vec(col_ptr_tanner->data_aligned,
                                           col_ptr_tanner->data_aligned + n + 1);
    const size_t num_aux = current_syndromes->sizes[0]; // number of columns
    std::vector<int8_t> current_syndromes_res(current_syndromes->data_aligned,
                                              current_syndromes->data_aligned + num_aux);

    // TODOs: Hardcoded for the [[7,1,3]] Steane code and current design choice of Tanner graph for
    // self-dual CSS code.
    const std::vector<size_t> aux_cols{7, 8, 9};

    auto csc_parity_matrix =
        get_parity_check_matrix(row_idx_tanner_vec, col_ptr_tanner_vec, aux_cols);
    auto row_idx_parity = csc_parity_matrix.first;
    auto col_ptr_parity = csc_parity_matrix.second;

    // Generate LUT from parity check matrix
    // TODOs: We should expect the follow code_size and code_distance info from the compilation.
    const size_t code_size = 7;
    const size_t code_distance = 3;

    auto lut = generate_lookup_table(row_idx_parity, col_ptr_parity, code_size, code_distance);

    auto current_syndrome_str = convert_syndrome_res_to_bitstr(current_syndromes_res);

    std::vector<size_t> error_indices = lut[current_syndrome_str];

    // We use `-1` to full fill the err_idx array if the number of
    // errors is less than (code_distance - 1)/2
    for (int i = 0; i < (code_distance - 1) / 2; i++) {
        if (i < error_indices.size()) {
            err_idx->data_allocated[i] = error_indices[i];
        }
        else {
            err_idx->data_allocated[i] = -1;
        }
    }
}
} // namespace Catalyst::Runtime::QEC
