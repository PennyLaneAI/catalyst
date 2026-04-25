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
#include <numeric>
#include <ranges>
#include <vector>

#include "DataView.hpp"
#include "LUTDecoderUtils.hpp"

namespace Catalyst::Runtime::QEC {
/**
 * @brief A runtime lookup table based decoder. This function uses tanner graph data in CSC format
 * information to generate the lookup table and find out the corresponding error qubit indices with
 * the syndrome data.
 * NOTE: As CAPI does not support setting default values for args, as discussed, we hardcode the
 * required args in the beginning of the function body. Those values are specifically for the [[7,
 * 1, 3]] Steane code. We expect those values are from args inputs later.
 * @param row_idx_tanner Pointer to the row_idx data of the Tanner graph of the QEC code.
 * @param col_ptr_tanner Pointer to the col_ptr data of the Tanner graph of the QEC code.
 * @param syndrome_results Pointer to the syndrome measurement data.
 * @param err_idx Pointer to the error qubit indices data.
 */
void __catalyst__qecp__lut_decoder(MemRefT_int64_1d *row_idx_tanner,
                                   MemRefT_int64_1d *col_ptr_tanner,
                                   MemRefT_int8_1d *current_syndromes, MemRefT_int64_1d *err_idx)
{
    // TODOs: We should expect the following const value from args.
    // The default values here only work for the [[7, 1, 3]] Steane code.
    const size_t code_size = 7;
    const size_t code_distance = 3;
    // The following parameter depends on the design choice of tanner graph would
    // change.
    const size_t aux_col_offset = 7;

    DataView<int64_t, 1> row_idx(row_idx_tanner->data_aligned, row_idx_tanner->offset,
                                 row_idx_tanner->sizes, row_idx_tanner->strides);
    DataView<int64_t, 1> col_ptr(col_ptr_tanner->data_aligned, col_ptr_tanner->offset,
                                 col_ptr_tanner->sizes, col_ptr_tanner->strides);

    // auto &&luts = LUTs::getInstance();

    // auto current_lut = luts.get_lut(aux_col_offset, code_size, code_distance, row_idx, col_ptr);

    auto current_lut =
        LUTs::getInstance().get_lut(aux_col_offset, code_size, code_distance, row_idx, col_ptr);

    DataView<int8_t, 1> syndromes_res(current_syndromes->data_aligned, current_syndromes->offset,
                                      current_syndromes->sizes, current_syndromes->strides);

    auto syndrome_str = convert_syndrome_res_to_bitstr<int8_t>(syndromes_res);

    std::vector<int64_t> error_indices = current_lut[syndrome_str];

    // We use `-1` to full fill the err_idx array if the number of
    // errors is less than (code_distance - 1)/2
    for (size_t i = 0; i < (code_distance - 1) / 2; i++) {
        if (i < error_indices.size()) {
            err_idx->data_allocated[i] = error_indices[i];
        }
        else {
            err_idx->data_allocated[i] = -1;
        }
    }
}
} // namespace Catalyst::Runtime::QEC
