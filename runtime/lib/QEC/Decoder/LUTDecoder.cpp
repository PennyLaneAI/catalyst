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

// Known Limitations
// --------------------
// 1. While the current implementation can be extended to a general LUT decoding process easily,
// a few QEC code parameters are hard coded as discussed. We can expect a few changes required in
// the decoder operation defined in the MLIR layer to pass the required information from the
// compilation pass if needed.
// 2. The current implementation is tied in to the current design of the Tanner graph type in
// the MLIR. Modifications / changes might be required if there is any change in the Tanner graph
// operation definition in the MLIR layer.
// 3. The LUT decoder is not scalable in both spatial and temporal complexity.

#include "LUTDecoder.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "DataView.hpp"
#include "Exception.hpp"
#include "LUTDecoderUtils.hpp"

extern "C" {

using namespace Catalyst::Runtime::QEC;

/**
 * @brief A runtime lookup table based decoder. The current implementation applies the singleton
 * pattern for the lookup table generation. Hence, we only generate the lookup table once and the
 * successive calls to the routine defined would access the cached the lookup table directly.
 *
 * NOTE: As CAPI does not support setting default values for args, as discussed, we hardcode the
 * required args in the beginning of the function body. Those values are specifically for the [[7,
 * 1, 3]] Steane code. We expect those values are from args inputs later.
 * NOTE: The dtype of syndromes is `I1` in the MLIR layer. Following the conversion of
 * `SetBasisStateOp` to the LLVM runtime subroutine, `I1` would be converted to `int8` as well.
 * @param row_idx_tanner Pointer to the row_idx data of a Tanner graph.
 * @param col_ptr_tanner Pointer to the col_ptr data of a Tanner graph.
 * @param syndrome_results Pointer to the syndrome measurement data.
 * @param err_idx Pointer to the error qubit indices data.
 */
void __catalyst__qecp__lut_decoder(MemRefT_int32_1d *row_idx_tanner,
                                   MemRefT_int32_1d *col_ptr_tanner,
                                   MemRefT_int8_1d *current_syndromes, MemRefT_int64_1d *err_idx)
{
    // TODOs: We should expect the following const value from args.
    // The default values here only work for the [[7, 1, 3]] Steane code.
    const size_t code_size = 7;
    const size_t code_distance = 3;
    // The following parameter depends on the design choice of tanner graph would
    // change.
    const size_t aux_col_offset = 7;

    DataView<int8_t, 1> syndromes_res(current_syndromes->data_aligned, current_syndromes->offset,
                                      current_syndromes->sizes, current_syndromes->strides);
    RT_FAIL_IF(syndromes_res.size() != (code_size - 1) / 2, "Bad syndrome result input.");
    RT_FAIL_IF(err_idx->sizes[0] != (code_distance - 1) / 2, "Bad err_idx input.");

    DataView<int32_t, 1> row_idx(row_idx_tanner->data_aligned, row_idx_tanner->offset,
                                 row_idx_tanner->sizes, row_idx_tanner->strides);
    DataView<int32_t, 1> col_ptr(col_ptr_tanner->data_aligned, col_ptr_tanner->offset,
                                 col_ptr_tanner->sizes, col_ptr_tanner->strides);

    auto &current_lut = LUTs<int32_t, int64_t>::getInstance().get_lut(
        aux_col_offset, code_size, code_distance, row_idx, col_ptr);

    auto syndrome_str = convert_syndrome_res_to_bitstr<int8_t>(syndromes_res);

    std::vector<int64_t> error_indices = current_lut.at(syndrome_str);
    // Copy the inquired error indices back to the err_idx
    std::copy(error_indices.begin(), error_indices.end(), err_idx->data_aligned);
}

void __catalyst__qecp__decode_physical_measurements(MemRefT_int8_1d *physical_meas,
                                                    MemRefT_int8_1d *logical_meas)
{
    std::string err_msg_note = "Note: physical-measurement decoding currently only supports the "
                               "[[7, 1, 3]] Steane code, with logical Pauli operators acting on "
                               "physical qubits 4, 5, and 6 in the physical codeblock.";

    const size_t code_k = 1;
    const size_t code_n = 7;

    RT_FAIL_IF(physical_meas->sizes[0] != code_n,
               ("Expected input buffer of physical measurements to have size " +
                std::to_string(code_n) + ", but got " + std::to_string(physical_meas->sizes[0]) +
                err_msg_note)
                   .c_str())
    RT_FAIL_IF(logical_meas->sizes[0] != code_k,
               ("Expected output buffer of logical measurement(s) to have size " +
                std::to_string(code_k) + ", but got " + std::to_string(physical_meas->sizes[0]) +
                err_msg_note)
                   .c_str())

    DataView<int8_t, 1> dv_physical_meas(physical_meas->data_aligned, physical_meas->offset,
                                         physical_meas->sizes, physical_meas->strides);

    bool m4 = dv_physical_meas(4);
    bool m5 = dv_physical_meas(5);
    bool m6 = dv_physical_meas(6);

    int8_t result = m4 ^ m5 ^ m6;

    DataView<int8_t, 1> dv_logical_meas(logical_meas->data_aligned, logical_meas->offset,
                                        logical_meas->sizes, logical_meas->strides);

    *dv_logical_meas.begin() = result;
}

} // extern "C"
