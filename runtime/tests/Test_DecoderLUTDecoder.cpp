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

#include <string>
#include <unordered_map>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include "LUTDecoder.hpp"
#include "Types.h"

using namespace Catalyst::Runtime::QEC;

struct tanner_graph_steane0 {
    /* Tanner graph representation for the [[7, 1, 3]] Steane code
       The shape of dense matrix that the [[7, 1, 3]] Steane code is (10, 10).
       The first 7 columns represent data qubits, while the last 3 columns
       represent auxillary qubits. The full dense matrix is:
       | 0 0 0 0 0 0 0 1 0 0|
       | 0 0 0 0 0 0 0 1 1 0|
       | 0 0 0 0 0 0 0 1 1 1|
       | 0 0 0 0 0 0 0 1 0 1|
       | 0 0 0 0 0 0 0 0 1 0|
       | 0 0 0 0 0 0 0 0 1 1|
       | 0 0 0 0 0 0 0 0 0 1|
       | 1 1 1 1 0 0 0 0 0 0|
       | 0 1 1 0 1 1 0 0 0 0|
       | 0 0 1 1 0 1 1 0 0 0|
    */
    const size_t code_size = 7;
    const size_t code_distance = 3;
    const std::vector<int64_t> row_idx = {7, 7, 8, 7, 8, 9, 7, 9, 8, 8, 9, 9,
                                          0, 1, 2, 3, 1, 2, 4, 5, 2, 3, 5, 6};
    const std::vector<int64_t> col_ptr = {0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 24};

    const std::vector<int64_t> row_idx_parity_matrix_transpose = {0, 1, 2, 3, 1, 2,
                                                                  4, 5, 2, 3, 5, 6};
    const std::vector<int64_t> col_ptr_parity_matrix_transpose = {0, 4, 8, 12};

    const std::unordered_map<int64_t, std::vector<int8_t>> lookup_table = {
        {7, std::vector<int8_t>({0, 0, 0})}, {0, std::vector<int8_t>({0, 0, 1})},
        {1, std::vector<int8_t>({0, 1, 0})}, {2, std::vector<int8_t>({0, 1, 1})},
        {3, std::vector<int8_t>({1, 0, 0})}, {4, std::vector<int8_t>({1, 0, 1})},
        {5, std::vector<int8_t>({1, 1, 0})}, {6, std::vector<int8_t>({1, 1, 1})},

    };
} tanner_graph0;

TEST_CASE("Test C-API Wrapper (Memref Interface)", "[LUTDecoder][lut_decoder]")
{

    // const std::vector<double> param{0.3, 0.7, 0.4};

    // std::vector<int64_t> trainParams{0, 1, 2};
    // size_t J = trainParams.size();
    // double *buffer = new double[J];
    // MemRefT_double_1d result = {buffer, buffer, 0, {J}, {1}};
    // double *buffer_tp = new double[J];
    // MemRefT_double_1d result_tp = {buffer_tp, buffer_tp, 0, {J}, {1}};
    // int64_t *buffer_tp_memref = trainParams.data();
    // MemRefT_int64_1d tp_memref = {buffer_tp_memref, buffer_tp_memref, 0, {trainParams.size()},
    // {1}};

    std::vector<int64_t> row_idx_tanner = tanner_graph0.row_idx;
    std::vector<int64_t> col_ptr_tanner = tanner_graph0.col_ptr;

    int64_t *buffer_row_idx_tanner_memref = row_idx_tanner.data();
    int64_t *buffer_col_ptr_tanner_memref = col_ptr_tanner.data();

    MemRefT_int64_1d row_idx_tanner_memref = {buffer_row_idx_tanner_memref,
                                              buffer_row_idx_tanner_memref,
                                              0,
                                              {row_idx_tanner.size()},
                                              {1}};
    MemRefT_int64_1d col_ptr_tanner_memref = {buffer_col_ptr_tanner_memref,
                                              buffer_col_ptr_tanner_memref,
                                              0,
                                              {col_ptr_tanner.size()},
                                              {1}};

    for (auto it = tanner_graph0.lookup_table.begin(); it != tanner_graph0.lookup_table.end(); ++it) {
        size_t expected_res = it->first;

        auto syndrome_res = it->second;

        int8_t *buffer_syndrome_res_memref = syndrome_res.data();
        MemRefT_int8_1d syndrome_res_memref = {
            buffer_syndrome_res_memref, buffer_syndrome_res_memref, 0, {syndrome_res.size()}, {1}};

        size_t decoded_res = __catalyst__qecp__lut_decoder(
            &row_idx_tanner_memref, &col_ptr_tanner_memref, &syndrome_res_memref);

        REQUIRE(decoded_res == expected_res);
    }
}
