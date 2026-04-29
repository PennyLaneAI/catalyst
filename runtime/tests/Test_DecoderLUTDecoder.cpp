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

#include <unordered_map>
#include <vector>

#include "catch2/catch_test_macros.hpp"

#include "LUTDecoder.hpp"
#include "TestUtils.hpp"
#include "Types.h"

using namespace Catalyst::Runtime::QEC;

TEST_CASE("Test C-API Wrapper (Memref Interface)", "[LUTDecoder][lut_decoder]")
{
    tanner_graph_steane<int64_t> tanner_graph;

    std::vector<int64_t> row_idx_tanner = tanner_graph.row_idx;
    std::vector<int64_t> col_ptr_tanner = tanner_graph.col_ptr;
    std::vector<int64_t> err_idx = std::vector<int64_t>((tanner_graph.code_distance - 1) / 2, -1);

    int64_t *buffer_row_idx_tanner_memref = row_idx_tanner.data();
    int64_t *buffer_col_ptr_tanner_memref = col_ptr_tanner.data();
    int64_t *buffer_err_idx_memref = err_idx.data();

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
    MemRefT_int64_1d err_idx_memref = {
        buffer_err_idx_memref, buffer_err_idx_memref, 0, {err_idx.size()}, {1}};

    for (auto it = tanner_graph.lookup_table_error_idx_to_syndrome.begin();
         it != tanner_graph.lookup_table_error_idx_to_syndrome.end(); ++it) {
        int64_t expected_res = it->first;

        auto syndrome_res = it->second;

        int8_t *buffer_syndrome_res_memref = syndrome_res.data();
        MemRefT_int8_1d syndrome_res_memref = {
            buffer_syndrome_res_memref, buffer_syndrome_res_memref, 0, {syndrome_res.size()}, {1}};

        __catalyst__qecp__lut_decoder(&row_idx_tanner_memref, &col_ptr_tanner_memref,
                                      &syndrome_res_memref, &err_idx_memref);

        REQUIRE(err_idx_memref.data_allocated[0] == expected_res);
    }
}
