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
#include "catch2/matchers/catch_matchers_string.hpp"

#include "LUTDecoder.hpp"
#include "TestUtils.hpp"
#include "Types.h"

using namespace Catalyst::Runtime::QEC;

using TANNER_GRAPH_INT = int32_t;
using ERR_IDX_INT = int64_t;
using SYNDROME_INT = int8_t;
using MemRefT_Tanner = MemRefT_int32_1d;
using MemRefT_Err_Idx = MemRefT_int64_1d;
using MemRefT_Syndrome = MemRefT_int8_1d;

TEST_CASE("Test C-API Wrapper (Memref Interface)", "[LUTDecoder][lut_decoder]")
{
    tanner_graph_steane<TANNER_GRAPH_INT> tanner_graph;

    std::vector<TANNER_GRAPH_INT> row_idx_tanner = tanner_graph.row_idx;
    std::vector<TANNER_GRAPH_INT> col_ptr_tanner = tanner_graph.col_ptr;
    std::vector<ERR_IDX_INT> err_idx =
        std::vector<int64_t>((tanner_graph.code_distance - 1) / 2, -1);

    TANNER_GRAPH_INT *buffer_row_idx_tanner_memref = row_idx_tanner.data();
    TANNER_GRAPH_INT *buffer_col_ptr_tanner_memref = col_ptr_tanner.data();
    ERR_IDX_INT *buffer_err_idx_memref = err_idx.data();

    MemRefT_Tanner row_idx_tanner_memref = {buffer_row_idx_tanner_memref,
                                            buffer_row_idx_tanner_memref,
                                            0,
                                            {row_idx_tanner.size()},
                                            {1}};
    MemRefT_Tanner col_ptr_tanner_memref = {buffer_col_ptr_tanner_memref,
                                            buffer_col_ptr_tanner_memref,
                                            0,
                                            {col_ptr_tanner.size()},
                                            {1}};
    MemRefT_Err_Idx err_idx_memref = {
        buffer_err_idx_memref, buffer_err_idx_memref, 0, {err_idx.size()}, {1}};

    SECTION("Syndrome results that can be decoded.")
    {
        for (auto it = tanner_graph.lookup_table_error_idx_to_syndrome.begin();
             it != tanner_graph.lookup_table_error_idx_to_syndrome.end(); ++it) {
            ERR_IDX_INT expected_res = it->first;

            auto syndrome_res = it->second;

            SYNDROME_INT *buffer_syndrome_res_memref = syndrome_res.data();
            MemRefT_Syndrome syndrome_res_memref = {buffer_syndrome_res_memref,
                                                    buffer_syndrome_res_memref,
                                                    0,
                                                    {syndrome_res.size()},
                                                    {1}};

            __catalyst__qecp__lut_decoder(&row_idx_tanner_memref, &col_ptr_tanner_memref,
                                          &syndrome_res_memref, &err_idx_memref);

            REQUIRE(err_idx_memref.data_aligned[0] == expected_res);
        }
    }

    SECTION("Error raised for bad syndrome res.")
    {
        std::vector<SYNDROME_INT> bad_syndrome_res = {1, 1};

        SYNDROME_INT *buffer_bad_syndrome_res_memref = bad_syndrome_res.data();
        MemRefT_Syndrome bad_syndrome_res_memref = {buffer_bad_syndrome_res_memref,
                                                    buffer_bad_syndrome_res_memref,
                                                    0,
                                                    {bad_syndrome_res.size()},
                                                    {1}};
        REQUIRE_THROWS_WITH(
            __catalyst__qecp__lut_decoder(&row_idx_tanner_memref, &col_ptr_tanner_memref,
                                          &bad_syndrome_res_memref, &err_idx_memref),
            Catch::Matchers::ContainsSubstring("Bad syndrome result input."));
    }

    SECTION("Error raised for bad err_idx input.")
    {
        std::vector<ERR_IDX_INT> bad_err_idx =
            std::vector<ERR_IDX_INT>((tanner_graph.code_distance - 1), -1);
        ERR_IDX_INT *buffer_bad_err_idx_memref = bad_err_idx.data();
        MemRefT_Err_Idx bad_err_idx_memref = {
            buffer_bad_err_idx_memref, buffer_bad_err_idx_memref, 0, {bad_err_idx.size()}, {1}};

        std::vector<SYNDROME_INT> syndrome_res = {1, 1, 1};
        SYNDROME_INT *buffer_syndrome_res_memref = syndrome_res.data();
        MemRefT_Syndrome syndrome_res_memref = {
            buffer_syndrome_res_memref, buffer_syndrome_res_memref, 0, {syndrome_res.size()}, {1}};

        REQUIRE_THROWS_WITH(
            __catalyst__qecp__lut_decoder(&row_idx_tanner_memref, &col_ptr_tanner_memref,
                                          &syndrome_res_memref, &bad_err_idx_memref),
            Catch::Matchers::ContainsSubstring("Bad err_idx input."));
    }
}
