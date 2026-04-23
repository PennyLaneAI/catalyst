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

#include "LUTDecoderUtils.hpp"
using namespace Catalyst::Runtime::QEC;

struct tanner_graph_steane {
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
    const std::vector<size_t> row_idx = {7, 7, 8, 7, 8, 9, 7, 9, 8, 8, 9, 9,
                                         0, 1, 2, 3, 1, 2, 4, 5, 2, 3, 5, 6};
    const std::vector<size_t> col_ptr = {0, 1, 3, 6, 8, 9, 11, 12, 16, 20, 24};

    const std::vector<size_t> row_idx_parity_matrix_transpose = {0, 1, 2, 3, 1, 2,
                                                                 4, 5, 2, 3, 5, 6};
    const std::vector<size_t> col_ptr_parity_matrix_transpose = {0, 4, 8, 12};

    const std::unordered_map<std::string, std::vector<uint8_t>> lookup_table = {
        {"000", std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 0})},
        {"001", std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 1})},
        {"010", std::vector<uint8_t>({0, 0, 0, 0, 1, 0, 0})},
        {"011", std::vector<uint8_t>({0, 0, 0, 0, 0, 1, 0})},
        {"100", std::vector<uint8_t>({1, 0, 0, 0, 0, 0, 0})},
        {"101", std::vector<uint8_t>({0, 0, 0, 1, 0, 0, 0})},
        {"110", std::vector<uint8_t>({0, 1, 0, 0, 0, 0, 0})},
        {"111", std::vector<uint8_t>({0, 0, 1, 0, 0, 0, 0})},

    };
} tanner_graph;

TEST_CASE("Test convert_sydrome_res_to_bitstr", "[LUTDecoderUtils::convert_syndrome_res_to_bitstr]")
{
    std::vector<size_t> bad_syndrome_inputs = {1, 2, 3};
    REQUIRE_THROWS_WITH(convert_syndrome_res_to_bitstr(bad_syndrome_inputs),
                        Catch::Matchers::ContainsSubstring("Assertion: bit == 0 || bit == 1"));

    std::vector<size_t> syndromes_size_t = {0, 1, 0};
    std::vector<int8_t> syndromes_int8_t = {0, 1, 0};
    std::string expected_syndrome_str = "010";
    std::string syndrome_str_size_t = convert_syndrome_res_to_bitstr(syndromes_size_t);
    std::string syndrome_str_int8_t = convert_syndrome_res_to_bitstr(syndromes_int8_t);

    REQUIRE(syndrome_str_size_t == expected_syndrome_str);
    REQUIRE(syndrome_str_int8_t == expected_syndrome_str);
}

TEST_CASE("Test get_error_indices", "[LUTDecoderUtils::get_error_indices]")
{
    std::vector<u_int8_t> error_vector = {0, 1, 0, 1, 0, 0, 0};
    std::vector<size_t> expected_indices = {1, 3};

    auto error_indices = get_error_indices(error_vector);

    REQUIRE(error_indices == expected_indices);
}

TEST_CASE("Test get_parity_check_matrix", "[LUTDecoderUtils::get_parity_check_matrix]")
{

    std::vector<size_t> aux_cols = {7, 8, 9};
    auto parity_mat_csc =
        get_parity_check_matrix(tanner_graph.row_idx, tanner_graph.col_ptr, aux_cols);

    REQUIRE(parity_mat_csc.first == tanner_graph.row_idx_parity_matrix_transpose);
    REQUIRE(parity_mat_csc.second == tanner_graph.col_ptr_parity_matrix_transpose);
}

TEST_CASE("Test get_syndrome_from_errors", "[LUTDecoderUtils::get_syndrome_from_errors]")
{

    std::vector<size_t> aux_cols = {7, 8, 9};
    auto parity_mat_csc =
        get_parity_check_matrix(tanner_graph.row_idx, tanner_graph.col_ptr, aux_cols);

    const size_t num_data_qubits = tanner_graph.code_size;
    const size_t num_aux_qubits = 3;
    for (auto it = tanner_graph.lookup_table.begin(); it != tanner_graph.lookup_table.end(); ++it) {
        auto err_vec = it->second;
        std::string expected_str = it->first;
        std::string syndrome_bitstr = get_syndrome_from_errors(
            parity_mat_csc.first, parity_mat_csc.second, num_data_qubits, num_aux_qubits, err_vec);
        REQUIRE(syndrome_bitstr == expected_str);
    }
}

TEST_CASE("Test generate_lookup_table", "[LUTDecoderUtils::generate_lookup_table]")
{
    auto lut = generate_lookup_table(tanner_graph.row_idx_parity_matrix_transpose,
                                     tanner_graph.col_ptr_parity_matrix_transpose,
                                     tanner_graph.code_size, tanner_graph.code_distance);

    std::unordered_map<std::string, std::vector<size_t>> expected_lut = {
        {"000", std::vector<size_t>({})},  {"001", std::vector<size_t>({6})},
        {"010", std::vector<size_t>({4})}, {"011", std::vector<size_t>({5})},
        {"100", std::vector<size_t>({0})}, {"101", std::vector<size_t>({3})},
        {"110", std::vector<size_t>({1})}, {"111", std::vector<size_t>({2})},
    };

    for (auto it = expected_lut.begin(); it != expected_lut.end(); ++it) {
        auto key = it->first;
        REQUIRE(lut[key] == it->second);
    }
}
