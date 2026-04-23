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

#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include "LUTDecoderUtils.hpp"
using namespace Catalyst::Runtime::QEC;
TEST_CASE("Test convert_sydrome_res_to_bitstr", "[LUTDecoderUtils::syndrome_res_convert]")
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