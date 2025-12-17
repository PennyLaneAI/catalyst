// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <cstdio>

#include "NormalForms.hpp"

using namespace Catch::Matchers;
using namespace RSDecomp::NormalForms;
using namespace RSDecomp::CliffordData;

TEST_CASE("Test MA normal form", "[RSDecomp][NormalForms]")
{
    // For this test, we construct the SO3 matrix, then test the decomposition
    std::vector<std::vector<GateType>> cl_list;
    for (const auto &[gates, matrix] : clifford_group_to_SO3) {
        cl_list.push_back(gates);
    }

    // Corresponds to initial T
    bool a = GENERATE(true, false);

    // Corresponds to sequence of HT and SHT
    std::vector<bool> b = GENERATE(values<std::vector<bool>>({
        {false, false, false, false, false}, // All false
        {true, true, true, true, true},      // All true
        {true, false, false, false, false},  // Single true (start)
        {false, false, true, false, false},  // Single true (middle)
        {true, false, true, false, true},    // Alternating
        {true, true, false, false, true}     // Mixed
    }));

    // Corresponds to final Clifford
    int c = GENERATE(range(0, 24));

    std::string b_str;
    for (bool bit : b) {
        b_str += (bit ? '1' : '0');
    }

    DYNAMIC_SECTION("a = " << a << ", b = " << b_str << ", c = " << c)
    {
        SO3Matrix so3mat({}, 0);
        std::vector<GateType> expected_sol;
        if (a) {
            so3mat = SO3Matrix(DyadicMatrix({0, 0, 0, 1}, {}, {}, {0, 0, 1, 0}));
            expected_sol.push_back(GateType::T);
        }
        else {
            so3mat = SO3Matrix(DyadicMatrix({0, 0, 0, 1}, {}, {}, {0, 0, 0, 1}));
        }

        for (bool bit : b) {
            if (bit) {
                so3mat =
                    so3_matrix_mul(so3mat, SO3Matrix(DyadicMatrix({0, 0, 0, 1}, {0, 0, 1, 0},
                                                                  {0, 1, 0, 0}, {-1, 0, 0, 0}, 1)));
                expected_sol.push_back(GateType::SHT);
            }
            else {
                so3mat =
                    so3_matrix_mul(so3mat, SO3Matrix(DyadicMatrix({0, 0, 0, 1}, {0, 0, 1, 0},
                                                                  {0, 0, 0, 1}, {0, 0, -1, 0}, 1)));
                expected_sol.push_back(GateType::HT);
            }
        }

        so3mat = so3_matrix_mul(so3mat, clifford_group_to_SO3.at(cl_list[c]));
        expected_sol.insert(expected_sol.end(), cl_list[c].begin(), cl_list[c].end());

        auto result = ma_normal_form(so3mat);
        CHECK(result.first == expected_sol);
    }
}
