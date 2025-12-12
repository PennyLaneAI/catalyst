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

#include "NormSolver.hpp"

using namespace Catch::Matchers;
using namespace RSDecomp::NormSolver;

TEST_CASE("Test Factorization", "[RSDecomp][NormSolver]")
{
    CHECK(prime_factorize(28) == std::nullopt); //  28 = 2^2 * 7 (7 is not included)
    CHECK(prime_factorize(1) == std::vector<INT_TYPE>{});
    CHECK(prime_factorize(0) == std::vector<INT_TYPE>{});
    CHECK(prime_factorize(100) == std::vector<INT_TYPE>{2, 2, 5, 5});
    CHECK(prime_factorize(60) == std::vector<INT_TYPE>{2, 2, 3, 5});
    CHECK(prime_factorize(53) == std::vector<INT_TYPE>{53});
}

TEST_CASE("Test Integer Factorization", "[RSDecomp][NormSolver]")
{
    auto [num, valid_values] = GENERATE(table<INT_TYPE, std::vector<INT_TYPE>>({
        {28, {2, 4, 7}},
        {1, {}},
        {0, {}},
        {100, {2, 4, 5, 10, 20, 50}},
        {60, {2, 3, 4, 5, 15, 20, 30}},
        {47, {}} // 47 is prime, expect None
    }));
    std::optional<INT_TYPE> result = integer_factorize(num);

    if (valid_values.empty()) {
        CHECK_FALSE(result.has_value());
    }
    else {
        REQUIRE(result.has_value());
        bool found = std::find(valid_values.begin(), valid_values.end(), result.value()) !=
                     valid_values.end();
        CHECK(found);
    }
}

TEST_CASE("Test Factorize Prime ZSqrtTwo", "[RSDecomp][NormSolver]")
{
    auto [num, valid_values] = GENERATE(table<INT_TYPE, std::vector<ZSqrtTwo>>({
        {2, {ZSqrtTwo(0, 1), ZSqrtTwo(0, 1)}},
        {3, {ZSqrtTwo(3, 0)}},
        {7, {ZSqrtTwo(3, 1), ZSqrtTwo(3, -1)}},
        {29, {ZSqrtTwo(29, 0)}},
        {47, {ZSqrtTwo(7, 1), ZSqrtTwo(7, -1)}},
        {15, {}},
    }));
    std::optional<std::vector<ZSqrtTwo>> result = factorize_prime_zsqrt_two(num);

    if (valid_values.empty()) {
        CHECK_FALSE(result.has_value());
    }
    else {
        REQUIRE(result.has_value());
        CHECK(result == valid_values);

        ZSqrtTwo product = std::accumulate(result->begin(), result->end(), ZSqrtTwo(1, 0),
                                           std::multiplies<ZSqrtTwo>());
        CHECK(product == ZSqrtTwo(num, 0));
    }
}

TEST_CASE("Test Factorize Prime ZOmega", "[RSDecomp][NormSolver]")
{
    auto [num, valid_values] = GENERATE(table<INT_TYPE, std::vector<ZOmega>>({
        {3, {ZOmega(-1, 0, -1, -1)}}, // ADD MORE TESTS AFTER UTKARSH UPDATE
    }));
    ZSqrtTwo zsqrt_two = (factorize_prime_zsqrt_two(num).value().back());

    std::optional<ZOmega> result = factorize_prime_zomega(zsqrt_two, num);

    if (valid_values.empty()) {
        CHECK_FALSE(result.has_value());
    }
    else {
        REQUIRE(result.has_value());
        CHECK(result.value() == valid_values[0]);
    }
}

TEST_CASE("Test Primality Test", "[RSDecomp][NormSolver]")
{
    CHECK(primality_test(2) == true);
    CHECK(primality_test(4) == false);
    CHECK(primality_test(5) == true);
    CHECK(primality_test(29) == true);
    CHECK(primality_test(561) == false);  // 3*11*17
    CHECK(primality_test(1729) == false); // 7*13*19
    CHECK(primality_test(7901) == true);
    CHECK(primality_test(41041) == false); // 7*11*13*41
    CHECK(primality_test(101 * 431) == false);
}

TEST_CASE("Test Legendre Symbol", "[RSDecomp][NormSolver]")
{
    CHECK(legendre_symbol(1, 3) == 1);
    CHECK(legendre_symbol(2, 3) == 2);
    CHECK(legendre_symbol(1, 5) == 1);
    CHECK(legendre_symbol(4, 5) == 1);
    CHECK(legendre_symbol(2, 5) == 4);
    CHECK(legendre_symbol(3, 5) == 4);
    CHECK(legendre_symbol(2, 7) == 1);
    CHECK(legendre_symbol(25, 101) == 1);
}

TEST_CASE("Test Sqrt Modulo", "[RSDecomp][NormSolver]")
{
    auto [input, expected] = GENERATE(table<std::vector<INT_TYPE>, std::optional<INT_TYPE>>({
        {{3, 2}, 1},
        {{0, 1}, 0},
        {{4, 5}, 3},
        {{9, 11}, 3},
        {{16, 17}, 4},
        {{25, 29}, 24},
        {{-2, 7}, std::nullopt},
        {{-1, 6}, std::nullopt},
        {{56, 101}, 37},
        {{3, 4}, std::nullopt},
    }));

    CHECK(sqrt_modulo_p(input[0], input[1]) == expected);
    if ((expected.has_value()) && (input[1] != 2)) {
        CHECK((expected.value() * expected.value()) % input[1] == input[0]);
    }
}

TEST_CASE("Test Solve Diophantine", "[RSDecomp][NormSolver]")
{
    SECTION("Small Numbers")
    {
        // Columns: Input ZSqrtTwo, Expect Solution (bool)
        auto [input, expect_solution] = GENERATE(table<ZSqrtTwo, bool>({{ZSqrtTwo(0, 0), true},
                                                                        {ZSqrtTwo(0, 1), false},
                                                                        {ZSqrtTwo(2, 1), true},
                                                                        {ZSqrtTwo(2, -1), true},
                                                                        {ZSqrtTwo(7, 0), false},
                                                                        {ZSqrtTwo(23, 0), false},
                                                                        {ZSqrtTwo(7, 2), true},
                                                                        {ZSqrtTwo(17, 0), false},
                                                                        {ZSqrtTwo(5, 2), true},
                                                                        {ZSqrtTwo(13, 6), true}}));

        std::optional<ZOmega> result = solve_diophantine(input);

        if (!expect_solution) {
            CHECK_FALSE(result.has_value());
        }
        else {
            REQUIRE(result.has_value());
            ZOmega sol = result.value();
            ZSqrtTwo sol_norm = (sol.conj() * sol).to_sqrt_two();
            CHECK(sol_norm == input);
        }
    }

    SECTION("Large Numbers")
    {
        // Columns: "u" (used to generate input), "k" (power of 2 factor)
        // These test cases are from the paper
        auto [u, k] = GENERATE(table<ZOmega, INT_TYPE>(
            {{ZOmega(-26687414, 10541729, 10614512, 40727366), 52},
             {ZOmega(-22067493351, 22078644868, 52098814989, 16270802723), 73}}));

        INT_TYPE two_pow_k = INT_TYPE(1) << static_cast<unsigned>(k);
        ZSqrtTwo xi = ZSqrtTwo(two_pow_k, 0) - u.norm2().to_sqrt_two();

        std::optional<ZOmega> result = solve_diophantine(xi);

        REQUIRE(result.has_value());

        ZOmega sol = result.value();
        ZSqrtTwo sol_norm = (sol.conj() * sol).to_sqrt_two();
        CHECK(sol_norm == xi);
    }
}
