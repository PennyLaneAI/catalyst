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
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdint>
#include <cstdio>

#include "rings.hpp"


using namespace Catch::Matchers;

TEST_CASE("Test ZSqrtTwo class", "[RSDecomp][Rings]")
{
    ZSqrtTwo z1(1, 2);
    ZSqrtTwo z2(3, 4);

    CHECK(z1.a == 1);
    CHECK(z1.b == 2);

    ZSqrtTwo z_add = z1 + z2;
    CHECK(z_add == ZSqrtTwo(4, 6));

    ZSqrtTwo z_sub = z1 - z2;
    CHECK(z_sub == ZSqrtTwo(-2, -2));

    ZSqrtTwo z_mul = z1 * z2;
    CHECK(z_mul == ZSqrtTwo(19, 10));

    ZSqrtTwo z_mul_scalar = z2 * 10;
    CHECK(z_mul_scalar == ZSqrtTwo(30, 40));

    ZSqrtTwo z_div = ZSqrtTwo(14, 7) / z1;
    CHECK(z_div == ZSqrtTwo(2, 3));

    ZSqrtTwo z_div_scalar = ZSqrtTwo(4, 2) / 2;
    CHECK(z_div_scalar == ZSqrtTwo(2, 1));

    REQUIRE_THROWS_WITH(ZSqrtTwo(1, 1) / 0, ContainsSubstring("Division by zero"));

    REQUIRE_THROWS_WITH(ZSqrtTwo(3, 1) / 2, ContainsSubstring("Non-integer division result"));

    CHECK(z1.to_double() == 1.0 + 2.0 * M_SQRT2);

    CHECK(z2.adj2() == ZSqrtTwo(3, -4));

    CHECK(z1.adj2() == ZSqrtTwo(1, -2));

    CHECK(z1.abs() == -7);

    CHECK(z2.abs() == -23);


}
