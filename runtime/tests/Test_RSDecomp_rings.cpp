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

#include "rings.cpp"


using namespace Catch::Matchers;

TEST_CASE("Test ZSqrtTwo class", "[RSDecomp][Rings]")
{
    ZSqrtTwo z1(1, 2);
    ZSqrtTwo z2(3, 4);

    CHECK(z1.a == 1);
    CHECK(z1.b == 2);

    CHECK((z1 + z2) == ZSqrtTwo(4, 6));

    CHECK((z1 - z2) == ZSqrtTwo(-2, -2));

    CHECK((z1 * z2) == ZSqrtTwo(19, 10));

    CHECK((z2 * 10) == ZSqrtTwo(30, 40));

    CHECK((ZSqrtTwo(14, 7) / z1) == ZSqrtTwo(2, 3));

    CHECK((ZSqrtTwo(4, 2) / 2) == ZSqrtTwo(2, 1));

    CHECK((z2 % ZSqrtTwo(2, 0)) == ZSqrtTwo(-1, 0));

    REQUIRE_THROWS_WITH(ZSqrtTwo(1, 1) / 0, ContainsSubstring("Division by zero"));
    REQUIRE_THROWS_WITH(ZSqrtTwo(3, 1) / 2, ContainsSubstring("Non-integer division result"));

    CHECK(z1.to_double() == 1.0 + 2.0 * M_SQRT2);

    CHECK(z2.adj2() == ZSqrtTwo(3, -4));
    CHECK(z1.adj2() == ZSqrtTwo(1, -2));

    CHECK(z1.abs() == -7);
    CHECK(z2.abs() == -23);

    CHECK(z1.pow(0) == ZSqrtTwo(1, 0));
    CHECK(z1.pow(2) == ZSqrtTwo(9, 4));

    CHECK(z1.sqrt().has_value() == false);
    CHECK(z2.sqrt().has_value() == false);

    CHECK(z1.pow(2).sqrt() == z1);

    CHECK(z1.to_omega() == ZOmega(-2, 0, 2, 1));
    CHECK(z2.to_omega() == ZOmega(-4, 0, 4, 3));
}


TEST_CASE("Test ZOmega class", "[RSDecomp][Rings]")
{
    const double tol = 1e-10;
    ZOmega z1(1, 2, 3, 4);
    ZOmega z2(5, 6, 7, 8);

    CHECK(z1.a == 1);
    CHECK(z1.b == 2);
    CHECK(z1.c == 3);
    CHECK(z1.d == 4);

    CHECK((z1 + z2) == ZOmega(6, 8, 10, 12));
    CHECK((z1 - z2) == ZOmega(-4, -4, -4, -4));
    CHECK((z1 * z2) == ZOmega(60, 56, 36, -2));
    CHECK((z1 * 10) == ZOmega(10, 20, 30, 40));
    REQUIRE_THROWS_WITH(z2 / 2, ContainsSubstring("Non-integer division result"));
    CHECK((ZOmega(2, 4, 6, 8) / 2) == z1);

    CHECK((ZOmega(1283, 130, 3092, 3091) % ZOmega(44, 67, 91, 3))== ZOmega(-24, -11, 10, 7));
    CHECK(z1.abs() == 388);
    CHECK(z2.abs() == 14788);

    auto z1_complex = z1.to_complex();
    CHECK(z1_complex.real() == Catch::Approx(5.41421356237309).margin(tol));
    CHECK(z1_complex.imag() == Catch::Approx(4.82842712474619).margin(tol));
    CHECK(z1.parity() == 0);
    CHECK(z2.parity() == 0);
    CHECK(ZOmega(0, 0, 1, 1).parity() == 1);

    CHECK(z1.adj2() == ZOmega(-1, 2, -3, 4));
    CHECK(z2.adj2() == ZOmega(-5, 6, -7, 8));

    ZOmega z1_conj = z1.conj();
    CHECK(z1_conj == ZOmega(-3, -2, -1, 4));
    CHECK(z1_conj.to_complex().real() == Catch::Approx(std::conj(z1_complex).real()).margin(tol));
    CHECK(z1_conj.to_complex().imag() == Catch::Approx(std::conj(z1_complex).imag()).margin(tol));

    CHECK(z2.norm().to_complex().real() == Catch::Approx((std::abs(z2.to_complex()) * std::abs(z2.conj().to_complex()))).margin(tol));

    CHECK((z1 - ZOmega(2, 2, 2, 0)).to_sqrt_two() == ZSqrtTwo(4, 1));    
}


TEST_CASE("Test DyadicMatrix class", "[RSDecomp][Rings]")
{
    ZOmega z1 = ZOmega(1, 2, 3, 4);
    ZOmega z2 = ZOmega(5, 6, 7, 8);
    DyadicMatrix m1{z1, z2, z1, z2};

    auto x = m1 * 2;
    auto y = DyadicMatrix(z1 * 2, z2 * 2, z1 * 2, z2 * 2);
    bool equal = (x == y);
    CHECK(equal);
    CHECK(m1 * 2 == DyadicMatrix(z1 * 2, z2 * 2, z1 * 2, z2 * 2));


}
