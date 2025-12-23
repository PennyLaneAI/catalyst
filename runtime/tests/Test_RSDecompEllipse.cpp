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

#include "Ellipse.hpp"

using namespace Catch::Matchers;
using namespace RSDecomp::GridProblem;

TEST_CASE("Test Ellipse class", "[RSDecomp][Ellipse]")
{
    Ellipse e{{1.0, 0.0, 1.0}, {4.0, 5.0}};

    CHECK(e.a == 1.0);
    CHECK(e.b == 0.0);
    CHECK(e.d == 1.0);
    CHECK(e.determinant() == 1.0);
    CHECK(e.discriminant() == 0.0);
    CHECK(e.bounding_box() == std::array<double, 4>{-1.0, 1.0, -1.0, 1.0});
    CHECK(e.offset(1.0) == Ellipse({1.0, 0.0, 1.0}, {5.0, 6.0}));
    CHECK(e.b_from_uprightness(0.2) == std::sqrt(std::pow((M_PI / 0.8), 2) - 1));
    CHECK(e.positive_semi_definite() == true);
    CHECK(e.x_points(4) == std::make_pair(4.0, 4.0));
    CHECK(e.y_points(5) == std::make_pair(5.0, 5.0));
    CHECK(e.uprightness() == 0.7853981633974483);

    REQUIRE_THROWS_WITH(e.x_points(24), ContainsSubstring("Cannot compute x_points:"));
    REQUIRE_THROWS_WITH(e.y_points(24), ContainsSubstring("Cannot compute y_points:"));
}

TEST_CASE("Test EllipseState class", "[RSDecomp][EllipseState]")
{
    Ellipse e1{{1.0, 0.0, 1.0}, {4.0, 5.0}};
    Ellipse e2{{2.0, 1.0, 2.0}, {2.0, 3.0}};
    EllipseState es{e1, e2};

    CHECK(es.e1 == e1);
    CHECK(es.e2 == e2);
    CHECK(es.skew() == 1.0);
    CHECK(es.bias() == 0.0);
}

TEST_CASE("Test GridOp class", "[RSDecomp][GridOp]")
{
    GridOp grid_op = GridOp::from_string("I");
    CHECK(grid_op.a == std::array<INT_TYPE, 2>{1, 0});
    CHECK(grid_op.b == std::array<INT_TYPE, 2>{0, 0});
    CHECK(grid_op.c == std::array<INT_TYPE, 2>{0, 0});
    CHECK(grid_op.d == std::array<INT_TYPE, 2>{1, 0});

    CHECK(grid_op.pow(3) == grid_op);

    GridOp grid_op_R = GridOp::from_string("R");
    CHECK(grid_op_R.pow(3) == GridOp({0, -1}, {0, -1}, {0, 1}, {0, -1}));

    CHECK(GridOp::from_string("K").is_special());

    GridOp g1{{0, 1}, {0, -1}, {1, 1}, {-1, 1}};
    GridOp g1_t{{-1, 1}, {0, -1}, {1, 1}, {0, 1}};
    CHECK(g1.transpose() == g1_t);

    GridOp grid_op_B = GridOp::from_string("B");
    Ellipse e1{{1.0, 0.0, 1.0}, {5.0, 4.0}};
    Ellipse e2{{2.0, 1.0, 2.0}, {2.0, 3.0}};
    EllipseState state{e1, e2};
    auto state1 = grid_op_B.apply_to_state(state);
    auto state2 = grid_op_B.inverse().apply_to_state(state1);
    CHECK(state.e1 == state2.e1);
    CHECK(state.e2 == state2.e2);
    CHECK(state1.e1 == grid_op_B.apply_to_ellipse(e1));

    REQUIRE_THROWS_WITH(GridOp({1, 0}, {0, 0}, {0, 0}, {0, 0}, true),
                        ContainsSubstring("sum of a_0, b_0, c_0, d_0 must be even"));
    REQUIRE_THROWS_WITH(GridOp({1, 0}, {1, 1}, {0, 0}, {0, 0}, true),
                        ContainsSubstring("a_1, b_1, c_1, d_1 must have same parity"));
}
