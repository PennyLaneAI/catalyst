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

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_string.hpp"

#include "Ellipse.hpp"

using namespace Catch::Matchers;
using namespace RSDecomp::GridProblem;

namespace {
bool near(const Real &x, const Real &y, const Real &tol = Real(1e-40)) { return abs(x - y) <= tol; }

bool near(const Ellipse &e1, const Ellipse &e2) {
    return near(e1.a, e2.a) && near(e1.b, e2.b) && near(e1.d, e2.d) && near(e1.p[0], e2.p[0]) &&
           near(e1.p[1], e2.p[1]);
}
} // namespace

TEST_CASE("Test Ellipse class", "[RSDecomp][Ellipse]") {
    Ellipse e{{Real(1), Real(0), Real(1)}, {Real(4), Real(5)}};

    CHECK(e.a == 1);
    CHECK(e.b == 0);
    CHECK(e.d == 1);
    CHECK(e.lambda_pow_2z() == 1);
    CHECK(near(Ellipse({lambda(), Real(0), 1 / lambda()}).lambda_pow_2z(),
               1 / (lambda() * lambda())));
    CHECK(e.determinant() == 1);
    CHECK(e.positive_semi_definite());
    CHECK(e.offset(Real(1)) == Ellipse({Real(1), Real(0), Real(1)}, {Real(5), Real(6)}));

    CHECK(e.contains(Real(4), Real(5)));
    CHECK(e.contains(Real(5), Real(5)));
    CHECK_FALSE(e.contains(Real(5), Real(6)));

    auto [x0, x1] = e.x_range();
    CHECK(x0 == 3);
    CHECK(x1 == 5);

    auto section = e.y_section(Real(4));
    REQUIRE(section.has_value());
    CHECK(section->first == 4);
    CHECK(section->second == 6);
    CHECK_FALSE(e.y_section(Real(6)).has_value());

    Ellipse e2{{Real(4), Real(1), Real(2)}, {Real(1), Real(-1)}};
    auto [normalized, factor] = e2.normalize();
    CHECK(near(normalized.determinant(), Real(1)));
    CHECK(near(factor * factor * e2.determinant(), Real(1)));

    // Dilation by f maps x to f * x.
    Ellipse dilated = e2.dilate(Real(3));
    CHECK(dilated.p == std::array<Real, 2>{Real(3), Real(-3)});
    CHECK(near(dilated.determinant() * 81, e2.determinant()));
    CHECK(dilated.contains(Real(3) * Real(1.2), Real(3) * Real(-1.1)) ==
          e2.contains(Real(1.2), Real(-1.1)));
}

TEST_CASE("Test Ellipse from_region", "[RSDecomp][Ellipse]") {
    // The ellipse contains {|u| <= 1, Re(u * conj(z)) >= 1 - eps^2 / 2} for z = e^{i theta}.
    double epsilon = GENERATE(1e-2, 1e-6, 1e-15);
    double theta = GENERATE(0.0, 0.3, -2.1, 3.0);
    CAPTURE(epsilon, theta);

    Real eps(epsilon), th(theta);
    Ellipse e = Ellipse::from_region(th, eps);
    CHECK(e.positive_semi_definite());

    Real zx = cos(th), zy = sin(th);
    Real chord = 1 - eps * eps / 2;
    Real half_width = sqrt(1 - chord * chord);
    // A point halfway up the region and both ends of its chord.
    Real mid = 1 - eps * eps / 4;
    CHECK(e.contains(mid * zx, mid * zy));
    CHECK(e.contains(chord * zx - half_width * zy, chord * zy + half_width * zx));
    CHECK(e.contains(chord * zx + half_width * zy, chord * zy - half_width * zx));
    // Points just beyond the region along z are not contained.
    CHECK_FALSE(e.contains(Real(1.01) * zx, Real(1.01) * zy));
}

TEST_CASE("Test EllipseState class", "[RSDecomp][EllipseState]") {
    Ellipse e1{{Real(1), Real(0), Real(1)}, {Real(4), Real(5)}};
    Ellipse e2{{Real(2), Real(1), Real(2)}, {Real(2), Real(3)}};
    EllipseState es{e1, e2};

    CHECK(es.e1 == e1);
    CHECK(es.e2 == e2);
    CHECK(es.skew() == 1);
    CHECK(es.bias() == 0);

    // z1 = -1 and z2 = 2, so that bias = 3.
    Real l = lambda();
    EllipseState skewed{Ellipse({l, Real(0), 1 / l}), Ellipse({1 / (l * l), Real(0), l * l})};
    CHECK(near(skewed.bias(), Real(3)));
    CHECK(skewed.bias_exceeds(2));
    CHECK_FALSE(skewed.bias_exceeds(4));
    CHECK_FALSE(skewed.negative_z_sum());
}

TEST_CASE("Test EllipseState skew_grid_op", "[RSDecomp][EllipseState]") {
    // The step lemma makes the region of every supported precision upright.
    double epsilon = GENERATE(1e-1, 1e-3, 1e-10, 1e-15);
    double theta = GENERATE(0.0, 0.4, 1.2, -2.7);
    CAPTURE(epsilon, theta);

    Ellipse region = Ellipse::from_region(Real(theta), Real(epsilon));
    EllipseState state(region.normalize().first, Ellipse());

    GridOp grid_op = state.skew_grid_op();
    CHECK(grid_op.is_special());
    CHECK(state.apply_grid_op(grid_op).skew() < 15);
}

TEST_CASE("Test GridOp class", "[RSDecomp][GridOp]") {
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
    Ellipse e1{{Real(1), Real(0), Real(1)}, {Real(5), Real(4)}};
    Ellipse e2{{Real(2), Real(1), Real(2)}, {Real(2), Real(3)}};
    EllipseState state{e1, e2};
    auto state1 = state.apply_grid_op(grid_op_B);
    auto state2 = state1.apply_grid_op(grid_op_B.inverse());
    CHECK(near(state.e1, state2.e1));
    CHECK(near(state.e2, state2.e2));
    CHECK(state1.e1 == e1.apply_grid_op(grid_op_B));
    CHECK(state1.e2 == e2.apply_grid_op(grid_op_B.adj2()));

    REQUIRE_THROWS_WITH(GridOp({1, 0}, {0, 0}, {0, 0}, {0, 0}, true),
                        ContainsSubstring("sum of a_0, b_0, c_0, d_0 must be even"));
    REQUIRE_THROWS_WITH(GridOp({1, 0}, {1, 1}, {0, 0}, {0, 0}, true),
                        ContainsSubstring("a_1, b_1, c_1, d_1 must have same parity"));
}

TEST_CASE("Test GridOp action on ZOmega and ellipses", "[RSDecomp][GridOp]") {
    // G v agrees with the real matrix of G, and v lies in G^{-1}(E) exactly if G v lies in E.
    std::string op = GENERATE("A", "B", "K", "R", "U", "X", "Z");
    CAPTURE(op);
    GridOp grid_op = GridOp::from_string(op).pow(2) * GridOp::from_string("K");
    const auto [ga, gb, gc, gd] = grid_op.flatten();

    ZOmega v(3, -1, 4, 2);
    ZOmega u = grid_op * v;
    Real vx = real_part(v), vy = imag_part(v);
    CHECK(near(real_part(u), ga * vx + gb * vy));
    CHECK(near(imag_part(u), gc * vx + gd * vy));

    Ellipse e{{Real(3), Real(1), Real(2)}, {real_part(u), imag_part(u)}};
    Ellipse pulled_back = e.apply_grid_op(grid_op);
    CHECK(near(pulled_back.determinant(), e.determinant() * pow(abs(ga * gd - gb * gc), 2)));
    CHECK(pulled_back.contains(vx, vy));
    CHECK(near(pulled_back.p[0], vx));
    CHECK(near(pulled_back.p[1], vy));
}
