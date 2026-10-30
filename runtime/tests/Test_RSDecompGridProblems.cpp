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

#include <set>

#include "catch2/catch_test_macros.hpp"

#include "GridProblems.hpp"

using namespace RSDecomp::GridProblem;

namespace {
using SqrtTwoSet = std::set<std::pair<INT_TYPE, INT_TYPE>>;
using OmegaSet = std::set<std::array<INT_TYPE, 4>>;

/**
 * @brief All α ∈ Z[√2] with α ∈ [x0, x1] and α• ∈ [y0, y1], by checking every α = a + b√2 with
 * a and b in the bounding box of the problem.
 */
SqrtTwoSet brute_force_one_dim(Real x0, Real x1, Real y0, Real y1) {
    SqrtTwoSet solutions;
    Real two_sqrt2 = 2 * sqrt2();
    for (INT_TYPE a = ceil_int(Real((x0 + y0) / 2)); a <= floor_int(Real((x1 + y1) / 2)); ++a) {
        for (INT_TYPE b = ceil_int(Real((x0 - y1) / two_sqrt2));
             b <= floor_int(Real((x1 - y0) / two_sqrt2)); ++b) {
            ZSqrtTwo alpha(a, b);
            Real s1 = to_real(alpha), s2 = to_real(alpha.adj2());
            if (x0 <= s1 && s1 <= x1 && y0 <= s2 && s2 <= y1) {
                solutions.insert({a, b});
            }
        }
    }
    return solutions;
}

SqrtTwoSet solve_one_dim(Real x0, Real x1, Real y0, Real y1) {
    SqrtTwoSet solutions;
    solve_one_dim_problem(x0, x1, y0, y1, [&](const ZSqrtTwo &alpha) {
        CHECK(solutions.insert({alpha.a, alpha.b}).second);
        return true;
    });
    return solutions;
}

bool in_state(const EllipseState &state, const ZOmega &u) {
    ZOmega u_conj = u.adj2();
    return state.e1.contains(real_part(u), imag_part(u)) &&
           state.e2.contains(real_part(u_conj), imag_part(u_conj));
}

/**
 * @brief All u ∈ Z[ω] with u ∈ state.e1 and u• ∈ state.e2, from the bounding boxes of both cosets.
 */
OmegaSet brute_force_two_dim(const EllipseState &state) {
    OmegaSet solutions;
    const Real &h = inv_sqrt2();
    std::array<std::pair<ZOmega, Real>, 2> cosets = {std::pair{ZOmega(0), Real(0)},
                                                     std::pair{ZOmega(0, 0, 1, 0), h}};
    for (const auto &[shift, offset] : cosets) {
        auto [x0, x1] = state.e1.x_range();
        auto [x2, x3] = state.e2.x_range();
        auto [y0, y1] = state.e1.y_range();
        auto [y2, y3] = state.e2.y_range();
        auto alphas = brute_force_one_dim(x0 - offset, x1 - offset, x2 + offset, x3 + offset);
        auto betas = brute_force_one_dim(y0 - offset, y1 - offset, y2 + offset, y3 + offset);
        for (const auto &[a1, b1] : alphas) {
            for (const auto &[a2, b2] : betas) {
                ZOmega u = zomega_from_sqrt_pair(ZSqrtTwo(a1, b1), ZSqrtTwo(a2, b2), shift);
                if (in_state(state, u)) {
                    solutions.insert({u.a, u.b, u.c, u.d});
                }
            }
        }
    }
    return solutions;
}

OmegaSet solve_two_dim(const EllipseState &state) {
    OmegaSet solutions;
    solve_two_dim_problem(state, [&](const ZOmega &u) {
        CHECK(solutions.insert({u.a, u.b, u.c, u.d}).second);
        return true;
    });
    return solutions;
}

/**
 * @brief Check that the first solution of the scaled grid problem is within epsilon of e^{i theta}.
 */
void check_rz_grid_search(double theta, double epsilon) {
    CAPTURE(theta, epsilon);
    RzGridSearch search{Real(theta), Real(epsilon)};

    std::optional<std::pair<ZOmega, int>> first;
    for (int k = 0; k < 200 && !first; k++) {
        search.visit_level(k, [&](const ZOmega &u) {
            first = {u, k};
            return false;
        });
    }
    REQUIRE(first.has_value());

    auto [u, k] = *first;
    CHECK((ZSqrtTwo(INT_TYPE(1) << k, 0) - u.norm2().to_sqrt_two()).sign() >= 0);
    Real scale = pow(sqrt2(), k);
    Real dx = real_part(u) / scale - cos(Real(theta));
    Real dy = imag_part(u) / scale - sin(Real(theta));
    CHECK(sqrt(dx * dx + dy * dy) <= Real(epsilon));
}

/**
 * @brief Check that RzGridSearch visits exactly the solutions found from sections of the bounding
 * ellipses, on the first levels that have solutions.
 */
void check_rz_grid_search_complete(double theta, double epsilon) {
    CAPTURE(theta, epsilon);
    Real th(theta), eps(epsilon);
    RzGridSearch search(th, eps);

    Ellipse region = Ellipse::from_region(th, eps);
    Ellipse disk;
    GridOp grid_op = EllipseState(region.normalize().first, disk).skew_grid_op();
    EllipseState upright = EllipseState(region, disk).apply_grid_op(grid_op);
    Real zx = cos(th), zy = sin(th), target = 1 - eps * eps / 2;

    size_t num_solutions = 0;
    for (int k = 0; k < 200 && num_solutions < 20; k++) {
        CAPTURE(k);
        Real scale = pow(sqrt2(), k);
        ZSqrtTwo two_pow_k(INT_TYPE(1) << k, 0);
        EllipseState state(upright.e1.dilate(scale), upright.e2.dilate(scale));

        OmegaSet expected;
        solve_two_dim_problem(state, [&](const ZOmega &v) {
            ZOmega u = grid_op * v;
            ZSqrtTwo xi = two_pow_k - u.norm2().to_sqrt_two();
            bool divisible = (v.a + v.c) % 2 == 0 && (v.b + v.d) % 2 == 0;
            if (!divisible && xi.sign() >= 0 && xi.adj2().sign() >= 0 &&
                zx * real_part(u) + zy * imag_part(u) >= target * scale) {
                expected.insert({u.a, u.b, u.c, u.d});
            }
            return true;
        });

        OmegaSet visited;
        search.visit_level(k, [&](const ZOmega &u) {
            visited.insert({u.a, u.b, u.c, u.d});
            return true;
        });
        CHECK(visited == expected);
        num_solutions += expected.size();
    }
    CHECK(num_solutions >= 20);
}
} // namespace

TEST_CASE("Test one dimensional grid problem", "[RSDecomp][GridProblems]") {
    SECTION("Case 1") {
        auto solutions = solve_one_dim(Real(8.9), Real(9.5), Real(-21), Real(-18));
        CHECK(!solutions.empty());
        CHECK(solutions == brute_force_one_dim(Real(8.9), Real(9.5), Real(-21), Real(-18)));
    }
    SECTION("Case 2") {
        Real x0("246.023423"), x1("248.5823575862261"), y0(778), y1("779.0106829464769");
        auto solutions = solve_one_dim(x0, x1, y0, y1);
        CHECK(!solutions.empty());
        CHECK(solutions == brute_force_one_dim(x0, x1, y0, y1));
    }
    SECTION("Case 3") {
        Real x0(13734300), x1(13734500), y0("-13874089.232"), y1("-13874089.181");
        auto solutions = solve_one_dim(x0, x1, y0, y1);
        CHECK(!solutions.empty());
        CHECK(solutions == brute_force_one_dim(x0, x1, y0, y1));
    }
    SECTION("Empty interval") {
        CHECK(solve_one_dim(Real(0.1), Real(0.2), Real(0.1), Real(0.2)).empty());
        CHECK(solve_one_dim(Real(1), Real(0), Real(-5), Real(5)).empty());
    }
}

TEST_CASE("Test one dimensional grid problem at high precision", "[RSDecomp][GridProblems]") {
    // At this magnitude, intervals of width ~1 cannot be resolved in double precision.
    Real x0("12345678901234567890123456789.25");
    Real y0("-9876543210987654321098765432.5");
    Real x1 = x0 + 7, y1 = y0 + Real(3.5);

    auto solutions = solve_one_dim(x0, x1, y0, y1);
    CHECK(!solutions.empty());
    CHECK(solutions == brute_force_one_dim(x0, x1, y0, y1));
}

TEST_CASE("Test two dimensional grid problem", "[RSDecomp][GridProblems]") {
    struct Params {
        Ellipse e1, e2;
        ZOmega expected_res;
    };
    std::vector<Params> params = {
        {Ellipse({Real(3.0), Real(0.5), Real(1.0)}, {Real(-1.7), Real(13.95)}),
         Ellipse({Real(3.0), Real(0.3), Real(0.3)}, {Real(-12.3), Real(-7.9)}),
         ZOmega(4, 3, 12, -7)},
        {Ellipse({Real(2.0), Real(0.6), Real(0.9)}, {Real(-1.8), Real(14.93)}),
         Ellipse({Real(2.0), Real(0.4), Real(0.2)}, {Real(-11.3), Real(-6.9)}),
         ZOmega(4, 4, 11, -6)},
        {Ellipse({Real(0.05), Real(0.01), Real(0.08)}, {Real(0.5), Real(-3.0)}),
         Ellipse({Real(0.2), Real(-0.05), Real(0.1)}, {Real(1.0), Real(2.0)}), ZOmega(0)},
    };

    for (size_t i = 0; i < params.size(); i++) {
        CAPTURE(i);
        EllipseState state(params[i].e1, params[i].e2);
        // Candidates may lie slightly outside the ellipses, and are filtered by the caller.
        OmegaSet solutions;
        for (const auto &u : solve_two_dim(state)) {
            if (in_state(state, ZOmega(u[0], u[1], u[2], u[3]))) {
                solutions.insert(u);
            }
        }
        CHECK(solutions == brute_force_two_dim(state));
        const ZOmega &e = params[i].expected_res;
        CHECK(solutions.count({e.a, e.b, e.c, e.d}) == 1);
    }
}

TEST_CASE("Test RzGridSearch completeness", "[RSDecomp][GridProblems]") {
    check_rz_grid_search_complete(0.4, 1e-3);
    check_rz_grid_search_complete(-1.1, 1e-10);
    // Close to multiples of π/8, where most grid points of the bounding ellipse lie outside the
    // epsilon-region.
    check_rz_grid_search_complete(5e-10, 1e-12);
    check_rz_grid_search_complete(M_PI / 8 - 1e-9, 1e-12);
    check_rz_grid_search_complete(1.3, 1e-15);
}

TEST_CASE("Test RzGridSearch", "[RSDecomp][GridProblems]") {
    std::vector<std::pair<double, double>> params = {
        {0.0, 1e-3},
        {M_PI / 4, 1e-4},
        {M_PI / 8, 1e-4},
        {M_PI / 3, 1e-3},
        {M_PI / 5, 1e-5},
        {0.392125483789636, 1e-6},
        {0.6789684841313233, 1e-3},
        {0.056202026824044335, 1e-5},
        {0.21375826964510297, 1e-4},
        {0.5549739238125396, 1e-6},
        {-0.454645364564563, 1e-3},
        {-0.5549739238125396, 1e-2},
        {2.9, 1e-10},
        {0.3, 1e-15},
        {-1.7, 1e-14},
    };
    for (const auto &[theta, epsilon] : params) {
        check_rz_grid_search(theta, epsilon);
    }
}
