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

#include "GridProblems.hpp"

using namespace Catch::Matchers;
using namespace RSDecomp::GridProblem;
using bbox = std::array<double, 4>;

struct OneDimProblemParams {
    double x0, x1, y0, y1;
    int num;
    std::string test_name;
};

TEST_CASE("Test one dimensional grid problem", "[RSDecomp][GridProblems]")
{
    std::vector<OneDimProblemParams> params = {
        {8.9, 9.5, -21, -18, 2, "Case 1"},
        {246.023423, 248.5823575862261, 778, 779.0106829464769, 3, "Case 2"},
        {13734300, 13734500, -13874089.232, -13874089.181, 6, "Case 3"}};

    for (const auto &p : params) {
        SECTION(p.test_name)
        {
            double x0 = p.x0;
            double x1 = p.x1;
            double y0 = p.y0;
            double y1 = p.y1;
            int num = p.num;

            one_dim_problem_solution_iterator sols(x0, x1, y0, y1);

            bbox box = {x0, x1, y0, y1};
            CHECK(bbox_grid_points(box) == num);

            int ix = 0;
            for (const ZSqrtTwo &sol : sols) {
                ix++;
                double s1 = sol.to_double();
                double s2 = sol.adj2().to_double();
                CHECK(s1 >= x0);
                CHECK(s1 <= x1);
                CHECK(s2 >= y0);
                CHECK(s2 <= y1);
            }
            REQUIRE(ix > 0);
            REQUIRE(ix <= num);
        }
    }
}

struct UprightProblemParams {
    bbox bbox1;
    bbox bbox2;
    ZOmega expected_res;
    std::string test_name;
};

TEST_CASE("Test upright problem", "[RSDecomp][GridProblems]")
{
    std::vector<UprightProblemParams> params = {
        {{5, 6, 4, 5}, {2, 3, -1, 0}, ZOmega(1, 2, 3, 4), "Case 1"},
        {{-4, -3.8, 2.2, 2.4}, {-9, -8, 2.5, 4.2}, ZOmega(-2, 3, 1, -6), "Case 2"}};

    std::array<double, 3> D = {1, 0, 1};
    bool is_beta_first = false;  // From num_b[0] = 0
    bool is_alpha_first = false; // From num_b[1] = 0

    ZOmega shift0{0, 0, 0, 0};
    ZOmega shift1(0, 0, 1, 0);

    EllipseState state{Ellipse(D), Ellipse(D)};
    for (const auto &p : params) {
        SECTION(p.test_name)
        {
            bbox bbox3 = p.bbox1;
            for (double &val : bbox3) {
                val -= M_SQRT1_2;
            }

            bbox bbox4 = p.bbox2;
            for (double &val : bbox4) {
                val += M_SQRT1_2;
            }

            upright_problem_solution_iterator sols1(state, p.bbox1, p.bbox2, is_beta_first,
                                                    is_alpha_first, shift0);

            upright_problem_solution_iterator sols2(state, bbox3, bbox4, is_beta_first,
                                                    is_alpha_first, shift1);

            std::vector<ZOmega> all_solutions;
            for (const auto &sol : sols1) {
                all_solutions.push_back(sol);
            }
            for (const auto &sol : sols2) {
                all_solutions.push_back(sol);
            }
            auto it = std::find(all_solutions.begin(), all_solutions.end(), p.expected_res);

            CHECK(it != all_solutions.end());
        }
    }
}

struct TwoDimProblemParams {
    Ellipse e1;
    Ellipse e2;
    ZOmega expected_res;
    std::string test_name;
};

TEST_CASE("Test two dimensional grid problem", "[RSDecomp][GridProblems]")
{
    std::vector<TwoDimProblemParams> params = {
        {Ellipse({3.0, 0.5, 1.0}, {-1.7, 13.95}), Ellipse({3.0, 0.3, 0.3}, {-12.3, -7.9}),
         ZOmega(4, 3, 12, -7), "Case 1"},
        {Ellipse({2.0, 0.6, 0.9}, {-1.8, 14.93}), Ellipse({2.0, 0.4, 0.2}, {-11.3, -6.9}),
         ZOmega(4, 4, 11, -6), "Case 2"}};

    for (const auto &p : params) {
        SECTION(p.test_name)
        {
            EllipseState state(p.e1, p.e2);

            two_dim_problem_solution_iterator sols(state);
            std::vector<ZOmega> all_solutions;
            for (const auto &sol : sols) {
                all_solutions.push_back(sol);
            }

            auto it = std::find(all_solutions.begin(), all_solutions.end(), p.expected_res);

            CHECK(it != all_solutions.end());
        }
    }
}

struct GridIteratorParams {
    double theta;
    double epsilon;
};

TEST_CASE("Test GridIterator", "[RSDecomp][GridProblems]")
{
    // Data from @pytest.mark.parametrize
    std::vector<GridIteratorParams> params = {
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
    };

    for (const auto &p : params) {
        std::string test_name =
            "theta=" + std::to_string(p.theta) + ", epsilon=" + std::to_string(p.epsilon);

        SECTION(test_name)
        {
            double theta = p.theta;
            double epsilon = p.epsilon;
            GridIterator grid_sols(theta, epsilon);

            auto it = grid_sols.begin();

            CHECK(it != grid_sols.end());

            auto [u_sol, k_sol] = *it;

            std::complex<double> u_sol_complex = u_sol.to_complex();
            double denominator = std::pow(M_SQRT2, k_sol);

            std::complex<double> final_complex = u_sol_complex / denominator;

            double expected_real = std::cos(theta);
            REQUIRE(final_complex.real() == Catch::Approx(expected_real).margin(epsilon));
            double expected_imag = std::sin(theta);
            REQUIRE(final_complex.imag() == Catch::Approx(expected_imag).margin(epsilon));
        }
    }
}
