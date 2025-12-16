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

#pragma once

#include "Ellipse.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

// Note:
// The construction of the grid problem solver here are described in https://arxiv.org/pdf/1403.2975
// Section 4, 5
//
// The use of doubles here also limits the precision we can achieve
//
// TODO: Rewrite this with C++20 Coroutines and C++23 Generators for significant simplification of
// the state-machine logic.
// TODO: Rewrite with base class for common iterator logic.

namespace RSDecomp::GridProblem {
using namespace RSDecomp::Rings;

using bbox = std::array<double, 4>;

/**
 * @brief Count the number of grid points in a bounding box.
 *
 * This gives an estimation on the expected number of solution within the bounding box. This is
 * based on the Lemma 16 of arXiv:1212.6253.
 * @param bbox The bounding box defined as {x_min, x_max, y_min, y_max}.
 * @return The number of grid points within the bounding box.
 */
inline int bbox_grid_points(const bbox &bbox)
{
    const double d_ = std::log2(LAMBDA_D);
    ZSqrtTwo l1(1, 1);
    ZSqrtTwo l2(-1, 1);
    auto [x_min, x_max, y_min, y_max] = bbox;

    double d1 = x_max - x_min;
    double d2 = y_max - y_min;

    // Find the integer scaling factor for the x and y intervals, such that
    // \delta_{1/2} \cdot (\lambda - 1)^{k_{1/2}} < 1, where \lambda= 1/√2.
    int k1 = static_cast<int>(std::floor(std::log2(d1) / d_ + 1.0));
    int k2 = static_cast<int>(std::floor(std::log2(d2) / d_ + 1.0));

    double current_x0 = x_min, current_x1 = x_max, current_y0 = y_min, current_y1 = y_max;
    // If y-interval is wider than x-interval, swap.
    if (std::abs(k1) > std::abs(k2)) {
        std::swap(k1, k2);
        current_x0 = bbox[2];
        current_x1 = bbox[3];
        current_y0 = bbox[0];
        current_y1 = bbox[1];
    }

    // Scale the x and y intervals to enter the intended interval.
    // Look at one_dim_problem_solution_iterator for more detail
    double x_scale = (k1 < 0 ? l1 : l2).pow(std::abs(k1)).to_double();
    double y_scale = (k1 < 0 ? l2 : l1).pow(std::abs(k1)).to_double();
    if (k1 % 2 != 0) {
        y_scale = -y_scale;
    }

    double x0_scaled = x_scale * current_x0;
    double x1_scaled = x_scale * current_x1;
    double y0_scaled = min(y_scale * current_y0, y_scale * current_y1);
    double y1_scaled = max(y_scale * current_y0, y_scale * current_y1);

    // Check if we are indeed within the intended interval.
    RT_FAIL_IF(x1_scaled - x0_scaled < 1.0 - M_SQRT2,
               "Value should be larger than 1 - sqrt(2) for bbox");

    // Use the constraints x0 <= a + b * sqrt(2) <= x1 and y0 <= a - b * sqrt(2) <= y1
    // to obtain the bounds on b and eliminate the variable a to obtain the bounds on b.
    double lower_bound_b = (x0_scaled - y1_scaled) / (2.0 * M_SQRT2);
    double upper_bound_b = (x1_scaled - y0_scaled) / (2.0 * M_SQRT2);

    return 1 + static_cast<int>(upper_bound_b - lower_bound_b);
}

/**
 * @brief Iterator class to find the solutions to the one dimensional grid problem given intervals
 * [x0, x1] and [y0, y1].
 *
 * Given two real intervals [x0, x1] and [y0, y1] \sqrt{(x1 - x0)*(y1 - y0)} >= (1 + \sqrt(2)),
 * iterates over all solutions of the form a + b\sqrt(2) such that a + b\sqrt(2) \in [x0, x1] and
 * a - b\sqrt(2) \in [y0, y1].
 *
 * This is based on the Lemmas 16 and 17 of arXiv:1212.6253.
 */
class one_dim_problem_solution_iterator {
  private:
    // --- State for the iteration ---
    long b_current; // The current 'b' value in the main loop
    long b_min;     // The minimum 'b' to check

    // --- Scaled problem parameters (calculated once in constructor) ---
    double x0_scaled, x1_scaled, y0_scaled, y1_scaled;
    int k1;
    bool f_adj2;
    ZSqrtTwo s_scale;
    ZSqrtTwo current_solution;
    bool is_done = false;

    /**
     * @brief Encapsulates the main loop logic.
     * It searches for the next valid solution starting from b_current.
     */
    void find_next_solution()
    {
        for (long b = b_current; b >= b_min; b--) {
            // Use the constraints x0 <= a + b * sqrt(2) <= x1 to obtain the bounds on a.
            double lower_bound_a = x0_scaled - b * M_SQRT2;
            double upper_bound_a = x1_scaled - b * M_SQRT2;

            RT_FAIL_IF(upper_bound_a - lower_bound_a >= 1.0,
                       "Scaled interval width for 'a' should be less than one.");

            // Check if the bounds on the interval contains an integer.
            if (std::ceil(lower_bound_a) == std::floor(upper_bound_a)) {
                long a = static_cast<long>(std::ceil(lower_bound_a));

                // Check if the solution satisfies both bounds on x and y.
                if ((x0_scaled + y0_scaled <= 2.0 * a) && (2.0 * a <= x1_scaled + y1_scaled)) {
                    double alpha = a + b * M_SQRT2;
                    double beta = a - b * M_SQRT2;

                    // Check if the consecutive solutions are within the desired bounds.
                    if (x0_scaled <= alpha && alpha <= x1_scaled && y0_scaled <= beta &&
                        beta <= y1_scaled) {
                        // A valid (a, b) has been found.

                        // Undo the scaling to obtain the solution.
                        ZSqrtTwo sol_scaled(a, b);
                        current_solution =
                            (k1 < 0) ? (sol_scaled / s_scale) : (sol_scaled * s_scale);

                        // Check if we need to apply the sqrt(2) conjugation (yield sol.adj2()).
                        if (f_adj2) {
                            current_solution = current_solution.adj2();
                        }

                        // Set state for the next call to find_next_solution()
                        b_current = b - 1;
                        return; // Exit, preserving the state for the next ++ operation.
                    }
                }
            }
        }

        // If the loop completes, no more solutions can be found.
        is_done = true;
    }

  public:
    // For standard iterator compliance
    using iterator_category = std::input_iterator_tag;
    using value_type = ZSqrtTwo;
    using difference_type = std::ptrdiff_t;
    using pointer = ZSqrtTwo *;
    using reference = ZSqrtTwo &;

    // Default constructor for the "end" iterator
    one_dim_problem_solution_iterator() : is_done(true) {}

    /**
     * @brief Main constructor that sets up the problem parameters.
     * @param x0 The lower bound of the x-interval.
     * @param x1 The upper bound of the x-interval.
     * @param y0 The lower bound of the y-interval.
     * @param y1 The upper bound of the y-interval.
     */
    one_dim_problem_solution_iterator(double x0, double x1, double y0, double y1)
    {
        ZSqrtTwo l1(1, 1);                     // 1 + sqrt(2)
        ZSqrtTwo l2(-1, 1);                    // -1 + sqrt(2)
        const double d_ = std::log2(LAMBDA_D); // log2(1 + sqrt(2))

        double d1 = x1 - x0;
        double d2 = y1 - y0;

        f_adj2 = false; // Check if we need to apply the sqrt(2) conjugation.

        // Find the integer scaling factor for the x and y intervals, such that
        // \delta_{1/2} \cdot (\lambda - 1)^{k_{1/2}} < 1, where \lambda= 1/√2.
        int local_k1 = static_cast<int>(std::floor(std::log2(d1) / d_ + 1.0));
        int local_k2 = static_cast<int>(std::floor(std::log2(d2) / d_ + 1.0));

        k1 = local_k1;
        int k2 = local_k2;

        // If y-interval is wider than x-interval, swap.
        if (std::abs(k1) > std::abs(k2)) {
            f_adj2 = true;
            k1 = local_k2;
            k2 = local_k1;
            std::swap(x0, y0);
            std::swap(x1, y1);
        }

        // Turn the problem into a scaled grid problem,
        // such that we get to solve the specific case of:
        // -1 + √2 <= x1 - x0 < 1 and (x1 - x0)(y1 - y0) >= (1 + √2)^2.
        s_scale = l1.pow(std::abs(k1));
        double x_scale = (k1 < 0 ? l1 : l2).pow(std::abs(k1)).to_double();
        double y_scale = (k1 < 0 ? l2 : l1).pow(std::abs(k1)).to_double();
        if (k1 % 2 != 0) {
            y_scale = -y_scale;
        }
        x0_scaled = x_scale * x0;
        x1_scaled = x_scale * x1;

        double y_temp0 = y_scale * y0;
        double y_temp1 = y_scale * y1;
        y0_scaled = std::min(y_temp0, y_temp1);
        y1_scaled = std::max(y_temp0, y_temp1);

        // Check if we are solving the problem for the intended interval.
        RT_FAIL_IF(x1_scaled - x0_scaled < 1.0 - M_SQRT2,
                   "Value should be larger than 1 - sqrt(2) for bbox.");

        // --- SETUP is complete. Now initialize the iteration state ---

        // Use the constraints y0 <= a - b * sqrt(2) <= y1 and x0 <= a + b * sqrt(2) <= x1
        // to obtain the bounds on b and eliminate the variable a to obtain the bounds on b.
        double lower_bound_b = (x0_scaled - y1_scaled) / (2.0 * M_SQRT2);
        double upper_bound_b = (x1_scaled - y0_scaled) / (2.0 * M_SQRT2);

        b_current = static_cast<long>(std::floor(upper_bound_b));
        b_min = static_cast<long>(std::ceil(lower_bound_b));

        if (b_current < b_min) {
            is_done = true;
            return;
        }

        // Find the very first solution
        find_next_solution();
    }

    // Dereference operator: gets the current solution
    const ZSqrtTwo &operator*() const
    {
        if (is_done) {
            throw std::out_of_range("Iterator is out of bounds.");
        }
        return current_solution;
    }

    // Pre-increment operator: advances to the next solution
    one_dim_problem_solution_iterator &operator++()
    {
        if (!is_done) {
            find_next_solution();
        }
        return *this;
    }

    // Comparison operators to check for the end of the iteration
    bool operator==(const one_dim_problem_solution_iterator &other) const
    {
        return is_done == other.is_done;
    }

    bool operator!=(const one_dim_problem_solution_iterator &other) const
    {
        return !(*this == other);
    }

    one_dim_problem_solution_iterator begin() { return *this; }

    one_dim_problem_solution_iterator end() { return one_dim_problem_solution_iterator(); }
};

/**
 * @brief Iterator class to find the solutions to the grid problem for two upright rectangles.
 *
 * The solutions u \in Z[\omega] are such that u \in A and
 * u.adj2() \in B, where ``adj2`` is \sqrt(2) conjugation
 * and two rectangles A and B form the subregions of
 * \mathbb{R}^2 of the form [x0, x1] \times [y0, y1].
 */
class upright_problem_solution_iterator {
  private:
    // --- Parameters for the problem ---
    EllipseState state;
    ZOmega shift;

    // --- Current state of the iterator ---
    ZOmega current_solution;
    bool is_done = false;

    // --- Internal logic selector ---
    enum class Mode { BETA_FIRST, ALPHA_FIRST, BALANCED };
    Mode mode;

    // --- State for BETA_FIRST and ALPHA_FIRST modes (nested loops) ---
    std::optional<one_dim_problem_solution_iterator> outer_iter;
    std::optional<one_dim_problem_solution_iterator> inner_iter;

    // --- State for BALANCED mode (cartesian product) ---
    std::vector<ZSqrtTwo> alpha_solutions_cache;
    std::vector<ZSqrtTwo> beta_solutions_cache;
    size_t alpha_idx = 0;
    size_t beta_idx = 0;

    // --- Private method to find the next valid solution ---
    void find_next_solution()
    {
        switch (mode) {
        case Mode::BETA_FIRST:
            find_next_beta_first();
            break;
        case Mode::ALPHA_FIRST:
            find_next_alpha_first();
            break;
        case Mode::BALANCED:
            find_next_balanced();
            break;
        }
    }

    // Logic corresponding to: if num_b[0]:
    void find_next_beta_first()
    {
        // Iterate over beta_solutions1 = self.solve_one_dim_problem(Ay0, Ay1, By0, By1)
        while (outer_iter && *outer_iter != outer_iter->end()) {
            // If we don't have an inner iterator, create one for the current outer element.
            if (!inner_iter) {
                const ZSqrtTwo &beta = **outer_iter;
                try {
                    auto [Ax0, Ax1] = state.e1.x_points(beta.to_double());
                    auto [Bx0, Bx1] = state.e2.x_points(beta.adj2().to_double());

                    if (Ax1 > Ax0 && Bx1 > Bx0) {
                        inner_iter.emplace(Ax0, Ax1, Bx0, Bx1);
                    }
                }
                catch (const std::runtime_error &) {
                    // This beta is invalid and won't create an inner_iter.
                }
            }

            // Check if we have a valid inner iterator with elements.
            // for alpha in new_alpha_solutions:
            if (inner_iter && *inner_iter != inner_iter->end()) {
                current_solution = zomega_from_sqrt_pair(**inner_iter, **outer_iter, shift);
                ++(*inner_iter); // Advance for next time
                return;
            }

            // Current outer element exhausted or invalid. Advance to the next beta.
            ++(*outer_iter);
            inner_iter.reset();
        }

        // Outer iterator is exhausted.
        is_done = true;
    }

    // Logic corresponding to: elif num_b[1]:
    void find_next_alpha_first()
    {
        while (outer_iter && *outer_iter != outer_iter->end()) {
            // If we don't have an inner iterator, create one.
            if (!inner_iter) {
                const ZSqrtTwo &alpha = **outer_iter;
                try {
                    auto [Ay0, Ay1] = state.e1.y_points(alpha.to_double());
                    auto [By0, By1] = state.e2.y_points(alpha.adj2().to_double());

                    if (Ay1 > Ay0 && By1 > By0) {
                        inner_iter.emplace(Ay0, Ay1, By0, By1);
                    }
                }
                catch (const std::runtime_error &) {
                    // This alpha is invalid.
                }
            }

            // Check if we have a valid inner iterator with elements.
            // for beta in new_beta_solutions:
            if (inner_iter && *inner_iter != inner_iter->end()) {
                current_solution = zomega_from_sqrt_pair(**outer_iter, **inner_iter, shift);
                ++(*inner_iter);
                return;
            }

            // Advance outer iterator.
            ++(*outer_iter);
            inner_iter.reset();
        }

        // Outer iterator is exhausted.
        is_done = true;
    }

    // Logic corresponding to: else (Balanced case)
    void find_next_balanced()
    {
        if (alpha_solutions_cache.empty() || beta_solutions_cache.empty() ||
            alpha_idx >= alpha_solutions_cache.size()) {
            is_done = true;
            return;
        }

        // (Iterating the Cartesian product of cached solutions)
        current_solution = zomega_from_sqrt_pair(alpha_solutions_cache[alpha_idx],
                                                 beta_solutions_cache[beta_idx], shift);

        beta_idx++;
        if (beta_idx >= beta_solutions_cache.size()) {
            beta_idx = 0;
            alpha_idx++;
        }
    }

  public:
    // For standard iterator compliance
    using iterator_category = std::input_iterator_tag;
    using value_type = ZOmega;
    using difference_type = std::ptrdiff_t;
    using pointer = ZOmega *;
    using reference = ZOmega &;

    // Default constructor for the "end" iterator
    upright_problem_solution_iterator()
        : state(Ellipse(), Ellipse()), is_done(true), mode(Mode::BALANCED)
    {
    }

    /**
     * @brief Main constructor that sets up the problem.
     *
     * @param state_in The state of the grid problem.
     * @param bbox1 The bounding box of the first rectangle (Ax0, Ax1, Ay0, Ay1).
     * @param bbox2 The bounding box of the second rectangle (Bx0, Bx1, By0, By1).
     * @param is_beta_first Corresponds to num_b[0] (solve for beta first).
     * @param is_alpha_first Corresponds to num_b[1] (solve for alpha first).
     * @param shift_in The shift operator.
     */
    upright_problem_solution_iterator(const EllipseState &state_in, const bbox &bbox1,
                                      const bbox &bbox2,
                                      bool is_beta_first,  // Corresponds to num_b[0]
                                      bool is_alpha_first, // Corresponds to num_b[1]
                                      const ZOmega &shift_in)
        : state(state_in), shift(shift_in)
    {
        double Ax0 = bbox1[0], Ax1 = bbox1[1], Ay0 = bbox1[2], Ay1 = bbox1[3];
        double Bx0 = bbox2[0], Bx1 = bbox2[1], By0 = bbox2[2], By1 = bbox2[3];

        // Check the strategy flags (num_b array in Python)

        if (is_beta_first) {
            // If it is easier to solve for beta first.
            mode = Mode::BETA_FIRST;
            // Initialize the outer iterator for beta solutions.
            outer_iter.emplace(Ay0, Ay1, By0, By1);
        }
        else if (is_alpha_first) {
            // If it is easier to solve for alpha first.
            mode = Mode::ALPHA_FIRST;
            // Initialize the outer iterator for alpha solutions.
            outer_iter.emplace(Ax0, Ax1, Bx0, Bx1);
        }
        else {
            // If both of them are balanced, solve for both and refine.
            // (Here we cache results to simulate the Cartesian product loop)
            mode = Mode::BALANCED;

            one_dim_problem_solution_iterator alpha_gen(Ax0, Ax1, Bx0, Bx1);
            for (const auto &alpha : alpha_gen) {
                alpha_solutions_cache.emplace_back(alpha);
            }

            one_dim_problem_solution_iterator beta_gen(Ay0, Ay1, By0, By1);
            for (const auto &beta : beta_gen) {
                beta_solutions_cache.emplace_back(beta);
            }
        }

        // Find the very first solution
        find_next_solution();
    }

    /**
     * @brief Dereference operator to get the current solution.
     */
    const ZOmega &operator*() const
    {
        if (is_done) {
            throw std::out_of_range("Iterator is out of bounds.");
        }
        return current_solution;
    }

    /**
     * @brief Pre-increment operator to advance to the next solution.
     */
    upright_problem_solution_iterator &operator++()
    {
        if (!is_done) {
            find_next_solution();
        }
        return *this;
    }

    /**
     * @brief Comparison operators to check for the end of the iteration.
     */
    bool operator==(const upright_problem_solution_iterator &other) const
    {
        return is_done == other.is_done;
    }

    bool operator!=(const upright_problem_solution_iterator &other) const
    {
        return !(*this == other);
    }

    upright_problem_solution_iterator begin() { return *this; }
    upright_problem_solution_iterator end() { return upright_problem_solution_iterator(); }
};

/**
 * @brief Iterator class to solve the grid problem for the state(E1, E2).
 *
 * The solutions u \in Z[\omega] are such that u \in E1 and
 * u.adj2() \in E2, where ``adj2`` is \sqrt(2) conjugation.
 *
 * This is based on Proposition 5.21 and Theorem 5.18 of arXiv:1403.2975.
 */
class two_dim_problem_solution_iterator {
  private:
    // --- Parameters for the problem ---
    EllipseState original_state;
    EllipseState shifted_state;

    // --- Current state of the iterator ---
    ZOmega current_solution;
    bool is_done = false;
    bool is_on_first_coset = true;
    int num_points;

    std::optional<upright_problem_solution_iterator> current_upright_iter;

    // Private method to find the next valid solution that passes the final check
    void find_next_solution()
    {
        while (true) {
            // Check if current iterator has a valid solution
            // Corresponds to iterating through potential_sols1 or potential_sols2
            if (current_upright_iter && *current_upright_iter != current_upright_iter->end()) {
                ZOmega potential_solution = **current_upright_iter;

                // Advance the underlying iterator for the next call
                ++(*current_upright_iter);

                // --- Final verification step ---
                std::complex<double> sol1 = potential_solution.to_complex();
                std::complex<double> sol2 = potential_solution.adj2().to_complex();

                const Ellipse &e1_check = original_state.e1;
                const Ellipse &e2_check = original_state.e2;

                if (e1_check.contains(sol1.real(), sol1.imag()) &&
                    e2_check.contains(sol2.real(), sol2.imag())) {
                    current_solution = potential_solution;
                    return; // Found a valid solution, yield it.
                }
                // If check fails, loop continues to get next potential solution immediately.
            }
            else {
                // The current upright iterator is exhausted.
                // This logic handles the `chain(potential_sols1, potential_sols2)` transition.
                if (is_on_first_coset) {
                    // Switch to the second coset.
                    // This corresponds to solving for potential_sols2 using state2.
                    is_on_first_coset = false;

                    // Create the second upright iterator using the shifted state.
                    auto [bb1_x0, bb1_x1, bb1_y0, bb1_y1] = shifted_state.e1.bounding_box();
                    auto [bb2_x0, bb2_x1, bb2_y0, bb2_y1] = shifted_state.e2.bounding_box();

                    auto [p1x, p1y] = shifted_state.e1.p;
                    auto [p2x, p2y] = shifted_state.e2.p;

                    bbox shifted_bbox1 = {bb1_x0 + p1x, bb1_x1 + p1x, bb1_y0 + p1y, bb1_y1 + p1y};
                    bbox shifted_bbox2 = {bb2_x0 + p2x, bb2_x1 + p2x, bb2_y0 + p2y, bb2_y1 + p2y};

                    // Check if it is easier to solve problem for either of x-or-y-interval.
                    int num_x = bbox_grid_points(
                        {shifted_bbox1[0], shifted_bbox1[1], shifted_bbox2[0], shifted_bbox2[1]});
                    int num_y = bbox_grid_points(
                        {shifted_bbox1[2], shifted_bbox1[3], shifted_bbox2[2], shifted_bbox2[3]});

                    bool is_beta_first = num_x > num_points * num_y;
                    bool is_alpha_first = num_y > num_points * num_x;

                    // Solve for the second coset of ZOmega ring and add non-zero offset
                    // (ZOmega(c=1)).
                    current_upright_iter.emplace(shifted_state, shifted_bbox1, shifted_bbox2,
                                                 is_beta_first, is_alpha_first, ZOmega(0, 0, 1, 0));
                }
                else {
                    // Both cosets have been processed. We are done.
                    is_done = true;
                    return;
                }
            }
        }
    }

  public:
    // For standard iterator compliance
    using iterator_category = std::input_iterator_tag;
    using value_type = ZOmega;
    using difference_type = std::ptrdiff_t;
    using pointer = ZOmega *;
    using reference = ZOmega &;

    // Default constructor for the "end" iterator
    two_dim_problem_solution_iterator()
        : original_state(Ellipse(), Ellipse()), shifted_state(Ellipse(), Ellipse()), is_done(true)
    {
    }

    /**
     * @brief Main constructor that sets up the problem.
     *
     * @param state The state corresponding to the grid problem.
     * @param num_points The number of points to use to determine if the rectangle is wider
     * than the other. Default is ``1000``.
     */
    two_dim_problem_solution_iterator(const EllipseState &state, int num_points = 1000)
        : original_state(state),
          shifted_state(state.e1.offset(-1.0 / M_SQRT2), state.e2.offset(1.0 / M_SQRT2)),
          is_on_first_coset(true), num_points(num_points)
    {
        // --- Setup for the first coset (potential_sols1) ---
        auto bbox1_orig = original_state.e1.bounding_box();
        auto bbox2_orig = original_state.e2.bounding_box();

        // Calculate bbox11 and bbox21
        bbox bbox11 = {
            bbox1_orig[0] + original_state.e1.p[0], bbox1_orig[1] + original_state.e1.p[0],
            bbox1_orig[2] + original_state.e1.p[1], bbox1_orig[3] + original_state.e1.p[1]};
        bbox bbox21 = {
            bbox2_orig[0] + original_state.e2.p[0], bbox2_orig[1] + original_state.e2.p[0],
            bbox2_orig[2] + original_state.e2.p[1], bbox2_orig[3] + original_state.e2.p[1]};

        // Check if it is easier to solve problem for either of x-or-y-interval.
        // Based on this, we can try to first solve for alpha-or-beta and then refine for other.
        // If both of them are balanced, rely on doing naive search over both.
        int num_x1 = bbox_grid_points({bbox11[0], bbox11[1], bbox21[0], bbox21[1]});
        int num_y1 = bbox_grid_points({bbox11[2], bbox11[3], bbox21[2], bbox21[3]});

        bool is_beta_first = num_x1 > num_points * num_y1;
        bool is_alpha_first = num_y1 > num_points * num_x1;

        // Initialize the first upright iterator (potential_sols1)
        current_upright_iter.emplace(original_state, bbox11, bbox21, is_beta_first, is_alpha_first,
                                     ZOmega());

        // Find the very first valid solution
        find_next_solution();
    }

    const ZOmega &operator*() const
    {
        if (is_done) {
            throw std::out_of_range("Iterator is out of bounds.");
        }
        return current_solution;
    }

    two_dim_problem_solution_iterator &operator++()
    {
        if (!is_done) {
            find_next_solution();
        }
        return *this;
    }

    bool operator==(const two_dim_problem_solution_iterator &other) const
    {
        return is_done == other.is_done;
    }

    bool operator!=(const two_dim_problem_solution_iterator &other) const
    {
        return !(*this == other);
    }

    // To make it usable in a range-based for loop
    two_dim_problem_solution_iterator begin() { return *this; }
    two_dim_problem_solution_iterator end() { return two_dim_problem_solution_iterator(); }
};

/**
 * @brief Iterate over the solutions to the scaled grid problem.
 *
 * This is based on the Section 5 of arXiv:1403.2975 and implements Proposition 5.22,
 * to enumerate all solutions to the scaled grid problem over the epsilon-region
 * and a unit disk.
 */
class GridIterator {
  public:
    // --- Standard C++ Iterator Type Definitions ---
    using iterator_category = std::input_iterator_tag;
    using value_type = std::pair<ZOmega, int>; // Corresponds to tuple[ZOmega, int]
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = value_type &;

  private:
    // --- Parameters ---
    double theta;
    double epsilon;
    int max_trials;
    std::pair<double, double> zval; // (cos(theta), sin(theta))
    int kmin;
    double target;

    // --- State for Iteration ---
    value_type current_solution;
    enum class IterState { GUESSING, MAIN_LOOP, FALLBACK, DONE } iter_state;

    // Main loop state
    int k;
    int i_;
    double e_;
    double t_;
    Ellipse e1;
    const Ellipse e2; // This is constant: Ellipse((1, 0, 1), (0, 0))
    GridOp grid_op;   // Result of skew_grid_op()

    // Phase 1: GUESSING
    std::array<ZOmega, 6> guess_solutions;
    size_t guess_idx = 0;

    // Phase 2: MAIN_LOOP
    int main_loop_idx = 0; // The 'ix'
    std::optional<two_dim_problem_solution_iterator> two_dim_iter;

    // Phase 3: FALLBACK
    std::vector<ZOmega> int_s; // Fallback solutions
    std::vector<int> init_k;   // Fallback k values
    size_t fallback_idx = 0;

    /**
     * @brief Phase 1: Iterates through trivial guesses.
     * @return true if a solution was found (yield), false if we should transition state.
     */
    bool run_guessing_phase()
    {
        // Solutions for the trivial cases.
        while (guess_idx < guess_solutions.size()) {
            const ZOmega &sol = guess_solutions[guess_idx];
            guess_idx++; // Advance for next time

            std::complex<double> complx_sol = sol.to_complex();
            double dot_prod = zval.first * complx_sol.real() + zval.second * complx_sol.imag();

            double norm_zsqrt_two = std::abs(sol.norm2().to_sqrt_two().to_double());

            // Check if solution is within bounds (<= 1) and meets target.
            if (norm_zsqrt_two <= 1.0) {
                if (dot_prod >= target) {
                    current_solution = {sol, 0};
                    return true; // Found a solution
                }
            }
        }
        // Guesses exhausted, move to main loop
        iter_state = IterState::MAIN_LOOP;
        main_loop_idx = 0;    // The *first* trial to run is 0
        two_dim_iter.reset(); // Ensure inner iterator is clear
        return false;
    }

    /**
     * @brief Phase 2: The main loop.
     * Corresponds to: for ix in range(self.max_trials):
     * @return true if a solution was found, false if we need to loop/transition.
     */
    bool run_main_loop_phase()
    {
        // Do we have an active inner iterator? If so, try to drain it.
        // Corresponds to: for solution in potential_solutions:
        if (two_dim_iter) {
            while (*two_dim_iter != two_dim_iter->end()) {
                ZOmega solution = **two_dim_iter;
                ++(*two_dim_iter); // Advance inner iterator

                // Normalize the solution and obtain the scaling exponent of sqrt(2).
                auto [scaled_sol, kf] = (grid_op * solution).normalize();

                std::complex<double> complx_sol = scaled_sol.to_complex();
                double sol_real = complx_sol.real();
                double sol_imag = complx_sol.imag();

                // Update the scaling exponent of sqrt(2).
                int k_ = k - kf;

                // Calculate dot product adjusted for scaling
                double k_div_2 = static_cast<double>(k_) / 2.0;
                double k_mod_2 = static_cast<double>(k_ % 2);
                double denominator = std::pow(2.0, k_div_2) * std::pow(M_SQRT2, k_mod_2);

                double dot_prod = (zval.first * sol_real + zval.second * sol_imag) / denominator;

                // Check if the solution follows the constraints of the target-region.
                double norm_zsqrt_two = std::abs(scaled_sol.norm2().to_sqrt_two().to_double());

                if (norm_zsqrt_two <= std::pow(2.0, k_)) {
                    if (dot_prod >= target) {
                        current_solution = {scaled_sol, k_};
                        return true; // Found a solution, exit
                    }
                    else if (dot_prod >= t_) {
                        // Fallback solution collection
                        int_s.emplace_back(scaled_sol);
                        init_k.emplace_back(k_);
                    }
                }
            } // end while(inner_iter)

            // Inner iterator is exhausted. Reset it.
            two_dim_iter.reset();

            // Run the "end of loop" logic (Logic to update k, e_, t_)
            if (main_loop_idx == i_) {
                k = std::max(kmin, k + 1);
                e_ = epsilon;
                t_ = t_ / 10.0;

                auto [en_, _] = Ellipse::from_region(theta, e_, kmin).normalize();
                grid_op = EllipseState(en_, e2).skew_grid_op();
            }
            else {
                k = k + 1;
            }
            e1 = Ellipse::from_region(theta, e_, k);

            // Advance the main loop index
            main_loop_idx++;
        }

        // We need a new inner iterator (either it's the first time,
        // or the previous one was just exhausted).

        // Check if we're done *before* creating a new one.
        if (main_loop_idx >= max_trials) {
            iter_state = IterState::FALLBACK;
            fallback_idx = 0;
            // Add standard fallback
            int_s.emplace_back(0, 0, 0, 1); // ZOmega(d=1)
            init_k.emplace_back(0);
            return false; // Go to FALLBACK state
        }

        // Create the inner iterator for the *current* main_loop_idx
        try {
            // Update the radius of the unit disk.
            double radius = std::pow(2.0, -k);
            Ellipse e2_({radius, 0.0, radius}, {0.0, 0.0});

            // Apply the grid operation to the state and solve the two-dimensional grid
            // problem.
            EllipseState state = EllipseState(e1, e2_).apply_grid_op(grid_op);

            two_dim_iter.emplace(state);
        }
        catch (const std::exception &e) {
            // Corresponds to Python's `except (ValueError, ZeroDivisionError): break`
            iter_state = IterState::FALLBACK;
            fallback_idx = 0;
            int_s.emplace_back(0, 0, 0, 1); // ZOmega(d=1)
            init_k.emplace_back(0);
            return false; // Go to FALLBACK state
        }

        // Loop again. The next pass will enter `if (two_dim_iter)`
        // and start processing the iterator we just made.
        return false;
    }

    /**
     * @brief Phase 3: Returns fallback solutions collected during main loop.
     * @return true if solution yielded, false if finished.
     */
    bool run_fallback_phase()
    {
        if (fallback_idx < int_s.size()) {
            current_solution = {int_s[fallback_idx], init_k[fallback_idx]};
            fallback_idx++;
            return true; // Found a solution, exit
        }

        // Fallback exhausted
        iter_state = IterState::DONE;
        return false;
    }

    /**
     * @brief Private helper to find the next valid solution.
     * This is the core state machine of the iterator.
     */
    void find_next_solution()
    {
        while (true) {
            switch (iter_state) {
            case IterState::GUESSING:
                if (run_guessing_phase())
                    return;
                break;
            case IterState::MAIN_LOOP:
                if (run_main_loop_phase())
                    return;
                break;
            case IterState::FALLBACK:
                if (run_fallback_phase())
                    return;
                break;
            case IterState::DONE:
                return;
            }
        }
    }

  public:
    /**
     * @brief Default constructor. Creates an "end" iterator.
     */
    GridIterator() : iter_state(IterState::DONE) {}

    /**
     * @brief Main constructor to start the iteration.
     *
     * @param theta_in The angle of the grid problem.
     * @param epsilon_in The epsilon of the grid problem.
     * @param max_trials_in The maximum number of iterations. Default is ``20``.
     */
    GridIterator(double theta_in, double epsilon_in, int max_trials_in = 20)
        : theta(theta_in), epsilon(epsilon_in), max_trials(max_trials_in),
          zval(std::cos(theta), std::sin(theta)),
          kmin(static_cast<int>(3.0 * std::log2(1.0 / epsilon) / 2.0)),
          target(1.0 - epsilon * epsilon / 2.0), iter_state(IterState::GUESSING),
          e1({1.0, 0.0, 1.0}, {0.0, 0.0}),
          e2({1.0, 0.0, 1.0}, {0.0, 0.0}) // Ellipse((1, 0, 1), (0, 0))
    {
        // --- Warm start for an initial guess ---
        // Values are obtained from PennyLane implementation
        // https://github.com/PennyLaneAI/pennylane/blob/2fa8e910437ccded0fe600f415dcc5f7436db4a8/pennylane/ops/op_math/decompositions/grid_problems.py#L597
        // where 14 is kmin for 1e-3.
        k = std::min(kmin, 14);
        i_ = 6; // Give 6 trials for warm start.
        e_ = std::max(epsilon, 1e-3);
        t_ = std::min(target, 0.9999995);

        // Ellipse for the epsilon-region.
        e1 = Ellipse::from_region(theta, e_, k);
        // Normalize the epsilon-region.
        auto [en, _] = e1.normalize();

        // Skew grid operation for the epsilon-region.
        grid_op = EllipseState(en, e2).skew_grid_op();

        // Initialize guess_solutions
        guess_solutions = {
            ZOmega(-1, 0, 0, 0), // ZOmega(a=-1)
            ZOmega(0, -1, 0, 0), // ZOmega(b=-1)
            ZOmega(0, 1, 0, 0),  // ZOmega(b=1)
            ZOmega(0, 0, 1, 0),  // ZOmega(c=1)
            ZOmega(0, 0, 0, 1),  // ZOmega(d=1)
            ZOmega(0, 0, 0, -1)  // ZOmega(d=-1)
        };
        guess_idx = 0;

        // Find the very first solution
        find_next_solution();
    }

    // --- Iterator Interface Implementation ---

    /**
     * @brief Dereference operator. Gets the current solution.
     */
    const value_type &operator*() const
    {
        if (iter_state == IterState::DONE) {
            throw std::out_of_range("Iterator is out of bounds.");
        }
        return current_solution;
    }

    /**
     * @brief Pre-increment operator. Advances to the next solution.
     */
    GridIterator &operator++()
    {
        if (iter_state != IterState::DONE) {
            find_next_solution();
        }
        return *this;
    }

    /**
     * @brief Equality comparison.
     */
    bool operator==(const GridIterator &other) const
    {
        // Two iterators are equal if they are both in the DONE state.
        return iter_state == IterState::DONE && other.iter_state == IterState::DONE;
    }

    bool operator!=(const GridIterator &other) const { return !(*this == other); }

    GridIterator begin() { return *this; }

    GridIterator end() { return GridIterator(); }
};

} // namespace RSDecomp::GridProblem
