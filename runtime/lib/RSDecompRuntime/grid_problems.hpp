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

#include "ellipse.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <memory> // For std::shared_ptr
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace RSDecomp::GridProblem {
using namespace RSDecomp::Rings;

using bbox = std::array<double, 4>;

inline int bbox_grid_points(const bbox &bbox)
{
    const double d_ = std::log2(ZSqrtTwo(1, 1).to_double());
    ZSqrtTwo l1(1, 1);
    ZSqrtTwo l2(-1, 1);
    double d1 = bbox[1] - bbox[0];
    double d2 = bbox[3] - bbox[2];

    int k1 = static_cast<int>(std::floor(std::log2(d1) / d_ + 1.0));
    int k2 = static_cast<int>(std::floor(std::log2(d2) / d_ + 1.0));

    double current_x0 = bbox[0], current_x1 = bbox[1], current_y0 = bbox[2], current_y1 = bbox[3];

    if (std::abs(k1) > std::abs(k2)) {
        std::swap(k1, k2);
        current_x0 = bbox[2];
        current_x1 = bbox[3];
        current_y0 = bbox[0];
        current_y1 = bbox[1];
    }

    double x_scale = (k1 < 0 ? l1 : l2).pow(std::abs(k1)).to_double();
    double y_scale = std::pow(-1.0, k1) * (k1 < 0 ? l2 : l1).pow(std::abs(k1)).to_double();

    double x0_scaled = x_scale * current_x0;
    double x1_scaled = x_scale * current_x1;
    double y0_scaled = min(y_scale * current_y0, y_scale * current_y1);
    double y1_scaled = max(y_scale * current_y0, y_scale * current_y1);

    if (x1_scaled - x0_scaled < 1.0 - M_SQRT2) {
        throw std::runtime_error("Value should be larger than 1 - sqrt(2) for bbox");
    }

    double lower_bound_b = (x0_scaled - y1_scaled) / (2.0 * M_SQRT2);
    double upper_bound_b = (x1_scaled - y0_scaled) / (2.0 * M_SQRT2);

    return 1 + static_cast<int>(upper_bound_b - lower_bound_b);
}

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

    void find_next_solution()
    {
        // This function encapsulates the main loop from the Python code.
        // It searches for the next valid solution starting from b_current.
        for (long b = b_current; b >= b_min; --b) {
            // Use the constraints x0 <= a + b * sqrt(2) <= x1 to obtain the bounds on a.
            double lower_bound_a = x0_scaled - b * M_SQRT2;
            double upper_bound_a = x1_scaled - b * M_SQRT2;

            if (upper_bound_a - lower_bound_a >= 1.0) {
                throw std::runtime_error("Scaled interval width for 'a' should be less than one.");
            }

            // Check if the interval [lower_bound_a, upper_bound_a] contains exactly one integer.
            if (std::ceil(lower_bound_a) == std::floor(upper_bound_a)) {
                long a = static_cast<long>(std::ceil(lower_bound_a));

                // Check if the solution satisfies both bounds on x and y.
                if ((x0_scaled + y0_scaled <= 2.0 * a) && (2.0 * a <= x1_scaled + y1_scaled)) {
                    double alpha = a + b * M_SQRT2;
                    double beta = a - b * M_SQRT2;

                    // Final check that the solutions are within the desired bounds.
                    if (x0_scaled <= alpha && alpha <= x1_scaled && y0_scaled <= beta &&
                        beta <= y1_scaled) {
                        // A valid (a, b) has been found.

                        // Undo the scaling to obtain the solution.
                        ZSqrtTwo sol_scaled(a, b);
                        current_solution =
                            (k1 < 0) ? (sol_scaled / s_scale) : (sol_scaled * s_scale);
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

    // Main constructor that sets up the problem
    one_dim_problem_solution_iterator(double x0, double x1, double y0, double y1)
    {
        ZSqrtTwo l1(1, 1);                           // 1 + sqrt(2)
        ZSqrtTwo l2(-1, 1);                          // -1 + sqrt(2)
        const double d_ = std::log2(l1.to_double()); // log2(1 + sqrt(2))

        double d1 = x1 - x0;
        double d2 = y1 - y0;

        f_adj2 = false; // Flag to apply sqrt(2) conjugation at the end.

        int local_k1 = static_cast<int>(std::floor(std::log2(d1) / d_ + 1.0));
        int local_k2 = static_cast<int>(std::floor(std::log2(d2) / d_ + 1.0));

        k1 = local_k1;
        int k2 = local_k2;

        if (std::abs(k1) > std::abs(k2)) {
            f_adj2 = true;
            k1 = local_k2;
            k2 = local_k1;
            std::swap(x0, y0);
            std::swap(x1, y1);
        }

        // Turn the problem into a scaled grid problem
        s_scale = l1.pow(std::abs(k1));
        double x_scale = (k1 < 0 ? l1 : l2).pow(std::abs(k1)).to_double();
        double y_scale = (k1 < 0 ? l2 : l1).pow(std::abs(k1)).to_double();
        y_scale *= std::pow(-1, k1);

        x0_scaled = x_scale * x0;
        x1_scaled = x_scale * x1;

        double y_temp0 = y_scale * y0;
        double y_temp1 = y_scale * y1;
        y0_scaled = min(y_temp0, y_temp1);
        y1_scaled = max(y_temp0, y_temp1);

        if (x1_scaled - x0_scaled < 1.0 - M_SQRT2) {
            throw std::runtime_error("Scaled interval width should be larger than 1 - sqrt(2).");
        }

        // --- SETUP is complete. Now initialize the iteration state ---

        // Calculate the search bounds for 'b'
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

    void find_next_beta_first()
    {
        while (outer_iter && *outer_iter != outer_iter->end()) {
            // If we don't have an inner iterator, create one for the current outer element.
            if (!inner_iter) {
                const ZSqrtTwo &beta = **outer_iter;
                try {
                    auto xp1 = state.e1.x_points(beta.to_double());
                    auto xp2 = state.e2.x_points(beta.adj2().to_double());
                    if (xp1.second > xp1.first && xp2.second > xp2.first) {
                        inner_iter.emplace(xp1.first, xp1.second, xp2.first, xp2.second);
                    }
                }
                catch (const std::runtime_error &) {
                    // This beta is invalid and won't create an inner_iter.
                }
            }

            // Check if we have a valid inner iterator with elements.
            if (inner_iter && *inner_iter != inner_iter->end()) {
                current_solution = zomega_from_sqrt_pair(**inner_iter, **outer_iter, shift);
                ++(*inner_iter); // Advance for next time
                return;
            }

            // Current outer element exhausted or invalid. Advance to the next.
            ++(*outer_iter);
            inner_iter.reset();
        }

        // Outer iterator is exhausted.
        is_done = true;
    }

    void find_next_alpha_first()
    {
        while (outer_iter && *outer_iter != outer_iter->end()) {
            // If we don't have an inner iterator, create one.
            if (!inner_iter) {
                const ZSqrtTwo &alpha = **outer_iter;
                try {
                    auto yp1 = state.e1.y_points(alpha.to_double());
                    auto yp2 = state.e2.y_points(alpha.adj2().to_double());
                    if (yp1.second > yp1.first && yp2.second > yp2.first) {
                        inner_iter.emplace(yp1.first, yp1.second, yp2.first, yp2.second);
                    }
                }
                catch (const std::runtime_error &) {
                    // This alpha is invalid.
                }
            }

            // Check if we have a valid inner iterator with elements.
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

    void find_next_balanced()
    {
        if (alpha_solutions_cache.empty() || beta_solutions_cache.empty() ||
            alpha_idx >= alpha_solutions_cache.size()) {
            is_done = true;
            return;
        }

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

    // Main constructor that sets up the problem
    upright_problem_solution_iterator(const EllipseState &state_in, const bbox &bbox1,
                                      const bbox &bbox2,
                                      bool is_beta_first,  // Corresponds to num_b[0]
                                      bool is_alpha_first, // Corresponds to num_b[1]
                                      const ZOmega &shift_in)
        : state(state_in), shift(shift_in)
    {

        double Ax0 = bbox1[0], Ax1 = bbox1[1], Ay0 = bbox1[2], Ay1 = bbox1[3];
        double Bx0 = bbox2[0], Bx1 = bbox2[1], By0 = bbox2[2], By1 = bbox2[3];

        if (is_beta_first) {
            mode = Mode::BETA_FIRST;
            outer_iter.emplace(Ay0, Ay1, By0, By1);
        }
        else if (is_alpha_first) {
            mode = Mode::ALPHA_FIRST;
            outer_iter.emplace(Ax0, Ax1, Bx0, Bx1);
        }
        else {
            mode = Mode::BALANCED;
            one_dim_problem_solution_iterator alpha_gen(Ax0, Ax1, Bx0, Bx1);
            for (const auto &alpha : alpha_gen) {
                alpha_solutions_cache.push_back(alpha);
            }

            one_dim_problem_solution_iterator beta_gen(Ay0, Ay1, By0, By1);
            for (const auto &beta : beta_gen) {
                beta_solutions_cache.push_back(beta);
            }
        }

        // Find the very first solution
        find_next_solution();
    }

    const ZOmega &operator*() const
    {
        if (is_done) {
            throw std::out_of_range("Iterator is out of bounds.");
        }
        return current_solution;
    }

    upright_problem_solution_iterator &operator++()
    {
        if (!is_done) {
            find_next_solution();
        }
        return *this;
    }

    bool operator==(const upright_problem_solution_iterator &other) const
    {
        return is_done == other.is_done;
    }

    bool operator!=(const upright_problem_solution_iterator &other) const
    {
        return !(*this == other);
    }

    // To make it usable in a range-based for loop
    upright_problem_solution_iterator begin() { return *this; }
    upright_problem_solution_iterator end() { return upright_problem_solution_iterator(); }
};

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
            if (current_upright_iter && *current_upright_iter != current_upright_iter->end()) {
                ZOmega potential_solution = **current_upright_iter;
                // std::cout << "Potential solution: " << potential_solution << std::endl;

                // Advance the underlying iterator for the next call
                ++(*current_upright_iter);

                // This is the crucial final check from the Python code
                std::complex<double> sol1 = potential_solution.to_complex();
                std::complex<double> sol2 = potential_solution.adj2().to_complex();

                // const Ellipse& e1_check = is_on_first_coset ? original_state.e1 :
                // shifted_state.e1; const Ellipse& e2_check = is_on_first_coset ? original_state.e2
                // : shifted_state.e2;

                const Ellipse &e1_check = original_state.e1;
                const Ellipse &e2_check = original_state.e2;

                if (e1_check.contains(sol1.real(), sol1.imag()) &&
                    e2_check.contains(sol2.real(), sol2.imag())) {
                    // std::cout << "Found valid solution: " << potential_solution << std::endl;
                    current_solution = potential_solution;
                    return; // Found a valid solution
                }
                // If check fails, loop continues to get next potential solution
            }
            else {
                // The current upright iterator is exhausted.
                if (is_on_first_coset) {
                    // Switch to the second coset
                    is_on_first_coset = false;

                    // Create the second upright iterator
                    auto bbox1 = shifted_state.e1.bounding_box();
                    auto bbox2 = shifted_state.e2.bounding_box();
                    bbox shifted_bbox1 = {
                        bbox1[0] + shifted_state.e1.p[0], bbox1[1] + shifted_state.e1.p[0],
                        bbox1[2] + shifted_state.e1.p[1], bbox1[3] + shifted_state.e1.p[1]};
                    bbox shifted_bbox2 = {
                        bbox2[0] + shifted_state.e2.p[0], bbox2[1] + shifted_state.e2.p[0],
                        bbox2[2] + shifted_state.e2.p[1], bbox2[3] + shifted_state.e2.p[1]};

                    int num_x = bbox_grid_points(
                        {shifted_bbox1[0], shifted_bbox1[1], shifted_bbox2[0], shifted_bbox2[1]});
                    int num_y = bbox_grid_points(
                        {shifted_bbox1[2], shifted_bbox1[3], shifted_bbox2[2], shifted_bbox2[3]});

                    bool is_beta_first = num_x > num_points * num_y;
                    bool is_alpha_first = num_y > num_points * num_x;

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
        : original_state(Ellipse({1, 0, 1}), Ellipse({1, 0, 1})),
          shifted_state(Ellipse({1, 0, 1}), Ellipse({1, 0, 1})), is_done(true)
    {
    }

    // Main constructor that sets up the problem
    two_dim_problem_solution_iterator(const EllipseState &state, int num_points = 1000)
        : original_state(state),
          shifted_state(state.e1.offset(-1.0 / M_SQRT2), state.e2.offset(1.0 / M_SQRT2)),
          is_on_first_coset(true), num_points(num_points)
    {

        // --- Setup for the first coset ---
        auto bbox1_orig = original_state.e1.bounding_box();
        auto bbox2_orig = original_state.e2.bounding_box();

        bbox bbox11 = {
            bbox1_orig[0] + original_state.e1.p[0], bbox1_orig[1] + original_state.e1.p[0],
            bbox1_orig[2] + original_state.e1.p[1], bbox1_orig[3] + original_state.e1.p[1]};
        bbox bbox21 = {
            bbox2_orig[0] + original_state.e2.p[0], bbox2_orig[1] + original_state.e2.p[0],
            bbox2_orig[2] + original_state.e2.p[1], bbox2_orig[3] + original_state.e2.p[1]};

        int num_x1 = bbox_grid_points({bbox11[0], bbox11[1], bbox21[0], bbox21[1]});
        int num_y1 = bbox_grid_points({bbox11[2], bbox11[3], bbox21[2], bbox21[3]});

        bool is_beta_first = num_x1 > num_points * num_y1;
        bool is_alpha_first = num_y1 > num_points * num_x1;

        // Initialize the first upright iterator
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

class GridIterator {
  public:
    // --- Standard C++ Iterator Type Definitions ---
    using iterator_category = std::input_iterator_tag;
    using value_type = std::pair<ZOmega, int>; // Corresponds to tuple[ZOmega, int]
    using difference_type = std::ptrdiff_t;
    using pointer = value_type *;
    using reference = value_type &;

  private:
    // --- Parameters (from Python __init__) ---
    double theta;
    double epsilon;
    int max_trials;
    std::pair<double, double> zval; // (cos(theta), sin(theta))
    int kmin;
    double target;

    // --- State for Iteration (from Python __iter__) ---
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
     * @brief Private helper to find the next valid solution.
     * This is the core state machine of the iterator.
     */
    /**
     * @brief Private helper to find the next valid solution.
     * This is the core state machine of the iterator.
     */
    void find_next_solution()
    {
        while (true) {
            if (iter_state == IterState::GUESSING) {
                // --- Phase 1: Guessing ---
                while (guess_idx < guess_solutions.size()) {
                    const ZOmega &sol = guess_solutions[guess_idx];
                    guess_idx++; // Advance for next time

                    std::complex<double> complx_sol = sol.to_complex();
                    double dot_prod =
                        zval.first * complx_sol.real() + zval.second * complx_sol.imag();

                    double norm_zsqrt_two = std::abs(sol.norm().to_sqrt_two().to_double());

                    if (norm_zsqrt_two <= 1.0) {
                        if (dot_prod >= target) {
                            current_solution = {sol, 0};
                            return; // Found a solution, exit
                        }
                    }
                }
                // Guesses exhausted, move to main loop
                iter_state = IterState::MAIN_LOOP;
                main_loop_idx = 0;    // The *first* trial to run is 0
                two_dim_iter.reset(); // Ensure inner iterator is clear
                continue;             // Re-enter the while(true) to start the MAIN_LOOP
            }

            if (iter_state == IterState::MAIN_LOOP) {
                // --- Phase 2: Main Loop ---

                // 1. Do we have an active inner iterator? If so, try to drain it.
                if (two_dim_iter) {
                    while (*two_dim_iter != two_dim_iter->end()) {
                        ZOmega solution = **two_dim_iter;
                        ++(*two_dim_iter); // Advance inner iterator

                        auto [scaled_sol, kf] = (grid_op * solution).normalize();

                        std::complex<double> complx_sol = scaled_sol.to_complex();
                        double sol_real = complx_sol.real();
                        double sol_imag = complx_sol.imag();

                        int k_ = k - kf;
                        double k_div_2 = static_cast<double>(k_ / 2);
                        double k_mod_2 = static_cast<double>(k_ % 2);
                        double denominator = std::pow(2.0, k_div_2) * std::pow(M_SQRT2, k_mod_2);

                        double dot_prod =
                            (zval.first * sol_real + zval.second * sol_imag) / denominator;

                        double norm_zsqrt_two =
                            std::abs(scaled_sol.norm().to_sqrt_two().to_double());
                        // std::cout << "Testing solution: idx = " << main_loop_idx << ", potential
                        // solution = " << solution << std::endl;

                        if (norm_zsqrt_two <= std::pow(2.0, k_)) {
                            if (dot_prod >= target) {
                                // std::cout << "Found solution!!" << std::endl;
                                current_solution = {scaled_sol, k_};
                                return; // Found a solution, exit
                            }
                            else if (dot_prod >= t_) {
                                int_s.push_back(scaled_sol);
                                init_k.push_back(k_);
                            }
                        }
                    } // end while(inner_iter)

                    // 1b. Inner iterator is exhausted. Reset it.
                    two_dim_iter.reset();

                    // 1c. Run the "end of loop" logic (your code 2b)
                    if (main_loop_idx == i_) {
                        k = max(kmin, k + 1);
                        e_ = epsilon;
                        t_ = t_ / 10.0;
                        auto [en_, _] = Ellipse::from_region(theta, e_, kmin).normalize();
                        grid_op = EllipseState(en_, e2).skew_grid_op();
                    }
                    else {
                        k = k + 1;
                    }
                    e1 = Ellipse::from_region(theta, e_, k);

                    // 1d. Advance the main loop index
                    main_loop_idx++;
                } // end if(two_dim_iter)

                // 2. We need a new inner iterator (either it's the first time,
                //    or the previous one was just exhausted).

                // Check if we're done *before* creating a new one.
                if (main_loop_idx >= max_trials) {
                    iter_state = IterState::FALLBACK;
                    fallback_idx = 0;
                    int_s.push_back(ZOmega(0, 0, 0, 1)); // ZOmega(d=1)
                    init_k.push_back(0);
                    continue; // Go to FALLBACK state
                }

                // 3. Create the inner iterator for the *current* main_loop_idx (your code 2c)
                try {
                    double radius = std::pow(2.0, -k);
                    Ellipse e2_({radius, 0.0, radius}, {0.0, 0.0});
                    EllipseState state = EllipseState(e1, e2_).apply_grid_op(grid_op);

                    two_dim_iter.emplace(
                        state); // Calls two_dim_problem_solution_iterator constructor
                }
                catch (const std::exception &e) {
                    // Corresponds to Python's `except (ValueError, ZeroDivisionError): break`
                    iter_state = IterState::FALLBACK;
                    fallback_idx = 0;
                    int_s.push_back(ZOmega(0, 0, 0, 1)); // ZOmega(d=1)
                    init_k.push_back(0);
                    continue; // Go to FALLBACK state
                }

                // 4. Loop again. The next pass will enter `if (two_dim_iter)`
                //    and start processing the iterator we just made.
                continue;
            }

            if (iter_state == IterState::FALLBACK) {
                // --- Phase 3: Fallback ---
                if (fallback_idx < int_s.size()) {
                    current_solution = {int_s[fallback_idx], init_k[fallback_idx]};
                    fallback_idx++;
                    return; // Found a solution, exit
                }

                // Fallback exhausted
                iter_state = IterState::DONE;
            }

            if (iter_state == IterState::DONE) {
                return; // Stay in DONE state
            }
        }
    }

  public:
    /**
     * @brief Default constructor. Creates an "end" iterator.
     */
    GridIterator()
        : iter_state(IterState::DONE), e1({1.0, 0.0, 1.0}, {0.0, 0.0}),
          e2({1.0, 0.0, 1.0}, {0.0, 0.0})
    {
    }

    /**
     * @brief Main constructor to start the iteration.
     *
     * @param theta The angle of the grid problem.
     * @param epsilon The epsilon of the grid problem.
     * @param max_trials The maximum number of iterations.
     */
    GridIterator(double theta_in, double epsilon_in, int max_trials_in = 20)
        : theta(theta_in), epsilon(epsilon_in), max_trials(max_trials_in),
          zval(std::cos(theta), std::sin(theta)),
          kmin(static_cast<int>(3.0 * std::log2(1.0 / epsilon) / 2.0)),
          target(1.0 - epsilon * epsilon / 2.0), iter_state(IterState::GUESSING),
          e1({1.0, 0.0, 1.0}, {0.0, 0.0}),
          e2({1.0, 0.0, 1.0}, {0.0, 0.0}) // Ellipse((1, 0, 1), (0, 0))
    {
        // --- Warm start (from Python __iter__) ---
        k = min(kmin, 14);
        i_ = 6;
        e_ = max(epsilon, 1e-3);
        t_ = min(target, 0.9999995);

        // Assumes Ellipse::from_region and Ellipse::normalize exist
        e1 = Ellipse::from_region(theta, e_, k);
        auto [en, _] = e1.normalize(); // Assumes C++17 structured binding

        // Assumes EllipseState and skew_grid_op exist
        grid_op = EllipseState(en, e2).skew_grid_op();

        // Initialize guess_solutions (assuming ZOmega(a,b,c,d))
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
     * @brief Equality comparison. Checks if two iterators are at the same position.
     * For input iterators, this is typically only used to check against the "end" iterator.
     */
    bool operator==(const GridIterator &other) const
    {
        // Two iterators are equal if they are both in the DONE state.
        return iter_state == IterState::DONE && other.iter_state == IterState::DONE;
    }

    /**
     * @brief Inequality comparison.
     */
    bool operator!=(const GridIterator &other) const { return !(*this == other); }

    /**
     * @brief Returns the begin iterator (i.e., itself).
     * Allows for use in range-based for loops.
     */
    GridIterator begin() { return *this; }

    /**
     * @brief Returns the end iterator (a default-constructed, "done" iterator).
     * Allows for use in range-based for loops.
     */
    GridIterator end() { return GridIterator(); }
};

} // namespace RSDecomp::GridProblem
