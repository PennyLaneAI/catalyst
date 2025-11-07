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

#include "grid_problems.hpp"
#include "norm_solver.hpp"
#include "clifford-data.hpp"
#include "normal_forms.hpp"
#include "rings.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map> // <<< ADDED: Include for std::map
#include <vector>

#define MAX_FACTORING_TRIALS 1000

/**
 * This is a dummy implementation of the rs decomposition
 */

// 1D memref descriptor
struct MemRef1D {
    int64_t *allocated;
    int64_t *aligned;
    int64_t offset;
    int64_t size;
    int64_t stride;
};

extern "C" {

// Decomposition body
MemRef1D rs_decomposition_0(double theta, double epsilon)
{
    std::cout << "Calling rs_decomposition runtime function!\n";
    // This returns a dummy gate sequence for testing
    std::vector<int64_t> gates_data = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};
    int64_t num_gates = gates_data.size();
    int64_t *heap_data = new int64_t[num_gates];
    memcpy(heap_data, gates_data.data(), num_gates * sizeof(int64_t));

    std::cout << "theta received = " << theta << "\n";
    std::cout << "epsilon received = " << epsilon << "\n";
    std::cout << "heap_data address: " << static_cast<void *>(heap_data) << "\n";

    MemRef1D result;
    result.allocated = heap_data;
    result.aligned = heap_data;
    result.offset = 0;
    result.size = num_gates;
    result.stride = 1;
    return result;
}

// Free memref
void free_memref_0(int64_t *allocated, int64_t *aligned, int64_t offset, int64_t size,
                   int64_t stride)
{
    // Mark other args as unused to prevent compiler warnings
    (void)aligned;
    (void)offset;
    (void)size;
    (void)stride;

    std::cout << "free_memref_0 called\n";
    std::cout << "deleting heap_data at: " << static_cast<void *>(allocated) << "\n";
    // delete[] allocated;
}

} // extern "C"

bool is_odd_multiple_of_pi_8(double angle)
{
    const double pi_over_8 = M_PI / 8.0;
    double multiple = angle / pi_over_8;
    int rounded_multiple = static_cast<int>(std::round(multiple));
    return (rounded_multiple % 2 != 0) && (std::abs(multiple - rounded_multiple) < 1e-10);
}

int multiple_of_pi_8(double angle)
{
    const double pi_over_8 = M_PI / 8.0;
    double multiple = angle / pi_over_8;
    int rounded_multiple = static_cast<int>(std::round(multiple));
    return rounded_multiple;
}

// <<< MODIFIED: Note I've changed std::__1::vector to std::vector for portability.
std::pair<std::vector<CliffordData::GateType>, double> eval_ross_algorithm(double angle,
                                                                           double epsilon)
{

    // <<< ADDED: Define cache key and value types for clarity
    using CacheKey = std::pair<double, double>;
    // <<< MODIFIED: Using std::vector instead of std::__1::vector
    using CacheValue = std::pair<std::vector<CliffordData::GateType>, double>;

    // <<< ADDED: Declare the thread-local static cache
    thread_local static std::map<CacheKey, CacheValue> ross_cache;

    // <<< ADDED: Create the key for the current call
    CacheKey key = {angle, epsilon};

    // <<< ADDED: Check if the result is already in the cache
    auto it = ross_cache.find(key);
    if (it != ross_cache.end()) {
        // Found it! Return the cached value immediately.
        return it->second;
    }

    // <<< ORIGINAL CODE: (If not in cache, proceed with computation)
    double shift = 0.0;
    // Note: The original 'angle = -angle / 2.0;' was here.
    // This is problematic, as the original 'angle' is part of the cache key.
    // We should use a new variable for the modified angle.
    double modified_angle = -angle / 2.0;
    ZOmega scale(0, 0, 0, 1);
    int max_search_trials = 10000;
    [[maybe_unused]] double phase = 0;

    ZOmega u(0, 0, 0, 1);
    ZOmega t(0, 0, 0, 0);
    int k = 0;

    DyadicMatrix dyd_mat(0, 0, 0, 0);
    // <<< MODIFIED: Use 'modified_angle' instead of 'angle' from here on
    if (is_odd_multiple_of_pi_8(modified_angle)) {

        // <<< MODIFIED: Use 'modified_angle'
        int coeff = ((multiple_of_pi_8(modified_angle) % 16) + 16) % 16;
        std::map<int, std::tuple<int, int, ZOmega, ZOmega>> precomputed_map = {
            {1, {1, 7, ZOmega(0, 1, 0, 0), ZOmega(0, 1, 0, 0)}},
            {3, {1, 5, ZOmega(0, 0, 0, -1), ZOmega(0, 0, 0, 1)}},
            {5, {1, 3, ZOmega(0, -1, 0, 0), ZOmega(0, -1, 0, 0)}},
            {7, {1, 1, ZOmega(0, 0, 0, 1), ZOmega(0, 0, 0, -1)}},
            {9, {-1, 7, ZOmega(0, 1, 0, 0), ZOmega(1, 0, 0, 0)}},
            {11, {-1, 5, ZOmega(0, 0, 0, -1), ZOmega(0, 1, 0, 0)}},
            {13, {-1, 3, ZOmega(0, -1, 0, 0), ZOmega(0, 0, 1, 0)}},
            {15, {-1, 1, ZOmega(0, 0, 0, 1), ZOmega(0, 0, 0, 1)}}};

        auto &[shift_0, shift_1, scale_0, scale_1] = precomputed_map[coeff];
        DyadicMatrix t_mat = dyadic_matrix_mul(DyadicMatrix(u, t, t, ZOmega(0, 0, 1, 0)),
                                               DyadicMatrix(scale_0, t, t, u));
        dyd_mat = t_mat * scale_1;
        if (shift_0 == -1) {
            phase += (8 - shift_1) * M_PI / 8.0;
        }
        else {
            phase += M_PI / 8.0;
        }
    }
    else {
        // <<< MODIFIED: Use 'modified_angle'
        double theta = modified_angle;
        int sign = 1;
        // std::fmod matches Python's % for positive divisors
        theta = std::fmod(theta, M_PI * 4.0);
        if (theta < 0.0) {
            theta += M_PI * 4.0;
        }
        if (theta > (2.0 * M_PI)) {
            sign = -1;
            theta -= 4.0 * M_PI;
        }
        double abs_theta = std::abs(theta);

        if (M_PI_4 <= abs_theta && abs_theta < (M_PI_4 * 3.0)) {
            // Region [pi/4, 3pi/4)
            shift = -sign * M_PI_2;
            scale = ZOmega(0, sign, 0, 0); // ZOmega(b=sign)
        }
        else if ((M_PI_4 * 3.0) <= abs_theta && abs_theta < (M_PI_4 * 5.0)) {
            // Region [3pi/4, 5pi/4)
            shift = -sign * M_PI;
            scale = ZOmega(0, 0, 0, -1); // ZOmega(d=-1)
        }
        else if ((M_PI_4 * 5.0) <= abs_theta && abs_theta < (M_PI_4 * 7.0)) {
            // Region [5pi/4, 7pi/4)
            shift = -sign * 3.0 * M_PI_2;
            scale = ZOmega(0, -sign, 0, 0); // ZOmega(b=-sign)
        }
        else {
            // Region [0, pi/4) or [7pi/4, 2pi]
            shift = 0.0;
            scale = ZOmega(0, 0, 0, 1); // ZOmega(d=1)
        }

        // <<< MODIFIED: Use 'modified_angle'
        GridProblem::GridIterator u_solutions(modified_angle + shift, epsilon, max_search_trials);

        // <<< MODIFIED: Use 'modified_angle' for the debug print
        std::cout << "angle: " << modified_angle << ", shift: " << shift << ", epsilon: " << epsilon
                  << std::endl;

        for (const auto &[u_sol, k_val] : u_solutions) {
            // std::cout << "Testing potential solution with k=" << k_val << std::endl << "u_sol: "
            // << u_sol << std::endl;
            auto xi = ZSqrtTwo(std::pow(2, k_val)) - u_sol.norm().to_sqrt_two();
            auto t_sol = NormSolver::solve_diophantine(xi, MAX_FACTORING_TRIALS);

            if (t_sol) {
                // std::cout << "Found t solution: " << *t_sol << std::endl;
                u = u_sol * scale;
                t = *t_sol * scale;
                k = k_val;
                break;
            }
        }

        dyd_mat = DyadicMatrix(u, -t.conj(), t, u.conj(), k);
    }
    SO3Matrix so3_mat(dyd_mat);
    // std::cout << "before/after"<< std::endl;
    auto normal_form_result = normal_forms::ma_normal_form(so3_mat);

    // <<< ADDED: Store the newly computed result in the cache
    ross_cache[key] = normal_form_result;

    return normal_form_result;
}

// Will be used to return global phase
double rs_decomposition_get_phase_0(double theta, double epsilon)
{
    std::cout << "phase got theta " << theta << std::endl;
    std::cout << "phase got epsilon " << epsilon << std::endl;
    return 1.23;
}
