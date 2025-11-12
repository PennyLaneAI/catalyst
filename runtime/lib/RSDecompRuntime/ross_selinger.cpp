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

#include "clifford-data.hpp"
#include "grid_problems.hpp"
#include "norm_solver.hpp"
#include "normal_forms.hpp"
#include "rings.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "DataView.hpp"

#define MAX_FACTORING_TRIALS 1000

/**
 * This is a dummy implementation of the rs decomposition
 */
extern "C" {
int64_t rs_decomposition_get_size_0(double theta, double epsilon, bool ppr_basis)
{
    // std::cout << "Calling rs_decomposition_get_size runtime function!\n";
    // This is a dummy implementation
    (void)theta;
    (void)epsilon;
    (void)ppr_basis;
    // The dummy sequence {0, 2, 4, 6, 8, 1, 3, 5, 7, 9} has 10 elements
    return 10;
}

/**
 * @brief Fills a pre-allocated memref with the gate sequence.
 *
 * This function signature matches the standard MLIR calling convention for
 * a 1D memref, which passes the struct fields as individual arguments.
 * Note: I have tried to use `MemRefT_int64_1d` directly from Types.h, but ran into
 * C++ ABI errors (on macOS) leading to segmentation faults. Thus, we manually unpack the memref
 * here.
 *
 * @param data_allocated Pointer to allocated data
 * @param data_aligned Pointer to aligned data
 * @param offset Data offset
 * @param size0 Size of dimension 0
 * @param stride0 Stride of dimension 0
 * @param theta Angle
 * @param epsilon Error
 * @param ppr_basis Whether to use PPR basis
 */
void rs_decomposition_get_gates_0([[maybe_unused]] int64_t *data_allocated, int64_t *data_aligned,
                                  size_t offset, size_t size0, size_t stride0, double theta,
                                  double epsilon, bool ppr_basis)
{
    // --- VERIFICATION LOGS ---
    std::cout << "[rs_decomposition_get_gates] VERIFICATION LOGS:" << std::endl;
    std::cout << "  data_allocated: " << static_cast<void *>(data_allocated) << std::endl;
    std::cout << "  data_aligned:   " << static_cast<void *>(data_aligned) << std::endl;
    std::cout << "  offset:         " << offset << std::endl;
    std::cout << "  size0:          " << size0 << std::endl;
    std::cout << "  stride0:        " << stride0 << std::endl;
    std::cout << "  theta:          " << theta << std::endl;
    std::cout << "  epsilon:        " << epsilon << std::endl;
    std::cout << "  ppr_basis:      " << (ppr_basis ? "true" : "false") << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    // This is the dummy gate sequence for testing
    std::vector<int64_t> gates_data = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9};

    // Re-construct the sizes and strides arrays for the DataView constructor
    const size_t sizes[1] = {size0};
    const size_t strides[1] = {stride0};

    // Wrap the memref descriptor in a DataView for access
    DataView<int64_t, 1> gates_view(data_aligned, offset, sizes, strides);

    // Ensure the MLIR-allocated buffer is at least as large as the data we're writing
    if (static_cast<size_t>(gates_view.size()) < gates_data.size()) {
        std::cerr << "Error: memref allocated for rs_decomposition is too small." << std::endl;
        return;
    }

    // Fill the memref data buffer
    for (size_t i = 0; i < gates_data.size(); ++i) {
        gates_view(i) = gates_data[i];
    }
}

/**
 * @brief Returns the global phase component of the decomposition.
 *
 * @param theta Angle
 * @param epsilon Error
 * @param ppr_basis Whether to use PPR basis
 * @return double The global phase
 */
double rs_decomposition_get_phase_0(double theta, double epsilon, bool ppr_basis)
{
    std::cout << "Calling rs_decomposition_get_phase runtime function!\n";
    std::cout << "phase got ppr_basis " << ppr_basis << std::endl;
    std::cout << "phase got theta " << theta << std::endl;
    std::cout << "phase got epsilon " << epsilon << std::endl;
    (void)theta;
    (void)epsilon;
    (void)ppr_basis;
    return 1.23;
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
            static INT_TYPE solution_count = 0;
            solution_count += 1;
            auto xi = ZSqrtTwo(std::pow(2, k_val)) - u_sol.norm().to_sqrt_two();
            auto t_sol = NormSolver::solve_diophantine(xi, MAX_FACTORING_TRIALS);

            if (t_sol) {
                // std::cout << "Found t solution: " << *t_sol << std::endl;
                static INT_TYPE found_solution_count = 0;
                found_solution_count += 1;
                std::cout << "Total solutions tried: " << solution_count
                          << ", found solutions: " << found_solution_count << std::endl;
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
