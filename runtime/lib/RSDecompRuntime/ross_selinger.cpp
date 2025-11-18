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
#include <variant>
#include <vector>

#include "DataView.hpp"

#define MAX_FACTORING_TRIALS 1000

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

using DecompResult =
    std::variant<std::vector<CliffordData::GateType>, std::vector<CliffordData::PPRGateType>>;
std::pair<DecompResult, double> eval_ross_algorithm(double angle, double epsilon, bool ppr_basis)
{
    using CacheKey = std::tuple<double, double, bool>;
    using CacheValue = std::pair<DecompResult, double>;
    static lru_cache<CacheKey, CacheValue, 10000> ross_cache;

    CacheKey key = {angle, epsilon, ppr_basis};

    if (auto val_opt = ross_cache.get(key); val_opt) {
        return *val_opt;
    }
    // std::cout << "Cache miss for angle: " << angle << ", epsilon: " << epsilon << std::endl;
    double shift = 0.0;
    double modified_angle = -angle / 2.0;
    ZOmega scale(0, 0, 0, 1);
    int max_search_trials = 10000;
    [[maybe_unused]] double phase = 0;

    ZOmega u(0, 0, 0, 1);
    ZOmega t(0, 0, 0, 0);
    int k = 0;

    DyadicMatrix dyd_mat(0, 0, 0, 0);
    if (is_odd_multiple_of_pi_8(modified_angle)) {
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
        double theta = modified_angle;
        int sign = 1;
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

        GridProblem::GridIterator u_solutions(modified_angle + shift, epsilon, max_search_trials);

        // std::cout << "angle: " << modified_angle << ", shift: " << shift << ", epsilon: " <<
        // epsilon
        //           << std::endl;

        for (const auto &[u_sol, k_val] : u_solutions) {
            // std::cout << "Testing potential solution with k=" << k_val << std::endl << "u_sol: "
            // << u_sol << std::endl;
            auto xi = ZSqrtTwo(std::pow(2, k_val)) - u_sol.norm().to_sqrt_two();
            auto t_sol = NormSolver::solve_diophantine(xi, MAX_FACTORING_TRIALS);

            if (t_sol) {
                // std::cout << "Found solution t_sol = " << *t_sol
                //           << std::endl;
                u = u_sol * scale;
                t = *t_sol * scale;
                k = k_val;
                break;
            }
        }

        dyd_mat = DyadicMatrix(u, -t.conj(), t, u.conj(), k);
    }
    SO3Matrix so3_mat(dyd_mat);
    auto normal_form_result = normal_forms::ma_normal_form(so3_mat);

    double phase_result = normal_form_result.second;

    CacheValue result_to_cache;
    if (ppr_basis) {
        // Convert to PPR basis
        std::vector<CliffordData::PPRGateType> ppr_gates =
            CliffordData::HSTtoPPR(normal_form_result.first);
        result_to_cache = std::make_pair(std::move(ppr_gates), phase_result);
    }
    else {
        // Use the standard Clifford+T basis
        result_to_cache = std::make_pair(std::move(normal_form_result.first), phase_result);
    }

    ross_cache.put(key, result_to_cache);
    return result_to_cache;
}

/**
 * This is a dummy implementation of the rs decomposition
 */
extern "C" {
int64_t rs_decomposition_get_size_0(double theta, double epsilon, bool ppr_basis)
{
    if (ppr_basis) {
        assert(false && "Simulating PPR basis not yet supported.");
    }

    auto result = eval_ross_algorithm(theta, epsilon, ppr_basis);
    const auto &gates_vector = result.first;

    size_t s = std::visit([](const auto &vec) { return vec.size(); }, gates_vector);

    return static_cast<int64_t>(s);
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
 * @param offset Offset
 * @param size0 Size of the first dimension
 * @param stride0 Stride of the first dimension
 * @param theta Angle
 * @param epsilon Error
 * @param ppr_basis Whether to use PPR basis
 */
void rs_decomposition_get_gates_0([[maybe_unused]] int64_t *data_allocated, int64_t *data_aligned,
                                  size_t offset, size_t size0, size_t stride0, double theta,
                                  double epsilon, bool ppr_basis)
{
    (void)ppr_basis;

    auto result = eval_ross_algorithm(theta, epsilon, ppr_basis);
    const auto &gates_data = result.first; // std::vector<CliffordData::GateType>

    // Re-construct the sizes and strides arrays for the DataView constructor
    const size_t sizes[1] = {size0};
    const size_t strides[1] = {stride0};

    // Wrap the memref descriptor in a DataView for access
    DataView<int64_t, 1> gates_view(data_aligned, offset, sizes, strides);

    // Ensure the MLIR-allocated buffer is at least as large as the data we're writing

    std::visit(
        [&gates_view](const auto &vec) {
            size_t s = vec.size();
            if (static_cast<size_t>(gates_view.size()) < s) {
                std::cerr << "Error: memref allocated for rs_decomposition is too small."
                          << " (Allocated: " << gates_view.size() << ", Needed: " << s << ")"
                          << std::endl;
                return;
            }

            for (size_t i = 0; i < s; ++i) {
                gates_view(i) = static_cast<int64_t>(vec[i]);
            }
        },
        gates_data);
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
    (void)ppr_basis;

    auto result = eval_ross_algorithm(theta, epsilon, ppr_basis);
    double phase = result.second;

    return phase;
}

} // extern "C"
