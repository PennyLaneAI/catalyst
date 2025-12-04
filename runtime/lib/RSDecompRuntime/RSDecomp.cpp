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

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "DataView.hpp"
#include "GridProblems.hpp"
#include "NormSolver.hpp"
#include "NormalForms.hpp"
#include "Rings.hpp"
#include "RossSelinger.hpp"

#define MAX_FACTORING_TRIALS 1000

namespace {
bool is_odd_multiple_of_pi_4(double angle)
{
    const double pi_over_4 = M_PI / 4.0;
    double multiple = angle / pi_over_4;
    int rounded_multiple = static_cast<int>(std::round(multiple));
    return (rounded_multiple % 2 != 0) && (std::abs(multiple - rounded_multiple) < 1e-10);
}
} // namespace

namespace RSDecomp::RossSelinger {

using namespace RSDecomp::Rings;
using namespace RSDecomp::Utils;
using namespace RSDecomp::CliffordData;
using namespace RSDecomp::NormalForms;

std::pair<std::vector<GateType>, double> compute_raw_decomposition(double angle, double epsilon)
{
    ZOmega scale(0, 0, 0, 1);
    int max_search_trials = 10000;
    double phase = 0.0;

    ZOmega u(0, 0, 0, 1);
    ZOmega t(0, 0, 0, 0);
    INT_TYPE k = 0;

    std::vector<GateType> decomposition;
    DyadicMatrix dyd_mat(ZOmega(0), ZOmega(0), ZOmega(0), ZOmega(0), INT_TYPE(0));

    if (is_odd_multiple_of_pi_4(angle)) {
        const double pi_over_4 = M_PI / 4.0;
        long units = std::lround(angle / pi_over_4);
        int normalized = ((units % 8) + 8) % 8;

        if (normalized & 4) {
            decomposition.push_back(GateType::Z);
        }
        if (normalized & 2) {
            decomposition.push_back(GateType::S);
        }
        if (normalized & 1) {
            decomposition.push_back(GateType::T);
        }

        if (decomposition.empty()) {
            decomposition.push_back(GateType::I);
        }

        phase = static_cast<double>(units) * (M_PI / 8.0);
    }
    else {
        double modified_angle = -angle / 2.0;
        long k = std::lround(modified_angle / M_PI_2);
        double shift = -static_cast<double>(k) * M_PI_2;
        int idx = ((k % 4) + 4) % 4;

        switch (idx) {
        case 0:                         // 0 shift (Identity)
            scale = ZOmega(0, 0, 0, 1); // d=1
            break;
        case 1:                         // pi/2 shift
            scale = ZOmega(0, 1, 0, 0); // b=1
            break;
        case 2:                          // pi shift
            scale = ZOmega(0, 0, 0, -1); // d=-1
            break;
        case 3:                          // 3pi/2 (or -pi/2) shift
            scale = ZOmega(0, -1, 0, 0); // b=-1
            break;
        }
        GridProblem::GridIterator u_solutions(modified_angle + shift, epsilon, max_search_trials);

        for (const auto &[u_sol, k_val] : u_solutions) {
            // Calculate 2^k_val as an INT_TYPE
            INT_TYPE two_pow_k = INT_TYPE(1) << k_val;
            auto xi = ZSqrtTwo(two_pow_k, 0) - u_sol.norm().to_sqrt_two();
            auto t_sol = NormSolver::solve_diophantine(xi, MAX_FACTORING_TRIALS);

            if (t_sol) {
                u = u_sol * scale;
                t = *t_sol * scale;
                k = k_val;
                break;
            }
        }

        dyd_mat = DyadicMatrix(u, -t.conj(), t, u.conj(), INT_TYPE(k));
        SO3Matrix so3_mat(dyd_mat);
        auto normal_form_result = ma_normal_form(so3_mat);
        decomposition = normal_form_result.first;
        phase = normal_form_result.second;
    }
    return {std::move(decomposition), phase};
}

// Cache for Standard Basis
using StdCacheKey = std::tuple<double, double>;
using StdCacheValue = std::pair<std::vector<GateType>, double>;
static lru_cache<StdCacheKey, StdCacheValue, 10000> ross_cache_std;

std::pair<std::vector<GateType>, double> eval_ross_algorithm(double angle, double epsilon)
{
    StdCacheKey key = {angle, epsilon};

    if (auto val_opt = ross_cache_std.get(key); val_opt) {
        return *val_opt;
    }

    auto result = compute_raw_decomposition(angle, epsilon);
    ross_cache_std.put(key, result);
    return result;
}

// Cache for PPR Basis
using PPRCacheKey = std::tuple<double, double>;
using PPRCacheValue = std::pair<std::vector<PPRGateType>, double>;
static lru_cache<PPRCacheKey, PPRCacheValue, 10000> ross_cache_ppr;

std::pair<std::vector<PPRGateType>, double> eval_ross_algorithm_ppr(double angle, double epsilon)
{
    PPRCacheKey key = {angle, epsilon};

    if (auto val_opt = ross_cache_ppr.get(key); val_opt) {
        return *val_opt;
    }

    auto raw_result = compute_raw_decomposition(angle, epsilon);

    std::vector<PPRGateType> ppr_gates = HSTtoPPR(raw_result.first);

    PPRCacheValue result = {std::move(ppr_gates), raw_result.second};

    ross_cache_ppr.put(key, result);
    return result;
}

// Extern C implementation
extern "C" {

size_t rs_decomposition_get_size(double theta, double epsilon, bool ppr_basis)
{
    if (ppr_basis) {
        auto result = eval_ross_algorithm_ppr(theta, epsilon);
        return result.first.size();
    }
    else {
        auto result = eval_ross_algorithm(theta, epsilon);
        return result.first.size();
    }
}

/**
 * @brief Fills a pre-allocated memref with the gate sequence.
 *
 * This function signature matches the standard MLIR calling convention for
 * a 1D memref (IndexType), which passes the struct fields as individual arguments.
 * Note: I have tried to use `MemRefT` directly from Types.h, but ran into
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
void rs_decomposition_get_gates([[maybe_unused]] size_t *data_allocated, size_t *data_aligned,
                                size_t offset, size_t size0, size_t stride0, double theta,
                                double epsilon, bool ppr_basis)
{
    (void)data_allocated;

    const size_t sizes[1] = {size0};
    const size_t strides[1] = {stride0};

    // Wrap the memref descriptor in a DataView for access
    DataView<size_t, 1> gates_view(data_aligned, offset, sizes, strides);

    if (ppr_basis) {
        auto result = eval_ross_algorithm_ppr(theta, epsilon);
        const auto &gates = result.first;

        size_t s = gates.size();
        RT_FAIL_IF(gates_view.size() < s, "Error: memref allocated too small for PPR gates.\n")

        for (size_t i = 0; i < s; ++i) {
            gates_view(i) = static_cast<size_t>(gates[i]);
        }
    }
    else {
        auto result = eval_ross_algorithm(theta, epsilon);
        const auto &gates = result.first;

        size_t s = gates.size();
        RT_FAIL_IF(gates_view.size() < s, "Error: memref allocated too small for PPR gates.\n")

        for (size_t i = 0; i < s; ++i) {
            gates_view(i) = static_cast<size_t>(gates[i]);
        }
    }
}

double rs_decomposition_get_phase(double theta, double epsilon, bool ppr_basis)
{
    if (ppr_basis) {
        return eval_ross_algorithm_ppr(theta, epsilon).second;
    }
    else {
        return eval_ross_algorithm(theta, epsilon).second;
    }
}

} // extern "C"

} // namespace RSDecomp::RossSelinger
