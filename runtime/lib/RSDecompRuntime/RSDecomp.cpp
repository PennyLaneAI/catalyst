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
#include <cstring>
#include <vector>

#include "DataView.hpp"
#include "GridProblems.hpp"
#include "NormSolver.hpp"
#include "NormalForms.hpp"
#include "RSDecomp.hpp"
#include "Rings.hpp"

#define MAX_SEARCH_TRIALS 10000
#define ROSS_CACHE_SIZE 10000

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

/**
 * @brief Core function to compute the Clifford+T decomposition using the Ross-Selinger algorithm.
 * @param angle The target rotation angle.
 * @param epsilon The desired approximation precision.
 * @return A pair containing the sequence of GateType representing the decomposition and the global
 * phase.
 */
std::pair<std::vector<GateType>, double> compute_clifford_T_decomposition(double angle,
                                                                          double epsilon)
{
    ZOmega scale(0, 0, 0, 1);
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
            decomposition.emplace_back(GateType::Z);
        }
        if (normalized & 2) {
            decomposition.emplace_back(GateType::S);
        }
        if (normalized & 1) {
            decomposition.emplace_back(GateType::T);
        }

        if (decomposition.empty()) {
            decomposition.emplace_back(GateType::I);
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
        GridProblem::GridIterator u_solutions(modified_angle + shift, epsilon, MAX_SEARCH_TRIALS);

        for (const auto &[u_sol, k_val] : u_solutions) {
            // Calculate 2^k_val as an INT_TYPE
            INT_TYPE two_pow_k = INT_TYPE(1) << k_val;
            auto xi = ZSqrtTwo(two_pow_k, 0) - u_sol.norm2().to_sqrt_two();
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
        std::tie(decomposition, phase) = ma_normal_form(so3_mat);
    }
    return {std::move(decomposition), phase};
}

// Cache for Standard Basis
using StdCacheKey = std::tuple<double, double>;
using StdCacheValue = std::pair<std::vector<GateType>, double>;
static lru_cache<StdCacheKey, StdCacheValue, ROSS_CACHE_SIZE> ross_cache_std;

std::pair<std::vector<GateType>, double> eval_ross_algorithm(double angle, double epsilon)
{
    StdCacheKey key = {angle, epsilon};

    if (auto val_opt = ross_cache_std.get(key); val_opt) {
        return *val_opt;
    }

    auto result = compute_clifford_T_decomposition(angle, epsilon);
    ross_cache_std.put(key, result);
    return result;
}

// Cache for PPR Basis
using PPRCacheKey = std::tuple<double, double>;
using PPRCacheValue = std::pair<std::vector<PPRGateType>, double>;
static lru_cache<PPRCacheKey, PPRCacheValue, ROSS_CACHE_SIZE> ross_cache_ppr;

std::pair<std::vector<PPRGateType>, double> eval_ross_algorithm_ppr(double angle, double epsilon)
{
    PPRCacheKey key = {angle, epsilon};

    if (auto val_opt = ross_cache_ppr.get(key); val_opt) {
        return *val_opt;
    }

    auto [gates, phase] = compute_clifford_T_decomposition(angle, epsilon);

    auto [ppr_gates, ppr_phase_update] = HST_to_PPR(gates);

    PPRCacheValue result = {std::move(ppr_gates), phase + ppr_phase_update};

    ross_cache_ppr.put(key, result);
    return result;
}

/**
 * @brief Try to convert a pair of gates for HST_to_PPR
 * @return std::pair<bool, double> true if a pair rule matched and was appended; false otherwise.
 * and the global phase update.
 */
std::pair<bool, double> try_append_pair_expansion(std::vector<PPRGateType> &out, GateType current,
                                                  GateType next)
{
    // We only care if the first gate is HT or SHT
    if (current != GateType::HT && current != GateType::SHT) {
        return {false, 0.0};
    }

    // Rule: HT, HT -> X8, Z8
    if (current == GateType::HT && next == GateType::HT) {
        out.insert(out.end(), {PPRGateType::X8, PPRGateType::Z8});
        return {true, -M_PI / 4.0};
    }

    // Rule: HT, SHT -> X4, X8, Z8
    if (current == GateType::HT && next == GateType::SHT) {
        out.insert(out.end(), {PPRGateType::X4, PPRGateType::X8, PPRGateType::Z8});
        return {true, -M_PI / 2.0};
    }

    // Rule: SHT, HT -> Z4, X8, Z8
    if (current == GateType::SHT && next == GateType::HT) {
        out.insert(out.end(), {PPRGateType::Z4, PPRGateType::X8, PPRGateType::Z8});
        return {true, -M_PI / 2.0};
    }

    // Rule: SHT, SHT -> Z4, X4, X8, Z8
    if (current == GateType::SHT && next == GateType::SHT) {
        out.insert(out.end(), {PPRGateType::Z4, PPRGateType::X4, PPRGateType::X8, PPRGateType::Z8});
        return {true, -3 * M_PI / 4.0};
    }

    return {false, 0.0};
}

/**
 * @brief Convert single gate for HST_to_PPR
 */
double append_single_gate_expansion(std::vector<PPRGateType> &out, GateType gate)
{
    switch (gate) {
    case GateType::T:
        out.emplace_back(PPRGateType::Z8);
        return -M_PI / 8.0;
    case GateType::I:
        out.emplace_back(PPRGateType::I);
        return 0.0;
    case GateType::X:
        out.emplace_back(PPRGateType::X2);
        return -M_PI / 2.0;
    case GateType::Y:
        out.emplace_back(PPRGateType::Y2);
        return -M_PI / 2.0;
    case GateType::Z:
        out.emplace_back(PPRGateType::Z2);
        return -M_PI / 2.0;
    case GateType::H:
        out.insert(out.end(), {PPRGateType::Z4, PPRGateType::X4, PPRGateType::Z4});
        return -M_PI / 2.0;
    case GateType::S:
        out.emplace_back(PPRGateType::Z4);
        return -M_PI / 4.0;
    case GateType::Sd:
        out.emplace_back(PPRGateType::adjZ4);
        return M_PI / 4.0;
    case GateType::HT:
        // Applied commutation rules via PPR playground
        out.insert(out.end(), {PPRGateType::X8, PPRGateType::Z4, PPRGateType::X4, PPRGateType::Z4});
        return -5 * M_PI / 8.0;
    case GateType::SHT:
        // Applied commutation rules via PPR playground
        out.insert(out.end(),
                   {PPRGateType::adjY8, PPRGateType::adjX4, PPRGateType::Z4, PPRGateType::Z2});
        return -7 * M_PI / 8.0;
    default:
        RT_FAIL("Unknown GateType encountered.");
    }
}

/**
 * @brief Converts a sequence of GateType in Clifford+T basis to PPR basis
 * using predefined conversion rules.
 * @param input_gates The input vector of GateType representing the Clifford+T sequence.
 * @return std::pair<std::vector<PPRGateType>, double> The converted vector of PPRGateType and
 * the global phase update.
 */
std::pair<std::vector<PPRGateType>, double> HST_to_PPR(const std::vector<GateType> &input_gates)
{
    std::vector<PPRGateType> output_gates;
    output_gates.reserve(input_gates.size() * 2);
    double phase_update = 0;

    size_t i = 0;
    while (i < input_gates.size()) {
        // Try to consume a pair
        if (i + 1 < input_gates.size()) {
            if (auto [success, phase] =
                    try_append_pair_expansion(output_gates, input_gates[i], input_gates[i + 1]);
                success) {
                phase_update += phase;
                i += 2; // Consumed two gates
                continue;
            }
        }

        // Fallback to consuming a single gate
        phase_update += append_single_gate_expansion(output_gates, input_gates[i]);
        i += 1; // Consumed one gate
    }

    return {output_gates, phase_update};
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
        const auto &[gates, phase] = eval_ross_algorithm_ppr(theta, epsilon);
        size_t s = gates.size();
        RT_FAIL_IF(gates_view.size() < s, "Error: memref allocated too small for PPR gates.\n")

        for (size_t i = 0; i < s; ++i) {
            gates_view(i) = static_cast<size_t>(gates[i]);
        }
    }
    else {
        const auto &[gates, phase] = eval_ross_algorithm(theta, epsilon);

        size_t s = gates.size();
        RT_FAIL_IF(gates_view.size() < s, "Error: memref allocated too small for PPR gates.\n")

        for (size_t i = 0; i < s; ++i) {
            gates_view(i) = static_cast<size_t>(gates[i]);
        }
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
