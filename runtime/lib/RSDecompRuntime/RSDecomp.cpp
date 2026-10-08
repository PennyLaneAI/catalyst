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

#include "RSDecomp.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "DataView.hpp"
#include "GridProblems.hpp"
#include "NormSolver.hpp"
#include "NormalForms.hpp"
#include "Rings.hpp"

#define MAX_SEARCH_TRIALS 10000
#define MAX_MIXED_CANDIDATES 64
// Mixed synthesis search: candidates may each have up to MIXED_LOOSENESS * epsilon / 2 error
// (1 is the even split), and the search continues MIXED_SEARCH_MARGIN T gates past the best
// pair. Increasing either slightly lowers the T-count at a large cost in synthesis time.
#define MIXED_LOOSENESS 1.0
#define MIXED_SEARCH_MARGIN 0.0
#define ROSS_CACHE_SIZE 10000

namespace {
bool is_odd_multiple_of_pi_4(double angle) {
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

namespace {
/**
 * @brief Reduce the grid-problem angle -angle/2 into [-pi/4, pi/4].
 * @return The reduced angle and the scale that maps grid solutions back to the original frame.
 */
std::pair<double, ZOmega> grid_problem_frame(double angle) {
    double modified_angle = -angle / 2.0;
    long k = std::lround(modified_angle / M_PI_2);
    double shift = -static_cast<double>(k) * M_PI_2;
    int idx = ((k % 4) + 4) % 4;

    ZOmega scale(0, 0, 0, 1);
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
    return {modified_angle + shift, scale};
}

/**
 * @brief Complete a grid solution `u_sol / sqrt(2)^k` to a unitary and compute its normal form.
 * @return The gate sequence and global phase, or std::nullopt if the norm equation has no solution.
 */
std::optional<std::pair<std::vector<GateType>, double>>
normal_form_from_grid_solution(const ZOmega &u_sol, int k_val, const ZOmega &scale) {
    INT_TYPE two_pow_k = INT_TYPE(1) << k_val;
    auto xi = ZSqrtTwo(two_pow_k, 0) - u_sol.norm2().to_sqrt_two();
    auto t_sol = NormSolver::solve_diophantine(xi, MAX_FACTORING_TRIALS);
    if (!t_sol) {
        return std::nullopt;
    }

    ZOmega u = u_sol * scale;
    ZOmega t = *t_sol * scale;
    DyadicMatrix dyd_mat(u, -t.conj(), t, u.conj(), INT_TYPE(k_val));
    SO3Matrix so3_mat(dyd_mat);
    return ma_normal_form(so3_mat);
}

size_t count_t_gates(const std::vector<GateType> &gates) {
    return std::count_if(gates.begin(), gates.end(), [](GateType gate) {
        return gate == GateType::T || gate == GateType::HT || gate == GateType::SHT;
    });
}
} // namespace

/**
 * @brief Core function to compute the Clifford+T decomposition using the Ross-Selinger algorithm.
 * @param angle The target rotation angle.
 * @param epsilon The desired approximation precision.
 * @return A pair containing the sequence of GateType representing the decomposition and the global
 * phase.
 */
std::pair<std::vector<GateType>, double> compute_clifford_T_decomposition(double angle,
                                                                          double epsilon) {
    double phase = 0.0;
    std::vector<GateType> decomposition;

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
    } else {
        auto [grid_angle, scale] = grid_problem_frame(angle);
        GridProblem::GridIterator u_solutions(grid_angle, epsilon, MAX_SEARCH_TRIALS);

        std::optional<std::pair<std::vector<GateType>, double>> result;
        for (const auto &[u_sol, k_val] : u_solutions) {
            if ((result = normal_form_from_grid_solution(u_sol, k_val, scale))) {
                break;
            }
        }

        if (!result) {
            DyadicMatrix identity(ZOmega(0, 0, 0, 1), ZOmega(0), ZOmega(0), ZOmega(0, 0, 0, 1),
                                  INT_TYPE(0));
            result = ma_normal_form(SO3Matrix(identity));
        }
        std::tie(decomposition, phase) = *result;
    }
    return {std::move(decomposition), phase};
}

// Cache for Standard Basis
using StdCacheKey = std::tuple<double, double>;
using StdCacheValue = std::pair<std::vector<GateType>, double>;
static lru_cache<StdCacheKey, StdCacheValue, ROSS_CACHE_SIZE> ross_cache_std;

std::pair<std::vector<GateType>, double> eval_ross_algorithm(double angle, double epsilon) {
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

std::pair<std::vector<PPRGateType>, double> eval_ross_algorithm_ppr(double angle, double epsilon) {
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
                                                  GateType next) {
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
 * @brief Convert single gate for HST_to_PPR and return the global phase update.
 */
double append_single_gate_expansion(std::vector<PPRGateType> &out, GateType gate) {
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
std::pair<std::vector<PPRGateType>, double> HST_to_PPR(const std::vector<GateType> &input_gates) {
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

/**
 * @brief Compute a mixed diagonal approximation of RZ(angle) with diamond-norm accuracy epsilon.
 *
 * Implements the mixed diagonal approximation of Kliuchnikov et al., "Shorter quantum circuits via
 * single-qubit gate approximation", Quantum 7, 1208 (2023), arXiv:2203.10064, Section 3.4
 * (Proposition 3.13, with the diamond-norm bound of Theorem 3.12). Writing each
 * candidate's top-left entry as `w * exp(-i angle/2)`, an under-rotation (Im(w) < 0) and an
 * over-rotation (Im(w) > 0) are mixed with probabilities p and 1 - p, and each is
 * {Z, S}-twirled when applied. Of the candidate pairs found, the one with the lowest expected
 * T-count whose mixture meets the diamond-norm bound of Theorem 3.12 is kept.
 */
MixedDecomposition compute_mixed_diagonal_decomposition(double angle, double epsilon) {
    if (is_odd_multiple_of_pi_4(angle)) {
        auto exact = compute_clifford_T_decomposition(angle, epsilon);
        return {{exact.first, exact.first}, {exact.second, exact.second}, 1.0};
    }

    // With MIXED_LOOSENESS > 1, candidates may individually miss the even split
    // 1 - Re(w)^2 <= epsilon/2 of Proposition 3.13, as long as the pair satisfies the exact bound
    // of Theorem 3.12.
    const double min_overlap = std::sqrt(1.0 - MIXED_LOOSENESS * epsilon / 2.0);
    // Operator-norm accuracy of the grid search whose target region is Re(w) >= min_overlap,
    // i.e. 1 - search_epsilon^2 / 2 = min_overlap, written to avoid cancellation.
    const double search_epsilon = std::sqrt(MIXED_LOOSENESS * epsilon / (1.0 + min_overlap));
    const std::complex<double> inverse_target = std::polar(1.0, angle / 2.0);

    auto [grid_angle, scale] = grid_problem_frame(angle);
    GridProblem::GridIterator u_solutions(grid_angle, search_epsilon, MAX_SEARCH_TRIALS);

    struct Candidate {
        std::vector<GateType> gates;
        double phase;
        std::complex<double> w;
        size_t t_count;
    };
    std::vector<Candidate> unders;
    std::vector<Candidate> overs;

    struct Pair {
        size_t under;
        size_t over;
        double probability;
        double cost;
    };
    std::optional<Pair> best;

    auto evaluate = [&](size_t i, size_t j) {
        const Candidate &under = unders[i];
        const Candidate &over = overs[j];
        // p = r2^2 sin(2 delta2) / (r2^2 sin(2 delta2) - r1^2 sin(2 delta1)), with
        // r^2 sin(2 delta) = 2 Re(w) Im(w).
        double under_weight = under.w.real() * under.w.imag();
        double over_weight = over.w.real() * over.w.imag();
        double denominator = over_weight - under_weight;
        double p = denominator > 0.0 ? over_weight / denominator : 1.0;
        double diamond = 2.0 * (1.0 - p * under.w.real() * under.w.real() -
                                (1.0 - p) * over.w.real() * over.w.real());
        if (diamond > epsilon) {
            return;
        }
        double cost = p * under.t_count + (1.0 - p) * over.t_count;
        if (!best || cost < best->cost) {
            best = Pair{i, j, p, cost};
        }
    };

    for (const auto &[u_sol, k_val] : u_solutions) {
        std::complex<double> w =
            (u_sol * scale).to_complex() * inverse_target / std::pow(M_SQRT2, k_val);
        if (w.real() < min_overlap) {
            continue;
        }

        bool is_under = w.imag() <= 0.0;
        bool is_over = w.imag() >= 0.0;
        // Until both sides have a candidate, further candidates on a filled side cannot form the
        // first pair, so skip their (expensive) norm equation.
        if (!best && (!is_under || !unders.empty()) && (!is_over || !overs.empty())) {
            continue;
        }
        if (unders.size() + overs.size() >= MAX_MIXED_CANDIDATES) {
            break;
        }
        auto normal_form = normal_form_from_grid_solution(u_sol, k_val, scale);
        if (!normal_form) {
            continue;
        }
        size_t t_count = count_t_gates(normal_form->first);
        // Later candidates have larger T-counts, so they can no longer improve the best pair.
        if (best && t_count > best->cost + MIXED_SEARCH_MARGIN) {
            break;
        }

        Candidate candidate{std::move(normal_form->first), normal_form->second, w, t_count};
        if (is_under) {
            unders.push_back(candidate);
            for (size_t j = 0; j < overs.size(); j++) {
                evaluate(unders.size() - 1, j);
            }
        }
        if (is_over) {
            overs.push_back(std::move(candidate));
            for (size_t i = 0; i < unders.size(); i++) {
                evaluate(i, overs.size() - 1);
            }
        }
    }

    if (!best) {
        // A unitary with operator-norm error epsilon/2 has diamond-norm error at most epsilon.
        auto fallback = compute_clifford_T_decomposition(angle, epsilon / 2.0);
        return {{fallback.first, fallback.first}, {fallback.second, fallback.second}, 1.0};
    }

    return {{std::move(unders[best->under].gates), std::move(overs[best->over].gates)},
            {unders[best->under].phase, overs[best->over].phase},
            best->probability};
}

namespace {
using MixedCacheKey = std::tuple<double, double>;
lru_cache<MixedCacheKey, MixedDecomposition, ROSS_CACHE_SIZE> mixed_cache;

MixedDecomposition eval_mixed_decomposition(double angle, double epsilon) {
    MixedCacheKey key = {angle, epsilon};
    if (auto val_opt = mixed_cache.get(key); val_opt) {
        return *val_opt;
    }

    auto result = compute_mixed_diagonal_decomposition(angle, epsilon);
    mixed_cache.put(key, result);
    return result;
}
} // namespace

/**
 * @brief Select the branch and twirl of a mixed decomposition from a uniform sample in [0, 1).
 */
MixedSample sample_mixed_decomposition(double probability, double uniform_sample) {
    double u = std::clamp(uniform_sample, 0.0, std::nextafter(1.0, 0.0));
    size_t branch = u < probability ? 0 : 1;
    double fraction = branch == 0 ? u / probability : (u - probability) / (1.0 - probability);
    size_t twirl = std::min<size_t>(3, static_cast<size_t>(4.0 * fraction));
    return {branch, twirl};
}

/**
 * @brief Apply the {Z, S} twirl sigma U sigma^dagger to a gate sequence (applied first to last).
 */
std::vector<GateType> twirl_sequence(const std::vector<GateType> &gates, size_t twirl) {
    // (sigma^dagger, sigma) for sigma in {I, S, Z, S^dagger}.
    static constexpr std::array<std::pair<GateType, GateType>, 4> twirls = {{
        {GateType::I, GateType::I},
        {GateType::Sd, GateType::S},
        {GateType::Z, GateType::Z},
        {GateType::S, GateType::Sd},
    }};
    if (twirl == 0) {
        return gates;
    }
    std::vector<GateType> twirled;
    twirled.reserve(gates.size() + 2);
    twirled.push_back(twirls[twirl].first);
    twirled.insert(twirled.end(), gates.begin(), gates.end());
    twirled.push_back(twirls[twirl].second);
    return twirled;
}

std::pair<std::vector<GateType>, double> eval_mixed_ross_algorithm(double angle, double epsilon,
                                                                   double uniform_sample) {
    MixedDecomposition mixed = eval_mixed_decomposition(angle, epsilon);
    auto [branch, twirl] = sample_mixed_decomposition(mixed.probability, uniform_sample);
    return {twirl_sequence(mixed.gates[branch], twirl), mixed.phases[branch]};
}

std::pair<std::vector<PPRGateType>, double>
eval_mixed_ross_algorithm_ppr(double angle, double epsilon, double uniform_sample) {
    auto [gates, phase] = eval_mixed_ross_algorithm(angle, epsilon, uniform_sample);
    auto [ppr_gates, ppr_phase_update] = HST_to_PPR(gates);
    return {std::move(ppr_gates), phase + ppr_phase_update};
}

// Extern C implementation
extern "C" {

size_t rs_decomposition_get_size(double theta, double epsilon, bool ppr_basis) {
    if (epsilon < 1e-6) {
        std::ostringstream oss;
        oss << std::scientific << epsilon;
        RT_WARN("Gridsynth received epsilon=" + oss.str() +
                ". For epsilon smaller than 1e-6, results may be inaccurate, or errors may"
                " occur during decomposition. To guarantee correctness, please provide a larger "
                "epsilon value.");
    }
    if (ppr_basis) {
        auto result = eval_ross_algorithm_ppr(theta, epsilon);
        return result.first.size();
    } else {
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
                                double epsilon, bool ppr_basis) {
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
    } else {
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
double rs_decomposition_get_phase(double theta, double epsilon, bool ppr_basis) {
    if (ppr_basis) {
        return eval_ross_algorithm_ppr(theta, epsilon).second;
    } else {
        return eval_ross_algorithm(theta, epsilon).second;
    }
}

/**
 * @brief Returns the length of the sampled mixed decomposition.
 *
 * The three `rs_mixed_decomposition_*` functions are deterministic in `uniform_sample`, so calling
 * them with the same sample returns the size, gates, and phase of the same sequence.
 *
 * @param theta Angle
 * @param epsilon Diamond-norm error of the mixed channel
 * @param ppr_basis Whether to use PPR basis
 * @param uniform_sample Uniform random number in [0, 1) selecting the branch and twirl
 */
size_t rs_mixed_decomposition_get_size(double theta, double epsilon, bool ppr_basis,
                                       double uniform_sample) {
    double search_epsilon = std::sqrt(epsilon / (1.0 + std::sqrt(1.0 - epsilon / 2.0)));
    if (search_epsilon < 1e-6) {
        std::ostringstream oss;
        oss << std::scientific << epsilon;
        RT_WARN("Mixed gridsynth received epsilon=" + oss.str() +
                ". For epsilon smaller than 2e-12, results may be inaccurate, or errors may"
                " occur during decomposition. To guarantee correctness, please provide a larger "
                "epsilon value.");
    }
    if (ppr_basis) {
        return eval_mixed_ross_algorithm_ppr(theta, epsilon, uniform_sample).first.size();
    }
    return eval_mixed_ross_algorithm(theta, epsilon, uniform_sample).first.size();
}

/**
 * @brief Fills a pre-allocated memref with the sampled mixed gate sequence.
 *
 * See `rs_decomposition_get_gates` for the memref arguments.
 */
void rs_mixed_decomposition_get_gates([[maybe_unused]] size_t *data_allocated, size_t *data_aligned,
                                      size_t offset, size_t size0, size_t stride0, double theta,
                                      double epsilon, bool ppr_basis, double uniform_sample) {
    const size_t sizes[1] = {size0};
    const size_t strides[1] = {stride0};
    DataView<size_t, 1> gates_view(data_aligned, offset, sizes, strides);

    auto fill = [&](const auto &gates) {
        RT_FAIL_IF(gates_view.size() < gates.size(),
                   "Error: memref allocated too small for mixed gates.\n")
        for (size_t i = 0; i < gates.size(); ++i) {
            gates_view(i) = static_cast<size_t>(gates[i]);
        }
    };

    if (ppr_basis) {
        fill(eval_mixed_ross_algorithm_ppr(theta, epsilon, uniform_sample).first);
    } else {
        fill(eval_mixed_ross_algorithm(theta, epsilon, uniform_sample).first);
    }
}

/**
 * @brief Returns the global phase of the sampled mixed decomposition.
 */
double rs_mixed_decomposition_get_phase(double theta, double epsilon, bool ppr_basis,
                                        double uniform_sample) {
    if (ppr_basis) {
        return eval_mixed_ross_algorithm_ppr(theta, epsilon, uniform_sample).second;
    }
    return eval_mixed_ross_algorithm(theta, epsilon, uniform_sample).second;
}

} // extern "C"

} // namespace RSDecomp::RossSelinger
