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
#define ROSS_CACHE_SIZE 10000
// Minimum success probability |u|^2 of a fallback projective rotation, and the share of the
// diamond-norm error budget given to the projective rotations (fallbacks are rare, so they can use
// the remainder at a much looser accuracy). These minimize the expected T-count; with 0.99 and 0.5
// it is about 2 T gates higher. The resource hints in
// mlir/lib/Quantum/Transforms/GridsynthPatterns.cpp, e.g. `fallbackFailureProbability`, are fitted
// for these values.
#define FALLBACK_SUCCESS_PROBABILITY 0.95
#define FALLBACK_PROJECTIVE_SHARE 0.9
#define MAX_FALLBACK_EPSILON 0.25
#define MAX_FALLBACK_LEVELS 200

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
 *
 * Applied first to last, the gate sequence implements [[u, -v*], [v, u*]] with u = u_sol /
 * sqrt(2)^k. If `v_out` is given, it receives v.
 * @return The gate sequence and global phase, or std::nullopt if the norm equation has no solution.
 */
std::optional<std::pair<std::vector<GateType>, double>>
normal_form_from_grid_solution(const ZOmega &u_sol, int k_val, const ZOmega &scale,
                               std::complex<double> *v_out = nullptr) {
    INT_TYPE two_pow_k = INT_TYPE(1) << k_val;
    auto xi = ZSqrtTwo(two_pow_k, 0) - u_sol.norm2().to_sqrt_two();
    auto t_sol = NormSolver::solve_diophantine(xi, MAX_FACTORING_TRIALS);
    if (!t_sol) {
        return std::nullopt;
    }

    ZOmega u = u_sol * scale;
    ZOmega t = *t_sol * scale;
    if (v_out) {
        // The normal form of [[u, -t*], [t, u*]] implements its transpose, [[u, t], [-t*, u*]].
        *v_out = -std::conj(t.to_complex()) / std::pow(M_SQRT2, k_val);
    }
    DyadicMatrix dyd_mat(u, -t.conj(), t, u.conj(), INT_TYPE(k_val));
    SO3Matrix so3_mat(dyd_mat);
    return ma_normal_form(so3_mat);
}

// Grid-search accuracy for which every candidate satisfies 1 - Re(w)^2 <= epsilon / 2 (written to
// avoid cancellation).
double mixed_search_epsilon(double epsilon) {
    return std::sqrt(epsilon / (1.0 + std::sqrt(1.0 - epsilon / 2.0)));
}

/**
 * @brief Power k of diag(1, i^k) for the diagonal Cliffords I, S, Z, S^dagger, or -1 otherwise.
 */
int diagonal_clifford_power(GateType gate) {
    switch (gate) {
    case GateType::I:
        return 0;
    case GateType::S:
        return 1;
    case GateType::Z:
        return 2;
    case GateType::Sd:
        return 3;
    default:
        return -1;
    }
}

/**
 * @brief Append `gate` to `gates`, merging it into the last gate when the product is exactly
 * (including global phase) a single gate.
 */
void append_merged(std::vector<GateType> &gates, GateType gate) {
    if (gates.empty()) {
        gates.push_back(gate);
        return;
    }
    GateType &last = gates.back();
    int last_power = diagonal_clifford_power(last);
    int power = diagonal_clifford_power(gate);
    if (last_power >= 0 && power >= 0) {
        static constexpr std::array<GateType, 4> powers = {GateType::I, GateType::S, GateType::Z,
                                                           GateType::Sd};
        last = powers[(last_power + power) % 4];
        if (last == GateType::I) {
            gates.pop_back();
        }
    } else if (last == GateType::S && gate == GateType::HT) {
        last = GateType::SHT;
    } else if (last == GateType::Sd && gate == GateType::SHT) {
        last = GateType::HT;
    } else {
        gates.push_back(gate);
    }
}

void warn_small_epsilon(double epsilon, const std::string &min_epsilon) {
    std::ostringstream oss;
    oss << std::scientific << epsilon;
    RT_WARN("Gridsynth received epsilon=" + oss.str() + ". For epsilon smaller than " +
            min_epsilon +
            ", results may be inaccurate, or errors may occur during decomposition. To guarantee "
            "correctness, please provide a larger epsilon value.");
}

/**
 * @brief Write a gate sequence into a memref passed as its unpacked descriptor fields.
 */
template <typename Gate>
void fill_gates(size_t *data_aligned, size_t offset, size_t size0, size_t stride0,
                const std::vector<Gate> &gates) {
    const size_t sizes[1] = {size0};
    const size_t strides[1] = {stride0};
    DataView<size_t, 1> gates_view(data_aligned, offset, sizes, strides);
    RT_FAIL_IF(gates_view.size() < gates.size(), "Error: memref allocated too small for gates.\n")
    for (size_t i = 0; i < gates.size(); ++i) {
        gates_view(i) = static_cast<size_t>(gates[i]);
    }
}

/**
 * @brief Select the branch and twirl of `mixed` from a uniform sample and return its sequence.
 */
std::pair<std::vector<GateType>, double> sample_twirled_sequence(const MixedDecomposition &mixed,
                                                                 double uniform_sample) {
    auto [branch, twirl] = sample_mixed_decomposition(mixed.probability, uniform_sample);
    return {twirl_sequence(mixed.gates[branch], twirl), mixed.phases[branch]};
}

std::pair<std::vector<PPRGateType>, double>
sequence_to_ppr(const std::pair<std::vector<GateType>, double> &sequence) {
    auto [ppr_gates, ppr_phase_update] = HST_to_PPR(sequence.first);
    return {std::move(ppr_gates), sequence.second + ppr_phase_update};
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
            result = normal_form_from_grid_solution(u_sol, k_val, scale);
            if (result) {
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

size_t count_t_gates(const std::vector<GateType> &gates) {
    return std::count_if(gates.begin(), gates.end(), [](GateType gate) {
        return gate == GateType::T || gate == GateType::HT || gate == GateType::SHT;
    });
}

namespace {
/**
 * @brief Search for the cheapest mixture of an under-rotating and an over-rotating candidate.
 *
 * Writing a candidate's top-left entry as `w * exp(-i angle/2)`, it under-rotates the target if
 * Im(w) <= 0 and over-rotates it if Im(w) >= 0. A pair is mixed with the probability p of
 * Equation (33) of arXiv:2203.10064, and admissible if `diamond_error(p, w_under, w_over)` is at
 * most `epsilon`. Of the admissible pairs found, the one with the lowest expected T-count is kept.
 */
class MixedPairSearch {
  public:
    struct Candidate {
        std::vector<GateType> gates;
        double phase;
        std::complex<double> w;
        // Bottom-left entry of the unitary the gate sequence implements.
        std::complex<double> v;
        size_t t_count;
    };
    using DiamondError = double (*)(double p, std::complex<double> w_under,
                                    std::complex<double> w_over);

    MixedPairSearch(const ZOmega &scale, double epsilon, DiamondError diamond_error)
        : scale(scale), epsilon(epsilon), diamond_error(diamond_error) {}

    /**
     * @brief Consider the grid solution `u_sol / sqrt(2)^k` with relative top-left entry `w`.
     * @return false if the search should stop.
     */
    bool offer(const ZOmega &u_sol, int k_val, std::complex<double> w) {
        bool is_under = w.imag() <= 0.0;
        bool is_over = w.imag() >= 0.0;
        // Until the other side has a candidate, more candidates on a filled side cannot form a
        // pair, so skip their (expensive) norm equation.
        bool same_side_filled = is_under ? !unders.empty() : !overs.empty();
        bool other_side_empty = is_under ? overs.empty() : unders.empty();
        if (!best && !(is_under && is_over) && same_side_filled && other_side_empty) {
            return true;
        }
        if (unders.size() + overs.size() >= MAX_MIXED_CANDIDATES) {
            return false;
        }
        std::complex<double> v;
        auto normal_form = normal_form_from_grid_solution(u_sol, k_val, scale, &v);
        if (!normal_form) {
            return true;
        }
        size_t t_count = count_t_gates(normal_form->first);
        // Heuristic stop: T-counts mostly, but not strictly, grow along the search.
        if (best && t_count > best->cost) {
            return false;
        }

        Candidate candidate{std::move(normal_form->first), normal_form->second, w, v, t_count};
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
        return true;
    }

    bool found() const { return best.has_value(); }
    const Candidate &under() const { return unders[best->under]; }
    const Candidate &over() const { return overs[best->over]; }
    double probability() const { return best->probability; }

  private:
    struct Pair {
        size_t under;
        size_t over;
        double probability;
        double cost;
    };

    void evaluate(size_t i, size_t j) {
        const Candidate &under = unders[i];
        const Candidate &over = overs[j];
        // p = r2^2 sin(2 delta2) / (r2^2 sin(2 delta2) - r1^2 sin(2 delta1)), with
        // r^2 sin(2 delta) = 2 Re(w) Im(w).
        double under_weight = under.w.real() * under.w.imag();
        double over_weight = over.w.real() * over.w.imag();
        double denominator = over_weight - under_weight;
        double p = denominator > 0.0 ? over_weight / denominator : 1.0;
        double diamond = diamond_error(p, under.w, over.w);
        // Tolerate rounding for candidates on the bound.
        if (diamond > epsilon * (1.0 + 1e-12)) {
            return;
        }
        double cost = p * under.t_count + (1.0 - p) * over.t_count;
        if (!best || cost < best->cost) {
            best = Pair{i, j, p, cost};
        }
    }

    ZOmega scale;
    double epsilon;
    DiamondError diamond_error;
    std::vector<Candidate> unders;
    std::vector<Candidate> overs;
    std::optional<Pair> best;
};
} // namespace

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

    // Each candidate satisfies the even split 1 - Re(w)^2 <= epsilon/2 of Proposition 3.13.
    const double min_overlap = std::sqrt(1.0 - epsilon / 2.0);
    const std::complex<double> inverse_target = std::polar(1.0, angle / 2.0);

    auto [grid_angle, scale] = grid_problem_frame(angle);
    GridProblem::GridIterator u_solutions(grid_angle, mixed_search_epsilon(epsilon),
                                          MAX_SEARCH_TRIALS);
    // Theorem 3.12.
    MixedPairSearch search(scale, epsilon,
                           [](double p, std::complex<double> w_under, std::complex<double> w_over) {
                               return 2.0 * (1.0 - p * w_under.real() * w_under.real() -
                                             (1.0 - p) * w_over.real() * w_over.real());
                           });

    for (const auto &[u_sol, k_val] : u_solutions) {
        std::complex<double> w =
            (u_sol * scale).to_complex() * inverse_target / std::pow(M_SQRT2, k_val);
        if (w.real() < min_overlap) {
            continue;
        }
        if (!search.offer(u_sol, k_val, w)) {
            break;
        }
    }

    if (!search.found()) {
        // A unitary with operator-norm error epsilon/2 has diamond-norm error at most epsilon.
        auto fallback = compute_clifford_T_decomposition(angle, epsilon / 2.0);
        return {{fallback.first, fallback.first}, {fallback.second, fallback.second}, 1.0};
    }

    return {{search.under().gates, search.over().gates},
            {search.under().phase, search.over().phase},
            search.probability()};
}

/**
 * @brief Select the branch and twirl of a mixed decomposition from a uniform sample in [0, 1).
 * @return The pair (branch, twirl).
 */
std::pair<size_t, size_t> sample_mixed_decomposition(double probability, double uniform_sample) {
    double u = std::clamp(uniform_sample, 0.0, std::nextafter(1.0, 0.0));
    size_t branch = u < probability ? 0 : 1;
    double fraction = branch == 0 ? u / probability : (u - probability) / (1.0 - probability);
    size_t twirl = std::min<size_t>(3, static_cast<size_t>(4.0 * fraction));
    return {branch, twirl};
}

/**
 * @brief Apply the {Z, S} twirl sigma U sigma^dagger to a gate sequence (applied first to last),
 * merging the twirl gates into adjacent gates where possible.
 */
std::vector<GateType> twirl_sequence(const std::vector<GateType> &gates, size_t twirl) {
    // (sigma^dagger, sigma) for sigma in {I, S, Z, S^dagger}.
    static constexpr std::array<std::pair<GateType, GateType>, 4> twirls = {{
        {GateType::I, GateType::I},
        {GateType::Sd, GateType::S},
        {GateType::Z, GateType::Z},
        {GateType::S, GateType::Sd},
    }};
    assert(twirl < twirls.size() && "twirl index must select one of {I, S, Z, S^dagger}");
    if (twirl == 0) {
        return gates;
    }
    std::vector<GateType> twirled;
    twirled.reserve(gates.size() + 2);
    twirled.push_back(twirls[twirl].first);
    for (GateType gate : gates) {
        append_merged(twirled, gate);
    }
    append_merged(twirled, twirls[twirl].second);
    if (twirled.empty()) {
        twirled.push_back(GateType::I);
    }
    return twirled;
}

// Cache for mixed decompositions
using MixedCacheKey = std::tuple<double, double>;
static lru_cache<MixedCacheKey, MixedDecomposition, ROSS_CACHE_SIZE> mixed_cache;

/**
 * @brief Sample a twirled sequence of the mixed decomposition of RZ(angle).
 *
 * `epsilon` has the meaning of `eval_ross_algorithm`: the mixed channel has diamond-norm error at
 * most 2 * epsilon, the guarantee of a single sequence with operator-norm error epsilon.
 */
std::pair<std::vector<GateType>, double> eval_mixed_ross_algorithm(double angle, double epsilon,
                                                                   double uniform_sample) {
    MixedCacheKey key = {angle, epsilon};
    auto mixed = mixed_cache.get(key);
    if (!mixed) {
        mixed = compute_mixed_diagonal_decomposition(angle, 2.0 * epsilon);
        mixed_cache.put(key, *mixed);
    }
    return sample_twirled_sequence(*mixed, uniform_sample);
}

std::pair<std::vector<PPRGateType>, double>
eval_mixed_ross_algorithm_ppr(double angle, double epsilon, double uniform_sample) {
    return sequence_to_ppr(eval_mixed_ross_algorithm(angle, epsilon, uniform_sample));
}

/**
 * @brief Compute a mixed fallback approximation of RZ(angle) with diamond-norm accuracy epsilon.
 *
 * Implements the mixed fallback protocol of Kliuchnikov et al., arXiv:2203.10064, Section 3.5
 * (Proposition 3.18 and Theorem 3.17). Each projective rotation V = [[u, -v*], [v, u*]] acts on an
 * ancilla between two CY gates controlled by the target. Measuring the ancilla in the Z basis
 * applies exp(i Arg(u) Z) to the target with probability |u|^2 and exp(i Arg(v) Z) otherwise
 * (Section 3.3), in which case a mixed diagonal fallback B ~ exp(i (-angle/2 - Arg(v)) Z) follows.
 * An under-rotating and an over-rotating projective rotation are mixed with probabilities p and
 * 1 - p. Of the candidate pairs found, the one with the lowest expected T-count of the projective
 * rotations is kept, as the fallbacks only add (1 - |u|^2) times their T-count.
 */
MixedFallbackDecomposition compute_mixed_fallback_decomposition(double angle, double epsilon) {
    if (is_odd_multiple_of_pi_4(angle)) {
        // The exact diagonal V never fails.
        auto [gates, phase] = compute_clifford_T_decomposition(angle, epsilon);
        FallbackBranch exact{gates, phase, 1.0, 0.0, epsilon};
        return {{exact, exact}, 1.0};
    }

    const double min_overlap = std::sqrt(FALLBACK_SUCCESS_PROBABILITY);
    // Each candidate satisfies sin|Arg(u) - target|^2 <= epsilon_1 / 2 of Proposition 3.18.
    const double projective_epsilon = FALLBACK_PROJECTIVE_SHARE * epsilon;
    const double max_sin = std::sqrt(projective_epsilon / 2.0);
    const std::complex<double> inverse_target = std::polar(1.0, angle / 2.0);
    auto [grid_angle, scale] = grid_problem_frame(angle);
    // Equation (34), with q sin(delta)^2 = Im(w)^2.
    MixedPairSearch::DiamondError projective_diamond = [](double p, std::complex<double> w_under,
                                                          std::complex<double> w_over) {
        return 2.0 *
               (p * w_under.imag() * w_under.imag() + (1.0 - p) * w_over.imag() * w_over.imag());
    };
    MixedPairSearch search(scale, projective_epsilon, projective_diamond);

    GridProblem::enumerate_sector_solutions(
        grid_angle, std::asin(max_sin), min_overlap, MAX_FALLBACK_LEVELS,
        [&](const ZOmega &u_sol, int k_val) {
            std::complex<double> w =
                (u_sol * scale).to_complex() * inverse_target / std::pow(M_SQRT2, k_val);
            return search.offer(u_sol, k_val, w);
        });

    if (!search.found()) {
        // V = -iY = [[0, -1], [1, 0]] always fails, which leaves the full accuracy to the fallback
        // of RZ(angle).
        FallbackBranch always_fail{{GateType::Y}, M_PI / 2, 0.0, 0.0, epsilon};
        return {{always_fail, always_fail}, 1.0};
    }

    // Equation (36): split the remaining accuracy evenly between the two fallback terms
    // p_k (1 - q_k) ||B_k - Z_theta||.
    const double fallback_budget =
        (epsilon - projective_diamond(search.probability(), search.under().w, search.over().w)) /
        2.0;
    auto make_branch = [&](const MixedPairSearch::Candidate &candidate, double weight) {
        double success = std::norm(candidate.w);
        double failure_weight = weight * (1.0 - success);
        double fallback_epsilon =
            failure_weight > 0.0 ? std::min(fallback_budget / failure_weight, MAX_FALLBACK_EPSILON)
                                 : MAX_FALLBACK_EPSILON;
        return FallbackBranch{candidate.gates, candidate.phase, success, std::arg(candidate.v),
                              fallback_epsilon};
    };
    return {{make_branch(search.under(), search.probability()),
             make_branch(search.over(), 1.0 - search.probability())},
            search.probability()};
}

// Cache for mixed fallback decompositions
static lru_cache<MixedCacheKey, MixedFallbackDecomposition, ROSS_CACHE_SIZE> fallback_cache;

/**
 * @brief The branch of the mixed fallback decomposition of RZ(angle) selected by `uniform_sample`,
 * with `epsilon` as in `eval_mixed_ross_algorithm`.
 */
static FallbackBranch eval_fallback_branch(double angle, double epsilon, double uniform_sample) {
    MixedCacheKey key = {angle, epsilon};
    auto fallback = fallback_cache.get(key);
    if (!fallback) {
        fallback = compute_mixed_fallback_decomposition(angle, 2.0 * epsilon);
        fallback_cache.put(key, *fallback);
    }
    size_t branch = sample_mixed_decomposition(fallback->probability, uniform_sample).first;
    return fallback->branches[branch];
}

std::pair<std::vector<GateType>, double> eval_fallback_projective(double angle, double epsilon,
                                                                  double uniform_sample) {
    FallbackBranch branch = eval_fallback_branch(angle, epsilon, uniform_sample);
    return {std::move(branch.projective_gates), branch.projective_phase};
}

std::pair<std::vector<PPRGateType>, double>
eval_fallback_projective_ppr(double angle, double epsilon, double uniform_sample) {
    return sequence_to_ppr(eval_fallback_projective(angle, epsilon, uniform_sample));
}

// Extern C implementation
extern "C" {

size_t rs_decomposition_get_size(double theta, double epsilon, bool ppr_basis) {
    if (epsilon < 1e-6) {
        warn_small_epsilon(epsilon, "1e-6");
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
    if (ppr_basis) {
        fill_gates(data_aligned, offset, size0, stride0,
                   eval_ross_algorithm_ppr(theta, epsilon).first);
    } else {
        fill_gates(data_aligned, offset, size0, stride0, eval_ross_algorithm(theta, epsilon).first);
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
 * @param epsilon Error, as in `rs_decomposition_get_size`
 * @param ppr_basis Whether to use PPR basis
 * @param uniform_sample Uniform random number in [0, 1) selecting the branch and twirl
 */
size_t rs_mixed_decomposition_get_size(double theta, double epsilon, bool ppr_basis,
                                       double uniform_sample) {
    // The mixed grid search runs at an accuracy of about sqrt(epsilon).
    if (epsilon < 1e-12) {
        warn_small_epsilon(epsilon, "1e-12");
    }
    if (ppr_basis) {
        return eval_mixed_ross_algorithm_ppr(theta, epsilon, uniform_sample).first.size();
    }
    return eval_mixed_ross_algorithm(theta, epsilon, uniform_sample).first.size();
}

/**
 * @brief Fills a pre-allocated memref with the sampled mixed gate sequence.
 *
 * The memref is unpacked into individual arguments, as in `rs_decomposition_get_gates`.
 *
 * @param data_allocated Pointer to allocated data
 * @param data_aligned Pointer to aligned data
 * @param offset Data offset
 * @param size0 Size of dimension 0
 * @param stride0 Stride of dimension 0
 * @param theta Angle
 * @param epsilon Error, as in `rs_decomposition_get_size`
 * @param ppr_basis Whether to use PPR basis
 * @param uniform_sample Uniform random number in [0, 1) selecting the branch and twirl
 */
void rs_mixed_decomposition_get_gates([[maybe_unused]] size_t *data_allocated, size_t *data_aligned,
                                      size_t offset, size_t size0, size_t stride0, double theta,
                                      double epsilon, bool ppr_basis, double uniform_sample) {
    if (ppr_basis) {
        fill_gates(data_aligned, offset, size0, stride0,
                   eval_mixed_ross_algorithm_ppr(theta, epsilon, uniform_sample).first);
    } else {
        fill_gates(data_aligned, offset, size0, stride0,
                   eval_mixed_ross_algorithm(theta, epsilon, uniform_sample).first);
    }
}

/**
 * @brief Returns the global phase of the sampled mixed decomposition.
 *
 * @param theta Angle
 * @param epsilon Error, as in `rs_decomposition_get_size`
 * @param ppr_basis Whether to use PPR basis
 * @param uniform_sample Uniform random number in [0, 1) selecting the branch and twirl
 * @return double The global phase
 */
double rs_mixed_decomposition_get_phase(double theta, double epsilon, bool ppr_basis,
                                        double uniform_sample) {
    if (ppr_basis) {
        return eval_mixed_ross_algorithm_ppr(theta, epsilon, uniform_sample).second;
    }
    return eval_mixed_ross_algorithm(theta, epsilon, uniform_sample).second;
}

/**
 * @brief Returns the length of the sampled projective rotation of a mixed fallback decomposition.
 *
 * The `rs_fallback_*` functions are deterministic in `uniform_sample`, so calling them with the
 * same sample refers to the same projective rotation.
 *
 * @param theta Angle
 * @param epsilon Error, as in `rs_decomposition_get_size`
 * @param ppr_basis Whether to use PPR basis
 * @param uniform_sample Uniform random number in [0, 1) selecting the projective rotation
 */
size_t rs_fallback_projective_get_size(double theta, double epsilon, bool ppr_basis,
                                       double uniform_sample) {
    // The projective rotation search runs at an accuracy of about sqrt(epsilon).
    if (epsilon < 1e-12) {
        warn_small_epsilon(epsilon, "1e-12");
    }
    if (ppr_basis) {
        return eval_fallback_projective_ppr(theta, epsilon, uniform_sample).first.size();
    }
    return eval_fallback_projective(theta, epsilon, uniform_sample).first.size();
}

/**
 * @brief Fills a pre-allocated memref with the sampled projective rotation sequence.
 *
 * The memref is unpacked into individual arguments, as in `rs_decomposition_get_gates`.
 */
void rs_fallback_projective_get_gates([[maybe_unused]] size_t *data_allocated, size_t *data_aligned,
                                      size_t offset, size_t size0, size_t stride0, double theta,
                                      double epsilon, bool ppr_basis, double uniform_sample) {
    if (ppr_basis) {
        fill_gates(data_aligned, offset, size0, stride0,
                   eval_fallback_projective_ppr(theta, epsilon, uniform_sample).first);
    } else {
        fill_gates(data_aligned, offset, size0, stride0,
                   eval_fallback_projective(theta, epsilon, uniform_sample).first);
    }
}

/**
 * @brief Returns the global phase of the sampled projective rotation sequence.
 */
double rs_fallback_projective_get_phase(double theta, double epsilon, bool ppr_basis,
                                        double uniform_sample) {
    if (ppr_basis) {
        return eval_fallback_projective_ppr(theta, epsilon, uniform_sample).second;
    }
    return eval_fallback_projective(theta, epsilon, uniform_sample).second;
}

/**
 * @brief Returns the angle of the RZ rotation that the fallback approximates when the projective
 * rotation selected by `uniform_sample` fails.
 *
 * The fallback is the mixed decomposition of the `rs_mixed_decomposition_*` functions for this
 * angle and the epsilon of `rs_fallback_recovery_epsilon`.
 *
 * @param theta Angle
 * @param epsilon Error, as in `rs_decomposition_get_size`
 * @param uniform_sample Uniform random number in [0, 1) selecting the projective rotation
 */
double rs_fallback_recovery_angle(double theta, double epsilon, double uniform_sample) {
    // B ~ exp(i (-theta/2 - Arg(v)) Z) = RZ(theta + 2 Arg(v)).
    return theta + 2.0 * eval_fallback_branch(theta, epsilon, uniform_sample).failure_angle;
}

/**
 * @brief Returns the epsilon of the fallback for `rs_fallback_recovery_angle`, in the convention of
 * `rs_mixed_decomposition_get_size`.
 */
double rs_fallback_recovery_epsilon(double theta, double epsilon, double uniform_sample) {
    return eval_fallback_branch(theta, epsilon, uniform_sample).fallback_epsilon / 2.0;
}

} // extern "C"

} // namespace RSDecomp::RossSelinger
