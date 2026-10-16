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
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "DataView.hpp"
#include "GridProblems.hpp"
#include "NormSolver.hpp"
#include "NormalForms.hpp"
#include "Real.hpp"
#include "Rings.hpp"

#define ROSS_CACHE_SIZE 10000
// Factoring attempts per candidate. Candidates whose norm is hard to factor are skipped, since the
// next levels of the grid problem provide plenty of further candidates.
#define CANDIDATE_FACTORING_TRIALS 2
#define MAX_CANDIDATES 100000
#define MAX_MIXED_CANDIDATES 64

namespace RSDecomp::RossSelinger {

using namespace RSDecomp::Rings;
using namespace RSDecomp::Reals;
using namespace RSDecomp::Utils;
using namespace RSDecomp::CliffordData;
using namespace RSDecomp::NormalForms;

namespace {

/**
 * @brief The unitary [[u, -t^†], [t, u^†]] / √2^k approximating RZ(angle).
 */
struct RzSolution {
    ZOmega u;
    ZOmega t;
    int k;
};

/**
 * @brief Find the integer number of units of π/4 that the angle is equal to, within an error
 * that keeps the decomposition within epsilon.
 * @param angle The target rotation angle.
 * @param epsilon The desired approximation precision.
 * @return The number of units of π/4, or std::nullopt if the angle is not close enough to one.
 */
std::optional<INT_TYPE> as_multiple_of_pi_4(double angle, double epsilon) {
    Real pi_4 = pi() / 4;
    INT_TYPE units = round_int(Real(angle / pi_4));
    // ||RZ(a) - RZ(b)|| <= |a - b| / 2
    if (abs(Real(angle - Real(units) * pi_4)) <= 2 * Real(epsilon)) {
        return units;
    }
    return std::nullopt;
}

/**
 * @brief Solve the Diophantine equation |t|^2 = 2^k - |u|^2 that completes u to a unitary.
 * @param u A solution of the grid problem at level k.
 * @param k The level.
 * @return t, or std::nullopt if no solution was found.
 */
std::optional<ZOmega> solve_norm_equation(const ZOmega &u, int k) {
    ZSqrtTwo xi = ZSqrtTwo(INT_TYPE(1) << k, 0) - u.norm2().to_sqrt_two();
    auto t = NormSolver::solve_diophantine(xi, CANDIDATE_FACTORING_TRIALS);
    if (t && t->norm2() == xi.to_omega()) {
        return t;
    }
    return std::nullopt;
}

/**
 * @brief The Clifford+T normal form and global phase of an RzSolution.
 */
std::pair<std::vector<GateType>, double> normal_form(const RzSolution &solution) {
    const auto &[u, t, k] = solution;
    SO3Matrix so3_mat(DyadicMatrix(u, -t.conj(), t, u.conj(), INT_TYPE(k)));
    return ma_normal_form(so3_mat);
}

/**
 * @brief Solve the grid problem and Diophantine equation for RZ(angle) at level k = 0, 1, 2, ....
 * @param angle The target rotation angle.
 * @param epsilon The desired approximation precision.
 * @return The solution at the smallest level, or std::nullopt if none was found up to the
 * maximum level.
 */
std::optional<RzSolution> solve_rz(double angle, double epsilon) {
    // RZ(angle)[0, 0] = e^{-i angle / 2}
    GridProblem::RzGridSearch search(-Real(angle) / 2, Real(epsilon));

    // Solutions are expected from level ~1.5 * log2(1/ε), and the number of candidates grows
    // fourfold with every further level.
    int max_k = static_cast<int>(1.5 * std::log2(std::max(1.0 / epsilon, 2.0))) + 40;
    size_t num_candidates = 0;
    std::optional<RzSolution> solution;

    for (int k = 0; k <= max_k && !solution; k++) {
        search.visit_level(k, [&](const ZOmega &u) {
            if (auto t = solve_norm_equation(u, k)) {
                solution = RzSolution{u, *t, k};
                return false;
            }
            RT_FAIL_IF(++num_candidates > MAX_CANDIDATES,
                       "Gridsynth exceeded the maximum number of candidate solutions.");
            return true;
        });
    }
    return solution;
}

/**
 * @brief Fail for inputs that have no decomposition. This must happen before the cache lookups,
 * as NaN keys compare equal to any cached key.
 * @param angle The target rotation angle.
 * @param epsilon The desired approximation precision.
 */
void validate_inputs(double angle, double epsilon) {
    RT_FAIL_IF(!std::isfinite(angle), "Gridsynth requires a finite angle.");
    RT_FAIL_IF(!(epsilon > 0) || !std::isfinite(epsilon),
               "Gridsynth requires a positive, finite epsilon.");
    if (epsilon < MIN_EPSILON) {
        std::ostringstream oss;
        oss << "Gridsynth received epsilon=" << std::scientific << epsilon
            << ", but the smallest supported epsilon is " << std::defaultfloat << MIN_EPSILON
            << ".";
        RT_FAIL(oss.str().c_str());
    }
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
    if (auto units = as_multiple_of_pi_4(angle, epsilon)) {
        std::vector<GateType> decomposition;
        int normalized = (((*units % 8) + 8) % 8).convert_to<int>();

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

        return {std::move(decomposition), units->convert_to<double>() * (M_PI / 8.0)};
    }

    auto solution = solve_rz(angle, epsilon);
    if (!solution) {
        std::ostringstream oss;
        oss << "Gridsynth failed to find a decomposition for angle=" << angle
            << " and epsilon=" << std::scientific << epsilon << ".";
        RT_FAIL(oss.str().c_str());
    }
    return normal_form(*solution);
}

// Cache for Standard Basis
using StdCacheKey = std::tuple<double, double>;
using StdCacheValue = std::pair<std::vector<GateType>, double>;
static lru_cache<StdCacheKey, StdCacheValue, ROSS_CACHE_SIZE> ross_cache_std;

std::pair<std::vector<GateType>, double> eval_ross_algorithm(double angle, double epsilon) {
    validate_inputs(angle, epsilon);
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
    validate_inputs(angle, epsilon);
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
 * Equation (33) of arXiv:2203.10064, and admissible if the diamond-norm error of the mixture
 * (Theorem 3.12) is at most `epsilon`. Of the admissible pairs found, the one with the lowest
 * expected T-count is kept.
 */
class MixedPairSearch {
  public:
    struct Candidate {
        std::vector<GateType> gates;
        double phase;
        std::complex<double> w;
        // 1 - Re(w)^2, computed in multiprecision since it is of order epsilon.
        double infidelity;
        size_t t_count;
    };

    explicit MixedPairSearch(double epsilon) : epsilon(epsilon) {}

    /**
     * @brief Consider the grid solution `u / sqrt(2)^k` with relative top-left entry `w`.
     * @return false if the search should stop.
     */
    bool offer(const ZOmega &u, int k, std::complex<double> w, double infidelity) {
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
        auto t = solve_norm_equation(u, k);
        if (!t) {
            return true;
        }
        auto [gates, phase] = normal_form(RzSolution{u, *t, k});
        size_t t_count = count_t_gates(gates);
        // Heuristic stop: T-counts mostly, but not strictly, grow along the search.
        if (best && t_count > best->cost) {
            return false;
        }

        Candidate candidate{std::move(gates), phase, w, infidelity, t_count};
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
        // Theorem 3.12: 2 (1 - p Re(w_under)^2 - (1 - p) Re(w_over)^2).
        double diamond = 2.0 * (p * under.infidelity + (1.0 - p) * over.infidelity);
        // Tolerate rounding for candidates on the bound.
        if (diamond > epsilon * (1.0 + 1e-12)) {
            return;
        }
        double cost = p * under.t_count + (1.0 - p) * over.t_count;
        if (!best || cost < best->cost) {
            best = Pair{i, j, p, cost};
        }
    }

    double epsilon;
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
 * candidate's top-left entry as `w * exp(-i angle/2)`, an under-rotation (Im(w) <= 0) and an
 * over-rotation (Im(w) >= 0) are mixed with probabilities p and 1 - p, and each is
 * twirled over {I, S, Z, S^dagger} when applied. Of the candidate pairs found, the one with the
 * lowest expected T-count whose mixture meets the diamond-norm bound of Theorem 3.12 is kept.
 */
MixedDecomposition compute_mixed_diagonal_decomposition(double angle, double epsilon) {
    // A unitary with operator-norm error epsilon/2 has diamond-norm error at most epsilon.
    if (as_multiple_of_pi_4(angle, epsilon / 2.0)) {
        auto exact = compute_clifford_T_decomposition(angle, epsilon / 2.0);
        return {{exact.first, exact.first}, {exact.second, exact.second}, 1.0};
    }

    // RZ(angle)[0, 0] = e^{-i angle / 2}. Every candidate of the search satisfies the even split
    // 1 - Re(w)^2 <= epsilon/2 of Proposition 3.13.
    double search_epsilon = mixed_search_epsilon(epsilon);
    Real theta = -Real(angle) / 2;
    Real zx = cos(theta);
    Real zy = sin(theta);
    GridProblem::RzGridSearch grid_search(theta, Real(search_epsilon));
    MixedPairSearch search(epsilon);

    int max_k = static_cast<int>(1.5 * std::log2(1.0 / search_epsilon)) + 40;
    size_t num_candidates = 0;
    bool searching = true;
    for (int k = 0; k <= max_k && searching; k++) {
        Real scale = pow(sqrt2(), k);
        searching = grid_search.visit_level(k, [&](const ZOmega &u) {
            if (++num_candidates > MAX_CANDIDATES) {
                return false;
            }
            // w = u * conj(z) / sqrt(2)^k with z = e^{i theta}.
            Real re = (zx * real_part(u) + zy * imag_part(u)) / scale;
            Real im = (zx * imag_part(u) - zy * real_part(u)) / scale;
            Real infidelity = 1 - re * re;
            return search.offer(u, k, {re.convert_to<double>(), im.convert_to<double>()},
                                infidelity.convert_to<double>());
        });
    }

    if (!search.found()) {
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
 * @brief Apply the twirl sigma U sigma^dagger, sigma in {I, S, Z, S^dagger}, to a gate sequence
 * (applied first to last), merging the twirl gates into adjacent gates where possible.
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
    validate_inputs(angle, epsilon);
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

// Extern C implementation
extern "C" {

size_t rs_decomposition_get_size(double theta, double epsilon, bool ppr_basis) {
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

} // extern "C"

} // namespace RSDecomp::RossSelinger
