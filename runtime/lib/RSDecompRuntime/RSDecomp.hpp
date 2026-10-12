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

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "CliffordData.hpp"

namespace RSDecomp::RossSelinger {
using namespace RSDecomp::Rings;
using namespace RSDecomp::CliffordData;
std::pair<std::vector<GateType>, double> eval_ross_algorithm(double angle, double epsilon);
std::pair<std::vector<PPRGateType>, double> eval_ross_algorithm_ppr(double angle, double epsilon);
std::pair<std::vector<PPRGateType>, double> HST_to_PPR(const std::vector<GateType> &vector);

/**
 * @brief Two gate sequences and their global phases, applied with probabilities
 * `probability` and `1 - probability`. Each is {Z, S}-twirled when applied.
 */
struct MixedDecomposition {
    std::array<std::vector<GateType>, 2> gates;
    std::array<double, 2> phases;
    double probability;
};

size_t count_t_gates(const std::vector<GateType> &gates);
MixedDecomposition compute_mixed_diagonal_decomposition(double angle, double epsilon);
std::pair<size_t, size_t> sample_mixed_decomposition(double probability, double uniform_sample);
std::vector<GateType> twirl_sequence(const std::vector<GateType> &gates, size_t twirl);
std::pair<std::vector<GateType>, double> eval_mixed_ross_algorithm(double angle, double epsilon,
                                                                   double uniform_sample);
std::pair<std::vector<PPRGateType>, double>
eval_mixed_ross_algorithm_ppr(double angle, double epsilon, double uniform_sample);

/**
 * @brief A projective rotation V = [[u, -v*], [v, u*]] of a mixed fallback decomposition of
 * RZ(angle), and the fallback applied when it fails.
 */
struct FallbackBranch {
    std::vector<GateType> projective_gates;
    double projective_phase;
    // |u|^2
    double success_probability;
    // Arg(v). On failure, exp(i Arg(v) Z) is applied to the target, followed by the mixed diagonal
    // decomposition of RZ(angle + 2 Arg(v)) with diamond-norm error `fallback_epsilon`.
    double failure_angle;
    double fallback_epsilon;
};

/**
 * @brief Two fallback branches, applied with probabilities `probability` and `1 - probability`.
 */
struct MixedFallbackDecomposition {
    std::array<FallbackBranch, 2> branches;
    double probability;
};

MixedFallbackDecomposition compute_mixed_fallback_decomposition(double angle, double epsilon);
std::pair<std::vector<GateType>, double> eval_fallback_projective(double angle, double epsilon,
                                                                  double uniform_sample);
std::pair<std::vector<PPRGateType>, double>
eval_fallback_projective_ppr(double angle, double epsilon, double uniform_sample);

extern "C" {

size_t rs_decomposition_get_size(double theta, double epsilon, bool ppr_basis);

void rs_decomposition_get_gates(size_t *data_allocated, size_t *data_aligned, size_t offset,
                                size_t size0, size_t stride0, double theta, double epsilon,
                                bool ppr_basis);

double rs_decomposition_get_phase(double theta, double epsilon, bool ppr_basis);

size_t rs_mixed_decomposition_get_size(double theta, double epsilon, bool ppr_basis,
                                       double uniform_sample);

void rs_mixed_decomposition_get_gates(size_t *data_allocated, size_t *data_aligned, size_t offset,
                                      size_t size0, size_t stride0, double theta, double epsilon,
                                      bool ppr_basis, double uniform_sample);

double rs_mixed_decomposition_get_phase(double theta, double epsilon, bool ppr_basis,
                                        double uniform_sample);

size_t rs_fallback_projective_get_size(double theta, double epsilon, bool ppr_basis,
                                       double uniform_sample);

void rs_fallback_projective_get_gates(size_t *data_allocated, size_t *data_aligned, size_t offset,
                                      size_t size0, size_t stride0, double theta, double epsilon,
                                      bool ppr_basis, double uniform_sample);

double rs_fallback_projective_get_phase(double theta, double epsilon, bool ppr_basis,
                                        double uniform_sample);

double rs_fallback_recovery_angle(double theta, double epsilon, double uniform_sample);

double rs_fallback_recovery_epsilon(double theta, double epsilon, double uniform_sample);

} // extern "C"
} // namespace RSDecomp::RossSelinger
