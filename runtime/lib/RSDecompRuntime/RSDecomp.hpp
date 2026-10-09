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

struct MixedSample {
    size_t branch;
    size_t twirl;
};

MixedDecomposition compute_mixed_diagonal_decomposition(double angle, double epsilon);
MixedSample sample_mixed_decomposition(double probability, double uniform_sample);
std::vector<GateType> twirl_sequence(const std::vector<GateType> &gates, size_t twirl);
std::pair<std::vector<GateType>, double> eval_mixed_ross_algorithm(double angle, double epsilon,
                                                                   double uniform_sample);
std::pair<std::vector<PPRGateType>, double>
eval_mixed_ross_algorithm_ppr(double angle, double epsilon, double uniform_sample);

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

} // extern "C"
} // namespace RSDecomp::RossSelinger
