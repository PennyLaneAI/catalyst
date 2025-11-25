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

#include "CliffordData.hpp"
#include <cstdint>

namespace RSDecomp::RossSelinger {
using namespace RSDecomp::Rings;
using namespace RSDecomp::CliffordData;
// C++ Interface
std::pair<std::vector<GateType>, double> eval_ross_algorithm(double angle, double epsilon);
std::pair<std::vector<PPRGateType>, double> eval_ross_algorithm_ppr(double angle, double epsilon);

// C Interface (typically for MLIR/Python bindings)
extern "C" {

int64_t rs_decomposition_get_size_0(double theta, double epsilon, bool ppr_basis);

void rs_decomposition_get_gates_0(int64_t *data_allocated, int64_t *data_aligned, size_t offset,
                                  size_t size0, size_t stride0, double theta, double epsilon,
                                  bool ppr_basis);

double rs_decomposition_get_phase_0(double theta, double epsilon, bool ppr_basis);

} // extern "C"
} // namespace RSDecomp::RossSelinger
