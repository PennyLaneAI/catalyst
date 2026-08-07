// Copyright 2026 Xanadu Quantum Technologies Inc.

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
#include <cstdint>

namespace catalyst::transport::common {

/**
 * @brief Number of stabilizer checks in the [[7,1,3]] Steane code.
 */
constexpr int STEANE_CHECKS = 3;

/**
 * @brief Syndrome-to-error-qubit lookup for the [[7,1,3]] Steane code; -1 is
 * no error. Indexed by the 3 checks packed with check 0 as the MSB.
 *
 * Single source of truth for the hard-coded Steane decoder shared by the CPU-
 * and GPU-verbs transport backends. Correctness against the frontend Steane
 * definition (qec_code_lib.py) is enforced by
 * runtime/tests/Test_transport_steane_LUT_decoder.cpp.
 */
constexpr std::int64_t STEANE_SYNDROME_TO_QUBIT[1 << STEANE_CHECKS] = {-1, 6, 4, 5, 0, 3, 1, 2};

} // namespace catalyst::transport::common
