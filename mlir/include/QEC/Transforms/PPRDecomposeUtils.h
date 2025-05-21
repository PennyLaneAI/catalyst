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

#ifndef QEC_TRANSFORMS_DECOMPOSE_UTILS_H
#define QEC_TRANSFORMS_DECOMPOSE_UTILS_H

#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace catalyst {
namespace qec {

/// Determine the Pauli measurement type and rotation sign
/// based on whether to avoid Pauli Y measurements
/// 1. avoidPauliYMeasure == true: Use |Y⟩ as axillary qubit and measure -P⊗Z
/// 2. avoidPauliYMeasure == false: Use |0⟩ as axillary qubit and measure P⊗Y
std::pair<mlir::StringRef, uint16_t>
determinePauliAndRotationSignOfMeasurement(bool avoidPauliYMeasure);

/// Initialize |0⟩ or Fabricate|Y⟩ based on avoidPauliYMeasure
mlir::OpResult initializeZeroOrPlusI(bool avoidPauliYMeasure, mlir::Location loc,
                                     mlir::PatternRewriter &rewriter);

} // namespace qec
} // namespace catalyst

#endif // QEC_TRANSFORMS_DECOMPOSE_UTILS_H
