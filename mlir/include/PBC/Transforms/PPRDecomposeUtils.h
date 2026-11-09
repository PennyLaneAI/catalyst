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

#ifndef PBC_TRANSFORMS_DECOMPOSE_UTILS_H
#define PBC_TRANSFORMS_DECOMPOSE_UTILS_H

#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"

#include "PBC/IR/PBCOps.h"

namespace catalyst {
namespace pbc {

/// Determine the Pauli measurement type and rotation sign
/// based on whether to avoid Pauli Y measurements
/// 1. avoidPauliYMeasure == true: Use |Y⟩ as axillary qubit and measure -P⊗Z
/// 2. avoidPauliYMeasure == false: Use |0⟩ as axillary qubit and measure P⊗Y
std::pair<mlir::StringRef, bool> determinePauliAndSignOfMeasurement(bool avoidPauliYMeasure);

/// Initialize |0⟩ or Fabricate|Y⟩ based on avoidPauliYMeasure
mlir::OpResult initializeZeroOrPlusI(bool avoidPauliYMeasure, mlir::Location loc,
                                     mlir::PatternRewriter &rewriter);

/// Replace `op` with a `pbc.gadget` that holds a copy of it, carrying `op` as the gadget's
/// source rotation, and return the copy. The copy is then decomposed in place, inside the gadget.
/// `op` must have no condition.
PPRotationOp wrapInGadget(PPRotationOp op, mlir::PatternRewriter &rewriter);

/// Whether a decomposition of `op` should go into its own gadget: gadgets are asked for, and `op`
/// is not already part of one.
bool needsGadget(PPRotationOp op, bool emitGadgets);

} // namespace pbc
} // namespace catalyst

#endif // PBC_TRANSFORMS_DECOMPOSE_UTILS_H
