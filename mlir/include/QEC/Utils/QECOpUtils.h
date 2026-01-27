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

#include "QEC/IR/QECOpInterfaces.h"

namespace catalyst {
namespace qec {

// Return the SSA value of `qubit` visible at the program point of `op`.
// Walk back through QEC PPR chains (out->in) until defined before `op` or a
// block argument. Returns nullptr if a non-QEC op defines it.
mlir::Value getReachingValueAt(mlir::Value qubit, QECOpInterface op);

// For each in-qubit of `srcOp`, return the SSA value visible at `dstOp`.
// One per operand; nullptr if a non-QEC definition intervenes.
// Precondition: same block with `dstOp` before `srcOp`. Consider caching.
std::vector<mlir::Value> getInQubitReachingValuesAt(QECOpInterface srcOp, QECOpInterface dstOp);

// Check if `rhsOp` commutes with `lhsOp` at the program point of `lhsOp`.
// This checks commutativity of the normalized ops with the same block.
bool commutes(QECOpInterface rhsOp, QECOpInterface lhsOp);

// Recursively resolve the constant parameter of a value and returns std::nullopt if not a constant.
std::optional<double> resolveConstantValue(mlir::Value value);

} // namespace qec
} // namespace catalyst
