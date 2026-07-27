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

#include "mlir/IR/IRMapping.h"

#include "QuantumCache.hpp"

namespace catalyst {
namespace quantum {

/// Generate the forward pass of the adjoint region, i.e. the classical portion (forwards).
///
/// This generates the forward computation: Classical preprocessing is cloned to the current
/// insertion point, and the information needed to deterministically replay the circuit in
/// reverse (gate parameters, dynamic wires, and control-flow structure) is recorded into the
/// `QuantumCache`. The reverse pass later consumes this cache.
void generateAdjointForwardPass(mlir::Region &region, mlir::OpBuilder &builder,
                                mlir::IRMapping &oldToCloned, QuantumCache &cache);

/// Generate the reverse pass of the adjoint region, i.e. the quantum portion (in reverse).
///
/// This generates the reverse computation: the quantum operations are cloned in reverse order
/// with the adjoint attribute applied, with any control flow reversed, and using recorded
/// values from the cache.
/// The `IRMapping` helps associating values in the original program to their counterpart in the
/// reversed program. It must be seeded with the adjoint operands before calling.
mlir::LogicalResult generateAdjointReversePass(mlir::Region &region, mlir::OpBuilder &builder,
                                               mlir::IRMapping &remappedValues,
                                               QuantumCache &cache);

} // namespace quantum
} // namespace catalyst
