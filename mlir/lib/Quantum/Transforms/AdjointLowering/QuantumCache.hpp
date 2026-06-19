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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "Catalyst/IR/CatalystDialect.h"

namespace catalyst {
namespace quantum {

/// A collection of the data required to reconstruct a deterministic hybrid quantum program with
/// classical preprocessing and arbitrary classical control flow.
///
/// The forward pass populates these tapes (see ForwardPass.hpp) and the reverse pass consumes them
/// (see ReversePass.hpp). The push/pop order is a shared contract between the two passes: values
/// are pushed in program order during the forward pass and popped in reverse during the backward
/// pass.
struct QuantumCache {
    mlir::TypedValue<ArrayListType> paramVector;
    mlir::TypedValue<ArrayListType> wireVector;
    /// For every structured control flow op, store the values required for it to execute.
    /// Specifically: store the conditions for scf.if ops, the start/stop/step of scf.for ops, and
    /// the number of iterations for scf.while ops.
    mlir::DenseMap<mlir::Operation *, mlir::TypedValue<ArrayListType>> controlFlowTapes;

    /// Initialize the quantum cache to traverse and store the necessary parameters for the given
    /// `topLevelRegion`.
    static QuantumCache initialize(mlir::Region &topLevelRegion, mlir::OpBuilder &builder,
                                   mlir::Location loc);

    void emitDealloc(mlir::OpBuilder &builder, mlir::Location loc);
};

/// Verify that `ty` is a type the cache knows how to record (an f64 scalar or a 2D tensor of
/// complex<f64>), emitting an error on `op` otherwise.
void verifyTypeIsCacheable(mlir::Type ty, mlir::Operation *op);

} // namespace quantum
} // namespace catalyst
