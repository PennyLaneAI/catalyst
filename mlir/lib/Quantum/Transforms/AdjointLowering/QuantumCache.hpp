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
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

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
    mlir::TypedValue<ArrayListType> intVector;
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

/// Verify that `ty` is a type the cache knows how to record: an f64 or integer (<= 64-bit) scalar,
/// a tensor of f64 or integer (<= 64-bit) elements, or a 2D tensor of complex<f64>. Emits an error
/// on `op` and returns failure otherwise.
mlir::LogicalResult verifyTypeIsCacheable(mlir::Type ty, mlir::Operation *op);

/// Returns true if `param`, a gate parameter used inside `adjointRegion`, is already available
/// when the reverse pass emits its gates, and therefore does not need to be recorded in the cache.
///
/// This holds in two cases:
///   - `param` is defined outside `adjointRegion`. It dominates the adjoint operation, hence it
///     also dominates everything the forward and reverse passes emit in its place.
///
///   - `param` is defined at the immediate top level of `adjointRegion`. These classical ops
///     are cloned during forward-pass emission. Because reverse-pass operations are emitted
///     only after the forward pass completes, their cloned results dominate and can be reused
///     directly. They values also have no nested control flow dependence.
///
/// It does not hold for values defined inside nested control flow, since the forward pass rebuilds
/// those regions and their values are neither visible nor loop-invariant: they must be recorded.
bool isAvailableToReversePass(mlir::Value param, mlir::Region &adjointRegion);

} // namespace quantum
} // namespace catalyst
