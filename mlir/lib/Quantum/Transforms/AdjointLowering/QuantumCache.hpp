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

#include "llvm/ADT/MapVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "Catalyst/IR/CatalystDialect.h"

namespace catalyst {
namespace quantum {

/// How a single gate parameter is laid out in the parameter cache.
enum class ParamStorage {
    /// The parameter holds exactly one value: either it is not a shaped type (an integer, index,
    /// float, complex, ...) or it is a rank-0 tensor. It is pushed to the cache as one element.
    Scalar,
    /// The parameter is a ranked tensor of rank at least 1. All of its elements are pushed to the
    /// cache as a single block, in row-major order.
    WholeTensor,
};

/// The cache layout chosen for a given parameter type.
struct ParamStorageInfo {
    ParamStorage kind;
    /// Element type of the array list that holds this parameter. Always a scalar type, so
    /// parameters that only differ in shape share a list.
    mlir::Type listElementType;
    /// Whether the parameter has dynamic dimensions, which are recorded separately in
    /// `QuantumCache::shapeVector` so that the reverse pass can rebuild the tensor shape.
    bool hasDynamicShape;
};

/// Decide how a parameter of type `ty` is stored, or return failure if it cannot be cached at all.
///
/// Both passes route through this function: the forward pass to emit the pushes, the reverse pass
/// to emit the matching pops, and `QuantumCache::initialize` to create the lists they use. A single
/// source of truth is what keeps the two sequences symmetric.
mlir::FailureOr<ParamStorageInfo> classifyParamStorage(mlir::Type ty);

/// A collection of the data required to reconstruct a deterministic hybrid quantum program with
/// classical preprocessing and arbitrary classical control flow.
///
/// The forward pass populates these tapes (see ForwardPass.hpp) and the reverse pass consumes them
/// (see ReversePass.hpp). The push/pop order is a shared contract between the two passes: values
/// are pushed in program order during the forward pass and popped in reverse during the backward
/// pass.
struct QuantumCache {
    /// One array list per parameter element type, keyed by that element type. Order is preserved
    /// within each list, so spreading parameters of different types over several lists keeps the
    /// push/pop contract intact while allowing parameters of any type to be cached.
    ///
    /// A `MapVector` (rather than a `DenseMap`) keeps the generated IR deterministic, since the
    /// lists are created, and later deallocated, in iteration order.
    llvm::MapVector<mlir::Type, mlir::TypedValue<ArrayListType>> paramVectors;
    /// The dynamic dimensions of cached tensor parameters. This is deliberately not part of
    /// `paramVectors`: an `index`-typed parameter would otherwise share a list with the recorded
    /// dimensions and interleave with them. Null when no cached parameter has a dynamic shape.
    mlir::TypedValue<ArrayListType> shapeVector;
    mlir::TypedValue<ArrayListType> wireVector;
    /// For every structured control flow op, store the values required for it to execute.
    /// Specifically: store the conditions for scf.if ops, the start/stop/step of scf.for ops, and
    /// the number of iterations for scf.while ops.
    mlir::DenseMap<mlir::Operation *, mlir::TypedValue<ArrayListType>> controlFlowTapes;

    /// Initialize the quantum cache to traverse and store the necessary parameters for the given
    /// `topLevelRegion`.
    static QuantumCache initialize(mlir::Region &topLevelRegion, mlir::OpBuilder &builder,
                                   mlir::Location loc);

    /// Record `param`, a value of the augmented circuit emitted by the forward pass, so that the
    /// reverse pass can restore it. `param`'s type must have been accepted by
    /// `verifyTypeIsCacheable`.
    void pushParam(mlir::Value param, mlir::OpBuilder &builder, mlir::Location loc);

    /// Emit the operations restoring the parameter of type `paramType` that `pushParam` recorded,
    /// and return the restored value. Parameters must be popped in the exact reverse of the order
    /// in which they were pushed.
    mlir::Value popParam(mlir::Type paramType, mlir::OpBuilder &builder, mlir::Location loc);

    void emitDealloc(mlir::OpBuilder &builder, mlir::Location loc);
};

/// Verify that `ty` is a type the cache knows how to record. Emits an error on `op` and returns
/// failure otherwise.
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
