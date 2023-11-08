// Copyright 2023 Xanadu Quantum Technologies Inc.

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

namespace catalyst {
namespace quantum {

/// A collection of the data required to reconstruct a deterministic hybrid quantum program with
/// classical preprocessing and arbitrary classical control flow.
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

class AugmentedCircuitGenerator {
  public:
    AugmentedCircuitGenerator(mlir::IRMapping &oldToCloned, QuantumCache &cache)
        : oldToCloned(oldToCloned), cache(cache)
    {
    }

    /// Given a `region` containing classical preprocessing and quantum operations, generate an
    /// augmented version that caches all the parameters required to deterministically re-execute
    /// the circuit (gate params, classical control flow, and dynamic wires).
    void generate(mlir::Region &region, mlir::OpBuilder &builder);

  private:
    mlir::IRMapping &oldToCloned;
    QuantumCache &cache;

    void visitOperation(mlir::scf::ForOp forOp, mlir::OpBuilder &builder);
    void visitOperation(mlir::scf::WhileOp forOp, mlir::OpBuilder &builder);
    void visitOperation(mlir::scf::IfOp forOp, mlir::OpBuilder &builder);

    void cloneTerminatorClassicalOperands(mlir::Operation *terminator, mlir::OpBuilder &builder);

    /// Update the internal mapping of the results of `oldOp` to the results of `clonedOp` using the
    /// given result remapping.
    void mapResults(mlir::Operation *oldOp, mlir::Operation *clonedOp,
                    const mlir::DenseMap<unsigned, unsigned> &argIdxMapping);

    // Emit an operation to cache a dynamic wire for quantum.insert/extract ops.
    template <typename IndexingOp> void cacheDynamicWire(IndexingOp op, mlir::OpBuilder &builder)
    {
        if (!op.getIdxAttr().has_value()) {
            builder.create<ListPushOp>(op.getLoc(), oldToCloned.lookupOrDefault(op.getIdx()),
                                       cache.wireVector);
        }
    }

    void cacheGate(quantum::ParametrizedGate, mlir::OpBuilder &builder);
};

void verifyTypeIsCacheable(mlir::Type ty, mlir::Operation *gate);

} // namespace quantum
} // namespace catalyst
