// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "Mitigation/IR/MitigationOps.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;

namespace catalyst {
namespace mitigation {

struct ZneLowering : public OpRewritePattern<mitigation::ZneOp> {
    using OpRewritePattern<mitigation::ZneOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mitigation::ZneOp op, PatternRewriter &rewriter) const override;

  private:
    /// Resolve the folded version of the `zne` op's callee, creating it if it does not exist
    /// yet. If the callee is a qnode, its `.folded` counterpart is returned directly. Otherwise
    /// the callee's call graph is traversed: every classical function gets a `.zne` copy with
    /// the fold count (of type `foldCountType`) appended as its last argument, qnode calls are
    /// redirected to their `.folded` versions, and the callee's own `.zne` copy is returned.
    static func::FuncOp getOrCreateFoldedCallee(Location loc, PatternRewriter &rewriter,
                                                mitigation::ZneOp op, func::FuncOp calleeOp,
                                                Folding foldingAlgorithm, Type foldCountType);

    /// Emit the loop replacing the `zne` op: iterate over the `numScaleFactors` entries of the
    /// `numFolds` tensor, call `fnFoldedOp` once per entry with the fold count appended to the
    /// original arguments, and collect the results into a tensor of type `resultType`. With
    /// `randomFolding` the fold count is passed through as an f64 (it may carry a fractional
    /// part); otherwise it is cast to an index.
    static Value buildFoldedResultsLoop(Location loc, PatternRewriter &rewriter,
                                        mitigation::ZneOp op, func::FuncOp fnFoldedOp,
                                        Value numFolds, bool randomFolding, int64_t numScaleFactors,
                                        RankedTensorType resultType);

    /// Get or create the `.folded` version of the qnode `op`, which takes the fold count as an
    /// extra final argument and applies the folding dictated by `foldingAlgorithm`.
    static FlatSymbolRefAttr getOrInsertFoldedCircuit(Location loc, PatternRewriter &builder,
                                                      func::FuncOp op, Folding foldingAlgorithm);
    /// Get or create the `.quantumAlloc` helper for `op`, which allocates a quantum register of
    /// the requested size (used by global folding).
    static FlatSymbolRefAttr getOrInsertQuantumAlloc(Location loc, PatternRewriter &rewriter,
                                                     func::FuncOp op);
    /// Get or create the `.withoutMeasurements` version of the qnode `op`: the circuit body
    /// without terminal measurements, taking and returning the quantum register (used by global
    /// folding to build the repeated `U U^dagger` blocks).
    static FlatSymbolRefAttr
    getOrInsertFnWithoutMeasurements(Location loc, PatternRewriter &rewriter, func::FuncOp op);
    /// Get or create the `.withMeasurements` version of the qnode `op`: the full circuit
    /// including measurements, operating on a quantum register passed as the last argument
    /// (used by global folding as the final application of the circuit).
    static FlatSymbolRefAttr getOrInsertFnWithMeasurements(Location loc, PatternRewriter &rewriter,
                                                           func::FuncOp op);
};

} // namespace mitigation
} // namespace catalyst
