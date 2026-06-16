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

#include "AdjointLowering.hpp"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumOps.h"

#include "QuantumCache.hpp"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

/// Orchestrates the adjoint lowering of a single `quantum.adjoint` op by running the forward pass
/// (recording the augmented circuit) followed by the reverse pass (emitting the adjoint circuit).
struct AdjointSingleOpRewritePattern : public OpRewritePattern<AdjointOp> {
    using OpRewritePattern<AdjointOp>::OpRewritePattern;

    /// We build a map from values mentioned in the source data flow to the values of
    /// the program where quantum control flow is reversed. Most of the time, there is a 1-to-1
    /// correspondence with a notable exception caused by `insert`/`extract` API asymmetry.
    LogicalResult matchAndRewrite(AdjointOp adjoint, PatternRewriter &rewriter) const override
    {
        QuantumCache cache =
            QuantumCache::initialize(adjoint.getRegion(), rewriter, adjoint.getLoc());

        // Forward pass: copy the classical computations to the target insertion point and record
        // the values needed to replay the circuit in reverse.
        IRMapping oldToCloned;
        generateAdjointForwardPass(adjoint.getRegion(), rewriter, oldToCloned, cache);

        // Seed the reverse pass with the operands of the quantum.yield.
        auto yieldOp = cast<YieldOp>(adjoint.getRegion().front().getTerminator());
        for (auto [yieldVal, adjointOperand] :
             llvm::zip_equal(yieldOp.getOperands(), adjoint.getArgs())) {
            oldToCloned.map(yieldVal, adjointOperand);
        }

        // Reverse pass: emit the adjoint quantum operations and reversed control flow, using the
        // cached values.
        if (failed(generateAdjointReversePass(adjoint.getRegion(), rewriter, oldToCloned, cache))) {
            return failure();
        }

        // Explicitly free the memory of the caches.
        cache.emitDealloc(rewriter, adjoint.getLoc());
        // The final quantum values are the re-mapped region arguments of the original adjoint op.
        SmallVector<Value> reversedOutputs;
        for (BlockArgument arg : adjoint.getRegion().getArguments()) {
            reversedOutputs.push_back(oldToCloned.lookup(arg));
        }
        rewriter.replaceOp(adjoint, reversedOutputs);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_ADJOINTLOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct AdjointLoweringPass : impl::AdjointLoweringPassBase<AdjointLoweringPass> {
    using AdjointLoweringPassBase::AdjointLoweringPassBase;

    void runOnOperation() final
    {
        RewritePatternSet patterns(&getContext());
        patterns.add<AdjointSingleOpRewritePattern>(patterns.getContext(), 1);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
