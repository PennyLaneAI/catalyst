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

#include "Catalyst/IR/CatalystDialect.h"
#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/Transforms/Passes.h"
#include "QRef/Transforms/Patterns.h"

#include "QuantumCache.hpp"

using namespace mlir;
using namespace catalyst::qref;

namespace {

/// Orchestrates the adjoint lowering of a single `qref.adjoint` op by running the forward pass
/// (recording the augmented circuit) followed by the reverse pass (emitting the adjoint circuit).
struct AdjointSingleOpRewritePattern : public OpRewritePattern<AdjointOp> {
    AdjointSingleOpRewritePattern(MLIRContext *context, bool &loweringFailed)
        : OpRewritePattern<AdjointOp>(context), loweringFailed(loweringFailed) {}

    bool &loweringFailed;

    LogicalResult matchAndRewrite(AdjointOp adjoint, PatternRewriter &rewriter) const override {
        if (loweringFailed) {
            return failure();
        }
        // Lower nested modifiers first. The greedy driver retries this region once its body
        // contains gates with op-level modifiers, including in the combined modifiers pass.
        if (adjoint.getRegion()
                .walk([](Operation *op) {
                    return isa<CtrlOp, AdjointOp>(op) ? WalkResult::interrupt()
                                                      : WalkResult::advance();
                })
                .wasInterrupted()) {
            return failure();
        }

        QuantumCache cache =
            QuantumCache::initialize(adjoint.getRegion(), rewriter, adjoint.getLoc());

        // Forward pass: copy the classical computations to the target insertion point and record
        // the values needed to replay the circuit in reverse.
        IRMapping oldToCloned;
        if (failed(generateAdjointForwardPass(adjoint.getRegion(), rewriter, oldToCloned, cache))) {
            loweringFailed = true;
            return failure();
        }

        // Reverse pass: emit the adjoint quantum operations and reversed control flow, using the
        // cached values.
        if (failed(generateAdjointReversePass(adjoint.getRegion(), rewriter, oldToCloned, cache))) {
            loweringFailed = true;
            return failure();
        }

        // Explicitly free the memory of the caches.
        cache.emitDealloc(rewriter, adjoint.getLoc());
        rewriter.eraseOp(adjoint);
        return success();
    }
};

} // namespace

void catalyst::qref::populateAdjointLoweringPatterns(RewritePatternSet &patterns, bool &failed) {
    patterns.add<AdjointSingleOpRewritePattern>(patterns.getContext(), failed);
}

namespace catalyst {
namespace qref {

#define GEN_PASS_DEF_ADJOINTLOWERINGPASS
#include "QRef/Transforms/Passes.h.inc"

struct AdjointLoweringPass : impl::AdjointLoweringPassBase<AdjointLoweringPass> {
    using AdjointLoweringPassBase::AdjointLoweringPassBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        bool loweringFailed = false;
        populateAdjointLoweringPatterns(patterns, loweringFailed);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) || loweringFailed) {
            return signalPassFailure();
        }
    }
};

} // namespace qref
} // namespace catalyst
