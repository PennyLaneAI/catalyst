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

#define DEBUG_TYPE "remove-global-phases"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace {

/// delete phase ops without control wires
struct RemoveGlobalPhasesRewritePattern : public OpRewritePattern<GlobalPhaseOp> {
    using OpRewritePattern<GlobalPhaseOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(GlobalPhaseOp op, PatternRewriter &rewriter) const override {

        // Cannot remove gphase ops in subroutines, as they might be called in a control region.
        if (auto parentFunc = op->getParentOfType<func::FuncOp>();
            !parentFunc || !parentFunc->hasAttr("quantum.node")) {
            return failure();
        }

        // Find out if there are any control regions in the parent chain,
        // Or if there are control qubits associated with this operation.
        // If so, it must be ignored
        if (op->getParentOfType<CtrlOp>() || !op.getInCtrlQubits().empty()) {
            return failure();
        }

        // Erase global phase op
        rewriter.eraseOp(op);

        // Successful application of pattern
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_REMOVEGLOBALPHASESPASS
#define GEN_PASS_DEF_REMOVEGLOBALPHASESPASS
#include "Quantum/Transforms/Passes.h.inc"

struct RemoveGlobalPhasesPass : public impl::RemoveGlobalPhasesPassBase<RemoveGlobalPhasesPass> {
    using impl::RemoveGlobalPhasesPassBase<RemoveGlobalPhasesPass>::RemoveGlobalPhasesPassBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        patterns.add<RemoveGlobalPhasesRewritePattern>(patterns.getContext(), 1);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
