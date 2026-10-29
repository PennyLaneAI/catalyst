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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/Transforms/Passes.h"
#include "QRef/Transforms/Patterns.h"

using namespace mlir;

namespace catalyst::qref {

// Distribute region controls onto the enclosed reference-semantics gates.
struct CtrlLoweringRewritePattern : public OpRewritePattern<CtrlOp> {
    CtrlLoweringRewritePattern(MLIRContext *context, bool &loweringFailed)
        : OpRewritePattern<CtrlOp>(context), loweringFailed(loweringFailed) {}

    bool &loweringFailed;

    LogicalResult matchAndRewrite(CtrlOp ctrl, PatternRewriter &rewriter) const override {
        if (loweringFailed) {
            return failure();
        }
        // Defer (not an error) if the region still contains a nested qref.adjoint region.
        // Distributing controls needs an op-level body, so the inner region must be reduced first.
        // Returning failure() drives the greedy fixpoint: adjoint-lowering reduces the inner region
        // to op-level gates, then the greedy driver re-tries this ctrl op and it lowers. This works
        // whether the two patterns run in separate alternating passes or together in the combined
        // modifiers-lowering pass. A pre-scan avoids a partial rewrite (creating ops, then bailing
        // out mid-region).
        if (ctrl.getRegion()
                .walk([&](Operation *op) {
                    if (isa<CallOpInterface>(op) && hasQuantumOperations(op)) {
                        op->emitError("cannot control a function call inside a qref.ctrl region");
                        loweringFailed = true;
                        return WalkResult::interrupt();
                    }
                    if (isa<AdjointOp>(op)) {
                        return WalkResult::interrupt();
                    }
                    if (isa<MeasureOp>(op)) {
                        return WalkResult::interrupt();
                    }
                    return WalkResult::advance();
                })
                .wasInterrupted()) {
            return failure();
        }

        ValueRange ctrlQubits = ctrl.getCtrlQubits();
        ValueRange ctrlValues = ctrl.getCtrlValues();

        ctrl.getRegion().walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
            if (auto gate = dyn_cast<QuantumGate>(op)) {
                rewriter.modifyOpInPlace(gate, [&] { gate.addControls(ctrlQubits, ctrlValues); });
                return WalkResult::advance();
            }
            if (auto inner = dyn_cast<CtrlOp>(op)) {
                rewriter.modifyOpInPlace(inner, [&] {
                    inner.getCtrlQubitsMutable().append(ctrlQubits);
                    inner.getCtrlValuesMutable().append(ctrlValues);
                });
                return WalkResult::skip();
            }
            return WalkResult::advance();
        });

        Block &block = ctrl.getRegion().front();
        rewriter.inlineBlockBefore(&block, ctrl);
        rewriter.eraseOp(ctrl);
        return success();
    }
};

} // namespace catalyst::qref

void catalyst::qref::populateCtrlLoweringPatterns(RewritePatternSet &patterns, bool &failed) {
    patterns.add<catalyst::qref::CtrlLoweringRewritePattern>(patterns.getContext(), failed);
}

namespace catalyst {
namespace qref {

#define GEN_PASS_DEF_CTRLLOWERINGPASS
#include "QRef/Transforms/Passes.h.inc"

struct CtrlLoweringPass : impl::CtrlLoweringPassBase<CtrlLoweringPass> {
    using CtrlLoweringPassBase::CtrlLoweringPassBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        bool loweringFailed = false;
        populateCtrlLoweringPatterns(patterns, loweringFailed);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) || loweringFailed) {
            return signalPassFailure();
        }
    }
};

} // namespace qref
} // namespace catalyst
