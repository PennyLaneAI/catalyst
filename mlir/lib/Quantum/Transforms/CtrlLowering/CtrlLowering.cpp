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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefOps.h"
#include "Quantum/Transforms/Passes.h"

using namespace mlir;

namespace catalyst::qref {

// match and rewrite a qref.ctrl op with reference semantics
// this needs to take in a qref.ctrl op and output qref.custum op
struct CtrlLoweringRewritePattern : public OpRewritePattern<CtrlOp> {
    using OpRewritePattern<CtrlOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CtrlOp ctrl, PatternRewriter &rewriter) const override {
        // Defer (not an error) if the region still contains a nested quantum.adjoint region.
        // Distributing controls needs an op-level body, so the inner region must be reduced first.
        // The pipeline runs (ctrl-lowering, adjoint-lowering) to a fixpoint: adjoint-lowering
        // reduces the inner region to op-level gates, then this ctrl op lowers on a later
        // iteration. A pre-scan avoids a partial rewrite (creating ops, then bailing out
        // mid-region).
        if (ctrl.getRegion()
                .walk([](Operation *op) {
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

        // The control qubits are threaded through every enclosed gate; the control values are
        // constant for the whole region.
        SmallVector<Value> currentCtrlQubits(ctrl.getCtrlQubits().begin(),
                                             ctrl.getCtrlQubits().end());
        ValueRange ctrlValues = ctrl.getCtrlValues();
        SmallVector<Operation *> opsToErase;

        // Can now preform the lowering after the pre-scan to check for errors in the region.
        ctrl.getRegion().walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
            if (auto gate = dyn_cast<QuantumGate>(op)) {
                rewriter.modifyOpInPlace(gate,
                                         [&] { gate.addControls(currentCtrlQubits, ctrlValues); });
                return WalkResult::advance();
            }
            if (auto inner = dyn_cast<CtrlOp>(op)) {
                rewriter.modifyOpInPlace(inner, [&] {
                    inner.getCtrlQubitsMutable().append(currentCtrlQubits);
                    inner.getCtrlValuesMutable().append(ctrlValues);
                });
                return WalkResult::skip();
            }
            return WalkResult::advance();
        });
        
        Block &block = ctrl.getRegion().front();
        rewriter.inlineBlockBefore(&block, ctrl);
        // Assemble the ctrl op results: out_ctrl_qubits followed by the target results.
        rewriter.eraseOp(ctrl);
        return success();
    }
};

} // namespace catalyst::qref

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_CTRLLOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct CtrlLoweringPass : impl::CtrlLoweringPassBase<CtrlLoweringPass> {
    using CtrlLoweringPassBase::CtrlLoweringPassBase;

    void runOnOperation() final {
        Operation *op = getOperation();
        // Convert to reference-semantics
        {
            OpPassManager ReferenceSemanticsPm(op->getName());
            ReferenceSemanticsPm.addPass(createReferenceSemanticsConversionPass());
            if (failed(runPipeline(ReferenceSemanticsPm, op))) {
                return signalPassFailure();
            }
        }

        RewritePatternSet patterns(&getContext());
        patterns.add<qref::CtrlLoweringRewritePattern>(patterns.getContext(), 1);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
