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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
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
        func::FuncOp parentFunc = op->getParentOfType<func::FuncOp>();
        if (!parentFunc) {
            return failure();
        }

        // If the gphase op itself is directly controlled or directly inside a CtrlOp,
        // cannot remove it (regardless of whether it's in subroutine or in main qnode func)
        if (op->getParentOfType<CtrlOp>() || !op.getInCtrlQubits().empty()) {
            return failure();
        }

        bool isInSubroutine = !parentFunc->hasAttr("quantum.node");

        // If in main qnode function, safe to erase
        if (!isInSubroutine) {
            rewriter.eraseOp(op);
            return success();
        }

        // Subroutine case: Traverse the call graph upwards to see if ANY ancestor is in a CtrlOp
        Operation *mod = op->getParentOfType<ModuleOp>();

        SmallVector<func::FuncOp> queue;
        llvm::DenseSet<Operation *> visited;

        queue.push_back(parentFunc);
        visited.insert(parentFunc);

        bool isControlledAnywhere = false;

        while (!queue.empty()) {
            func::FuncOp currentFunc = queue.pop_back_val();

            auto uses = SymbolTable::getSymbolUses(currentFunc, mod);
            if (!uses) {
                continue;
            }

            for (auto use : *uses) {
                Operation *user = use.getUser();
                if (auto callOp = dyn_cast<func::CallOp>(user)) {
                    // 1. Is this specific callsite inside a CtrlOp?
                    if (callOp->getParentOfType<CtrlOp>()) {
                        isControlledAnywhere = true;
                        break;
                    }
                    // 2. If not, add the caller function to the queue to keep traversing up
                    if (func::FuncOp callerFunc = callOp->getParentOfType<func::FuncOp>()) {
                        // DenseSet::insert().second is true if the element was newly inserted
                        if (visited.insert(callerFunc).second) {
                            queue.push_back(callerFunc);
                        }
                    }
                }
            }

            if (isControlledAnywhere) {
                break;
            }
        }

        if (isControlledAnywhere) {
            return failure();
        }

        rewriter.eraseOp(op);
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
