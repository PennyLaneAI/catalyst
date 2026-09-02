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

#define DEBUG_TYPE "resolve-basis-state-operator"

#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

struct ResolveBasisStateOperatorPattern : public OpRewritePattern<OperatorOp> {
    using OpRewritePattern<OperatorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(OperatorOp op, PatternRewriter &rewriter) const override {
        StringRef name = op.getOpName();
        bool isBasisState = (name == "BasisState");
        bool isStatePrep = (name == "StatePrep");
        if (!isBasisState && !isStatePrep) {
            return failure();
        }

        assert(op.getCtrlQubitOperands().empty() && "Cannot control a SetState/SetBasisState op");
        assert(!op.getAdjointFlag() && "Cannot adjoint a SetState/SetBasisState op");
        assert(op.getParams().size() == 1 &&
               "OperatorOp of BasisState/StatePrep expected the state as the only param");

        Value state = op.getParams()[0];
        auto stateType = dyn_cast<RankedTensorType>(state.getType());

        if (isBasisState) {
            assert(stateType && stateType.getRank() == 1 &&
                   stateType.getElementType().isInteger(1) &&
                   "OperatorOp of BasisState must take in a rank-1 tensor of bools as the state");

            rewriter.replaceOpWithNewOp<SetBasisStateOp>(op, TypeRange(op.getNonCtrlQubitResults()),
                                                         state, op.getNonCtrlQubitOperands());
            return success();
        }

        if (isStatePrep) {
            auto stateElementType = dyn_cast<ComplexType>(stateType.getElementType());
            assert(stateType && stateType.getRank() == 1 && stateElementType &&
                   stateElementType.getElementType().isF64() &&
                   "OperatorOp of StatePrep must take in a rank-1 tensor of complex<f64> as the "
                   "state");

            rewriter.replaceOpWithNewOp<SetStateOp>(op, TypeRange(op.getNonCtrlQubitResults()),
                                                    state, op.getNonCtrlQubitOperands());
            return success();
        }

        return failure();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_RESOLVEBASISSTATEOPERATORPASS
#define GEN_PASS_DECL_RESOLVEBASISSTATEOPERATORPASS
#include "Quantum/Transforms/Passes.h.inc"

void populateResolveBasisStateOperatorPatterns(RewritePatternSet &patterns) {
    patterns.add<ResolveBasisStateOperatorPattern>(patterns.getContext());
}

struct ResolveBasisStateOperatorPass
    : impl::ResolveBasisStateOperatorPassBase<ResolveBasisStateOperatorPass> {
    using ResolveBasisStateOperatorPassBase::ResolveBasisStateOperatorPassBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        populateResolveBasisStateOperatorPatterns(patterns);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
