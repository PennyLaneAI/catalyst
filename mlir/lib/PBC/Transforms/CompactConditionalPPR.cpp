// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "PBC/IR/PBCOps.h"
#include "PBC/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::pbc;

namespace {

template <typename PPROpType> LogicalResult matchSimpleIfPPR(scf::IfOp ifOp, PPROpType &pprOp)
{
    if (!ifOp.getThenRegion().hasOneBlock() || !ifOp.getElseRegion().hasOneBlock()) {
        return failure();
    }

    Block &thenBlock = ifOp.getThenRegion().front();
    Block &elseBlock = ifOp.getElseRegion().front();

    if (!llvm::hasNItems(thenBlock.without_terminator(), 1) ||
        !elseBlock.without_terminator().empty()) {
        return failure();
    }

    Operation *singleThenOp = &thenBlock.front();
    pprOp = dyn_cast<PPROpType>(singleThenOp);
    if (!pprOp) {
        return failure();
    }

    auto thenYield = dyn_cast<scf::YieldOp>(thenBlock.getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(elseBlock.getTerminator());
    if (!thenYield || !elseYield) {
        return failure();
    }

    if (thenYield.getNumOperands() != pprOp->getNumResults() ||
        elseYield.getNumOperands() != pprOp.getInQubits().size()) {
        return failure();
    }

    if (!llvm::equal(thenYield.getOperands(), pprOp.getOutQubits())) {
        return failure();
    }

    if (!llvm::equal(elseYield.getOperands(), pprOp.getInQubits())) {
        return failure();
    }

    return success();
}

Value combineConds(PatternRewriter &rewriter, Location loc, Value ifCond, Value pprCond)
{
    if (!pprCond) {
        return ifCond;
    }
    return arith::AndIOp::create(rewriter, loc, ifCond, pprCond).getResult();
}

struct CompactConditionalPPR : public OpRewritePattern<scf::IfOp> {
    using OpRewritePattern<scf::IfOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::IfOp ifOp, PatternRewriter &rewriter) const override
    {
        Location loc = ifOp.getLoc();

        PPRotationOp pprOp;
        if (succeeded(matchSimpleIfPPR<PPRotationOp>(ifOp, pprOp))) {
            Value combinedCond =
                combineConds(rewriter, loc, ifOp.getCondition(), pprOp.getCondition());
            auto newOp = PPRotationOp::create(rewriter, loc, pprOp.getPauliProduct(),
                                              pprOp.getRotationKindAttr(), pprOp.getInQubits(),
                                              combinedCond);
            rewriter.replaceOp(ifOp, newOp.getResults());
            return success();
        }

        PPRotationArbitraryOp pprArbitraryOp;
        if (succeeded(matchSimpleIfPPR<PPRotationArbitraryOp>(ifOp, pprArbitraryOp))) {
            Value combinedCond =
                combineConds(rewriter, loc, ifOp.getCondition(), pprArbitraryOp.getCondition());
            auto newOp = PPRotationArbitraryOp::create(
                rewriter, loc, pprArbitraryOp.getPauliProduct(), pprArbitraryOp.getArbitraryAngle(),
                pprArbitraryOp.getInQubits(), combinedCond);
            rewriter.replaceOp(ifOp, newOp.getResults());
            return success();
        }

        return failure();
    }
};

} // namespace

namespace catalyst {
namespace pbc {

void populateCompactConditionalPPRPatterns(RewritePatternSet &patterns)
{
    patterns.add<CompactConditionalPPR>(patterns.getContext());
}

} // namespace pbc
} // namespace catalyst
