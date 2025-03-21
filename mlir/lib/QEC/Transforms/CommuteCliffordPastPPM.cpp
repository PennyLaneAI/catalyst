// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "ppr_to_ppm"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/Transforms/Patterns.h"
#include "QEC/Utils/PauliStringWrapper.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::qec;

namespace {

bool verifyNextNonClifford(PPMeasurementOp op, Operation *nextOp)
{
    if (nextOp == nullptr)
        return true;

    if (nextOp == op)
        return false;

    if (nextOp->isBeforeInBlock(op))
        return true;

    for (auto userOp : nextOp->getUsers()) {
        if (!verifyNextNonClifford(op, userOp))
            return false;
    }

    return true;
}

bool verifyPrevNonClifford(PPMeasurementOp op, PPRotationOp prevOp)
{
    if (prevOp == nullptr)
        return false;

    if (prevOp.isNonClifford())
        return false;

    for (auto qubit : prevOp->getOperands()) {
        auto defOp = qubit.getDefiningOp();

        if (defOp == prevOp)
            continue;

        if (!verifyNextNonClifford(op, defOp))
            return false;
    }

    return true;
}

LogicalResult visitPPMeasurementOp(PPMeasurementOp op,
                                   std::function<LogicalResult(PPRotationOp)> callback)
{
    for (auto qubit : op->getOperands()) {
        if (qubit.getDefiningOp() == nullptr)
            continue;

        if (auto pprOp = llvm::dyn_cast<PPRotationOp>(qubit.getDefiningOp())) {
            if (verifyPrevNonClifford(op, pprOp)) {
                return callback(pprOp);
            }
        }
    }

    return failure();
}

void moveCliffordPastPPM(const PauliStringWrapper &lhsPauli, const PauliStringWrapper &rhsPauli,
                         PauliStringWrapper *result, PatternRewriter &rewriter)
{
    assert(lhsPauli.op != nullptr && "LHS Operation is not found");
    assert(rhsPauli.op != nullptr && "RHS Operation is not found");
    assert(llvm::isa<PPRotationOp>(lhsPauli.op) && "LHS Operation is not PPRotationOp");
    assert(llvm::isa<PPMeasurementOp>(rhsPauli.op) && "RHS Operation is not PPMeasurementOp");

    auto lhs = llvm::cast<PPRotationOp>(lhsPauli.op);
    auto rhs = llvm::cast<PPMeasurementOp>(rhsPauli.op);

    // Update Pauli words of RHS
    if (result != nullptr) {
        updatePauliWord(rhs, result->get_pauli_word(), rewriter);
        updatePauliWordSign(rhs, result->isNegative(), rewriter);
    }
    else {
        updatePauliWord(rhs, rhsPauli.get_pauli_word(), rewriter);
    }

    SmallVector<Value> newRHSOperands = replaceValueWithOperands(lhsPauli, rhsPauli);

    // Remove the Identity gate in the Pauli product
    SmallVector<StringRef> pauliProductArrayRef = removeIdentityPauli(rhs, newRHSOperands);
    mlir::ArrayAttr pauliProduct = rewriter.getStrArrayAttr(pauliProductArrayRef);

    // Get the type list from new RHS
    SmallVector<Type> newOutQubitTypes;
    for (auto qubit : newRHSOperands) {
        newOutQubitTypes.push_back(qubit.getType());
    }

    Type mresType = rhs.getMres().getType();

    auto newPPM =
        rewriter.create<PPMeasurementOp>(rhs->getLoc(), mresType, newOutQubitTypes, pauliProduct,
                                         rhs.getRotationSign(), newRHSOperands);
    rewriter.moveOpBefore(newPPM, rhs);

    // Update the use of value in newRHSOperands
    for (unsigned i = 0; i < newRHSOperands.size(); i++) {
        newRHSOperands[i].replaceAllUsesExcept(newPPM.getOutQubits()[i], newPPM);
    }

    rewriter.replaceAllUsesWith(rhs.getMres(), newPPM.getMres());
    rewriter.replaceAllUsesWith(rhs.getOutQubits(), rhs.getInQubits());

    rewriter.eraseOp(rhs);

    sortTopologically(lhs->getBlock());
}

bool shouldRemovePPR(PPRotationOp op)
{
    if (op->getUsers().empty())
        return true;

    SetVector<Operation *> slice;
    ForwardSliceOptions options;
    getForwardSlice(op, &slice);

    for (Operation *forwardOp : slice) {

        if (isa<PPMeasurementOp>(forwardOp))
            return false;

        if (!isa<catalyst::quantum::InsertOp>(forwardOp) &&
            !isa<catalyst::quantum::DeallocOp>(forwardOp))
            return false;
    }

    return true;
}

struct CommuteCliffordPastPPM : public OpRewritePattern<PPMeasurementOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PPMeasurementOp op, PatternRewriter &rewriter) const override
    {
        return visitPPMeasurementOp(op, [&](PPRotationOp pprOp) {
            PauliWordPair normOps = normalizePPROps(pprOp, op);

            if (normOps.first.commutes(normOps.second)) {
                moveCliffordPastPPM(normOps.first, normOps.second, nullptr, rewriter);
            }
            else {
                auto resultStr = normOps.first.computeCommutationRulesWith(normOps.second);
                moveCliffordPastPPM(normOps.first, normOps.second, &resultStr, rewriter);
            }
            return success();
        });
    }
};

struct RemoveDeadPPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        if (shouldRemovePPR(op)) {
            rewriter.replaceOp(op, op.getInQubits());
            return success();
        }
        return failure();
    }
};

} // namespace

namespace catalyst {
namespace qec {

void populateCommuteCliffordPastPPMPatterns(RewritePatternSet &patterns)
{
    patterns.add<CommuteCliffordPastPPM>(patterns.getContext(), 1);
    patterns.add<RemoveDeadPPR>(patterns.getContext(), 1);
}

} // namespace qec

} // namespace catalyst
