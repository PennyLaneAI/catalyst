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

#define DEBUG_TYPE "commute-ppr"

#include "mlir/Analysis/TopologicalSortUtils.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/Transforms/Patterns.h"
#include "QEC/Utils/PauliStringWrapper.h"

using namespace mlir;
using namespace catalyst::qec;

namespace {

bool verifyPrevNonClifford(PPRotationOp op, Operation *prevOp)
{
    if (prevOp == nullptr)
        return true;

    if (prevOp == op)
        return false;

    if (prevOp->isBeforeInBlock(op))
        return true;

    for (auto operandValue : prevOp->getOperands()) {
        if (!verifyPrevNonClifford(op, operandValue.getDefiningOp()))
            return false;
    }
    return true;
}

bool verifyNextNonClifford(PPRotationOp op, PPRotationOp nextOp)
{
    if (!nextOp.isNonClifford())
        return false;

    if (nextOp == nullptr)
        return false;

    for (auto operandValue : nextOp.getOperands()) {
        auto defOp = operandValue.getDefiningOp();

        if (defOp == op)
            continue;

        if (!verifyPrevNonClifford(op, defOp))
            return false;
    }

    return true;
}

LogicalResult visitValidNonCliffordPPR(PPRotationOp op,
                                       std::function<LogicalResult(PPRotationOp)> callback)
{
    if (op.isNonClifford())
        return failure();

    for (auto userOp : op->getUsers()) {
        if (auto pprOp = llvm::dyn_cast<PPRotationOp>(userOp)) {
            if (verifyNextNonClifford(op, pprOp)) {
                return callback(pprOp);
            }
        }
    }

    return failure();
}

void moveCliffordPastNonClifford(const PauliStringWrapper &lhsPauli,
                                 const PauliStringWrapper &rhsPauli, PauliStringWrapper *result,
                                 PatternRewriter &rewriter)
{
    assert(lhsPauli.op != nullptr && "LHS Operation is not found");
    assert(rhsPauli.op != nullptr && "RHS Operation is not found");
    assert(llvm::isa<PPRotationOp>(lhsPauli.op) && "LHS Operation is not PPRotationOp");
    assert(llvm::isa<PPRotationOp>(rhsPauli.op) && "RHS Operation is not PPRotationOp");

    auto lhs = llvm::cast<PPRotationOp>(lhsPauli.op);
    auto rhs = llvm::cast<PPRotationOp>(rhsPauli.op);

    assert(lhs.isClifford() && "LHS Operation is not Clifford");
    assert(rhs.isNonClifford() && "RHS Operation is not non-Clifford");
    assert(lhs.getPauliProduct().size() == lhs.getOutQubits().size() &&
           "LHS Pauli product size mismatch before commutation.");
    assert(rhs.getPauliProduct().size() == rhs.getInQubits().size() &&
           "RHS Pauli product size mismatch before commutation.");

    // Update Pauli words of RHS
    if (result != nullptr) {
        updatePauliWord(rhs, result->get_pauli_word(), rewriter);
        updatePauliWordSign(rhs, result->isNegative(), rewriter);
    }
    else {
        updatePauliWord(rhs, rhsPauli.get_pauli_word(), rewriter);
        updatePauliWordSign(rhs, rhsPauli.isNegative(), rewriter);
    }

    // Fullfill Operands of RHS
    SmallVector<Value> newRHSOperands = replaceValueWithOperands(lhsPauli, rhsPauli);

    // Remove the Identity gate in the Pauli product
    SmallVector<StringRef> pauliProductArrayRef = removeIdentityPauli(rhs, newRHSOperands);
    mlir::ArrayAttr pauliProduct = rewriter.getStrArrayAttr(pauliProductArrayRef);

    // Get the type list from new RHS
    SmallVector<Type> newOutQubitsTypesList;
    for (auto qubit : newRHSOperands) {
        newOutQubitsTypesList.push_back(qubit.getType());
    }

    // Create the new PPR
    auto nonCliffordOp =
        rewriter.create<PPRotationOp>(rhs->getLoc(), newOutQubitsTypesList, pauliProduct,
                                      rhs.getRotationKindAttr(), newRHSOperands);
    rewriter.moveOpBefore(nonCliffordOp, rhs);

    // Update the use of value in newRHSOperands
    for (unsigned i = 0; i < newRHSOperands.size(); i++) {
        newRHSOperands[i].replaceAllUsesExcept(nonCliffordOp.getOutQubits()[i], nonCliffordOp);
    }

    rewriter.replaceOp(rhs, rhs.getInQubits());
}

struct CommutePPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    size_t MAX_PAULI_SIZE;

    CommutePPR(mlir::MLIRContext *context, size_t maxPauliSize, PatternBenefit benefit)
        : OpRewritePattern(context), MAX_PAULI_SIZE(maxPauliSize)
    {
    }

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        return visitValidNonCliffordPPR(op, [&](PPRotationOp nonCliffordPPR) {
            auto [normCliffordPPR, normNonCliffordPPR] = normalizePPROps(op, nonCliffordPPR);

            // Handle commuting case
            if (normCliffordPPR.commutes(normNonCliffordPPR)) {
                moveCliffordPastNonClifford(normCliffordPPR, normNonCliffordPPR, nullptr, rewriter);
                sortTopologically(op->getBlock());
                return success();
            }

            // Handle non-commuting case
            auto commutedResult = normCliffordPPR.computeCommutationRulesWith(normNonCliffordPPR);

            // Skip if Pauli size is too large
            size_t pauliSize = commutedResult.get_pauli_word().size();
            if (exceedPauliSizeLimit(pauliSize, MAX_PAULI_SIZE)) {
                return failure();
            }

            moveCliffordPastNonClifford(normCliffordPPR, normNonCliffordPPR, &commutedResult,
                                        rewriter);
            sortTopologically(op->getBlock());
            return success();
        });
    }
};
} // namespace

namespace catalyst {
namespace qec {

void populateCommutePPRPatterns(mlir::RewritePatternSet &patterns, unsigned int maxPauliSize)
{
    patterns.add<CommutePPR>(patterns.getContext(), maxPauliSize, 1);
}
} // namespace qec

} // namespace catalyst
