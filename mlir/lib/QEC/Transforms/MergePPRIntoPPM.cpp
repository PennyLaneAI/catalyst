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

#define DEBUG_TYPE "merge-ppr-ppm"

#include "llvm/Support/Casting.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/Transforms/Patterns.h"
#include "QEC/Utils/PauliStringWrapper.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::qec;

namespace {

/// recursively check if the next users of the NextOp are not PPMeasurementOp
bool verifyNextNonClifford(PPMeasurementOp op, Operation *nextOp)
{
    // Avoid segmentation fault (should not happen)
    if (nextOp == nullptr)
        return true;

    for (auto userOp : nextOp->getUsers()) {
        if (userOp == op)
            return false;

        if (!userOp->isBeforeInBlock(op))
            continue;

        if (!verifyNextNonClifford(op, userOp))
            return false;
    }

    return true;
}

/// The prevOp is valid when:
/// 1. prevOp is a non-Clifford operation. (We want to absorb the Clifford PPR into PPM)
/// 2. The users of the users of prevOp are not PPMeasurementOp.
///
/// For example, if PPRotationOp is Z⊗Z and prevOp is X⊗X:
///
/// ---| X |---------| Z |
///    |   |         |   |
/// ---| X |--| Y |--| Z |
///
/// Users of prevOp can be PPMeasurementOp,
/// but the users of Y (a user of prevOp) should not be PPMeasurementOp.
bool verifyPrevNonClifford(PPMeasurementOp op, PPRotationOp prevOp)
{
    // Avoid segmentation fault (should not happen)
    if (prevOp == nullptr)
        return false;

    if (prevOp.isNonClifford())
        return false;

    for (auto opUser : prevOp->getUsers()) {
        if (opUser == op)
            continue;

        if (!verifyNextNonClifford(op, opUser))
            return false;
    }

    return true;
}

LogicalResult visitValidCliffordPPR(PPMeasurementOp op,
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
    getForwardSlice(op, &slice);

    for (Operation *forwardOp : slice) {
        if (!isa<catalyst::quantum::InsertOp>(forwardOp) &&
            !isa<catalyst::quantum::DeallocOp>(forwardOp))
            return false;
    }

    return true;
}

struct MergePPRIntoPPM : public OpRewritePattern<PPMeasurementOp> {
    using OpRewritePattern::OpRewritePattern;

    size_t MAX_PAULI_SIZE;

    MergePPRIntoPPM(mlir::MLIRContext *context, size_t maxPauliSize, PatternBenefit benefit)
        : OpRewritePattern(context, benefit), MAX_PAULI_SIZE(maxPauliSize)
    {
    }

    LogicalResult matchAndRewrite(PPMeasurementOp PPMOp, PatternRewriter &rewriter) const override
    {
        return visitValidCliffordPPR(PPMOp, [&](PPRotationOp cliffordPPROp) {
            auto [normPPROp, normPPMOp] = normalizePPROps(cliffordPPROp, PPMOp);

            // Handle commuting case
            if (normPPROp.commutes(normPPMOp)) {
                moveCliffordPastPPM(normPPROp, normPPMOp, nullptr, rewriter);
                return success();
            }

            // Handle non-commuting case
            auto commutedResult = normPPROp.computeCommutationRulesWith(normPPMOp);

            // Skip if Pauli size is too large
            size_t pauliSize = commutedResult.get_pauli_word().size();
            if (exceedPauliSizeLimit(pauliSize, MAX_PAULI_SIZE)) {
                return failure();
            }

            moveCliffordPastPPM(normPPROp, normPPMOp, &commutedResult, rewriter);
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

void populateMergePPRIntoPPMPatterns(RewritePatternSet &patterns, unsigned int maxPauliSize)
{
    patterns.add<MergePPRIntoPPM>(patterns.getContext(), maxPauliSize, 1);
    patterns.add<RemoveDeadPPR>(patterns.getContext(), 1);
}

} // namespace qec

} // namespace catalyst
