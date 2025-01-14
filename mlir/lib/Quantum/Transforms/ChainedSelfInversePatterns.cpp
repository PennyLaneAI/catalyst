// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "chained-self-inverse"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

#include "VerifyParentGateAnalysis.hpp"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

static const mlir::StringSet<> HermitianOps = {"Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT",
                                               "CY",       "CZ",     "SWAP",   "Toffoli"};

namespace {

struct ChainedNamedHermitianOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    /// We simplify consecutive Hermitian quantum gates by removing them.
    /// Hermitian gates are self-inverse and applying the same gate twice in succession
    /// cancels out the effect. This pattern rewrites such redundant operations by
    /// replacing the operation with its "grandparent" operation in the quantum circuit.
    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");

        StringRef OpGateName = op.getGateName();
        if (!HermitianOps.contains(OpGateName)) {
            return failure();
        }

        VerifyParentGateAndNameAnalysis<CustomOp> vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        // Replace uses
        ValueRange InQubits = op.getInQubits();
        auto parentOp = cast<CustomOp>(InQubits[0].getDefiningOp());

        // TODO: it would make more sense for getQubitOperands()
        // to return ValueRange, like the other getters
        std::vector<mlir::Value> originalQubits = parentOp.getQubitOperands();

        rewriter.replaceOp(op, originalQubits);
        return success();
    }
};

struct ChainedNamedHermitianStaticOpRewritePattern : public mlir::OpRewritePattern<StaticCustomOp> {
    using mlir::OpRewritePattern<StaticCustomOp>::OpRewritePattern;

    /// We simplify consecutive Hermitian quantum gates by removing them.
    /// Hermitian gates are self-inverse and applying the same gate twice in succession
    /// cancels out the effect. This pattern rewrites such redundant operations by
    /// replacing the operation with its "grandparent" operation in the quantum circuit.
    mlir::LogicalResult matchAndRewrite(StaticCustomOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");

        StringRef OpGateName = op.getGateName();
        if (!HermitianOps.contains(OpGateName)) {
            return failure();
        }

        VerifyParentGateAndNameAnalysis<StaticCustomOp> vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        // Replace uses
        ValueRange InQubits = op.getInQubits();
        auto parentOp = cast<StaticCustomOp>(InQubits[0].getDefiningOp());

        // TODO: it would make more sense for getQubitOperands()
        // to return ValueRange, like the other getters
        std::vector<mlir::Value> originalQubits = parentOp.getQubitOperands();

        rewriter.replaceOp(op, originalQubits);
        return success();
    }
};

template <typename OpType>
struct ChainedUUadjOpRewritePattern : public mlir::OpRewritePattern<OpType> {
    using mlir::OpRewritePattern<OpType>::OpRewritePattern;

    bool verifyParentGateParams(OpType op, OpType parentOp) const
    {
        // Verify that the parent gate has the same parameters
        auto opParams = op.getAllParams();
        auto parentOpParams = parentOp.getAllParams();

        if (opParams.size() != parentOpParams.size()) {
            return false;
        }

        for (auto [opParam, parentOpParam] : llvm::zip(opParams, parentOpParams)) {
            if (opParam != parentOpParam) {
                return false;
            }
        }

        return true;
    }

    bool verifyOneAdjoint(OpType op, OpType parentOp) const
    {
        // Verify that exactly one of the neighbouring pair is an adjoint
        bool opIsAdj = op->hasAttr("adjoint");
        bool parentIsAdj = parentOp->hasAttr("adjoint");
        return opIsAdj != parentIsAdj; // "XOR" to check just one true
    }

    /// Remove generic neighbouring gate pairs of the form
    /// --- gate --- gate{adjoint} ---
    /// Conditions:
    ///  1. Parent gate verification must pass. See VerifyParentGateAnalysis.hpp.
    ///  2. If there are parameters, both gate must have the same parameters.
    ///     [This pattern assumes the IR is already processed by CSE]
    mlir::LogicalResult matchAndRewrite(OpType op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");

        if (isa<CustomOp>(op)) {
            VerifyParentGateAndNameAnalysis<CustomOp> vpga(cast<CustomOp>(op));
            if (!vpga.getVerifierResult()) {
                return failure();
            }
        }
        else if (isa<StaticCustomOp>(op)) {
            VerifyParentGateAndNameAnalysis<StaticCustomOp> vpga(cast<StaticCustomOp>(op));
            if (!vpga.getVerifierResult()) {
                return failure();
            }
        }
        else {
            VerifyParentGateAnalysis<OpType> vpga(op);
            if (!vpga.getVerifierResult()) {
                return failure();
            }
        }

        ValueRange InQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<OpType>(InQubits[0].getDefiningOp());

        if (!verifyParentGateParams(op, parentOp)) {
            return failure();
        }
        if (!verifyOneAdjoint(op, parentOp)) {
            return failure();
        }

        // Replace uses
        ValueRange originalNonCtrlQubits = parentOp.getNonCtrlQubitOperands();
        ValueRange originalCtrlQubits = parentOp.getCtrlQubitOperands();

        ResultRange nonCtrlQubitResults = op.getNonCtrlQubitResults();
        nonCtrlQubitResults.replaceAllUsesWith(originalNonCtrlQubits);

        ResultRange ctrlQubitResults = op.getCtrlQubitResults();
        ctrlQubitResults.replaceAllUsesWith(originalCtrlQubits);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateSelfInversePatterns(RewritePatternSet &patterns)
{
    patterns.add<ChainedNamedHermitianStaticOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedNamedHermitianOpRewritePattern>(patterns.getContext(), 1);

    // TODO: better organize the quantum dialect
    // There is an interface `QuantumGate` for all the unitary gate operations,
    // but interfaces cannot be accepted by pattern matchers, since pattern
    // matchers require the target operations to have concrete names in the IR.
    patterns.add<ChainedUUadjOpRewritePattern<StaticCustomOp>>(patterns.getContext(), 1);
    patterns.add<ChainedUUadjOpRewritePattern<CustomOp>>(patterns.getContext(), 1);
    patterns.add<ChainedUUadjOpRewritePattern<QubitUnitaryOp>>(patterns.getContext(), 1);
    patterns.add<ChainedUUadjOpRewritePattern<MultiRZOp>>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
