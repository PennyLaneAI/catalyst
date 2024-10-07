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

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
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

        ValueRange InQubits = op.getInQubits();
        auto ParentOp = dyn_cast_or_null<CustomOp>(InQubits[0].getDefiningOp());
        if (!ParentOp || ParentOp.getGateName() != OpGateName) {
            return failure();
        }

        ValueRange ParentOutQubits = ParentOp.getOutQubits();
        // Check if the input qubits to the current operation match the output qubits of the parent.
        for (const auto &[Idx, Qubit] : llvm::enumerate(InQubits)) {
            if (Qubit.getDefiningOp<CustomOp>() != ParentOp || Qubit != ParentOutQubits[Idx]) {
                return failure();
            }
        }
        ValueRange simplifiedVal = ParentOp.getInQubits();
        rewriter.replaceOp(op, simplifiedVal);
        return success();
    }
};

template <typename OpType>
struct ChainedUUadjOpRewritePattern : public mlir::OpRewritePattern<OpType> {
    using mlir::OpRewritePattern<OpType>::OpRewritePattern;

    bool verifyParentGateType(OpType op, OpType parentOp) const
    {
        // Verify that the parent gate is of the same type,
        // and parent's results and current gate's inputs are in the same order
        // If OpType is quantum.custom, also verify that parent gate has the
        // same gate name.

        if (!parentOp || !isa<OpType>(parentOp)) {
            return false;
        }

        if (isa<CustomOp>(op)) {
            StringRef opGateName = cast<CustomOp>(op).getGateName();
            StringRef parentGateName = cast<CustomOp>(parentOp).getGateName();
            if (opGateName != parentGateName) {
                return false;
            }
        }

        ValueRange InQubits = op.getInQubits();
        ValueRange ParentOutQubits = parentOp.getOutQubits();
        for (const auto &[Idx, Qubit] : llvm::enumerate(InQubits)) {
            if (Qubit.getDefiningOp<OpType>() != parentOp || Qubit != ParentOutQubits[Idx]) {
                return false;
            }
        }

        return true;
    }

    bool verifyParentGateParams(OpType op, OpType parentOp) const
    {
        // Verify that the parent gate has the same parameters

        ValueRange opParams = op.getAllParams();
        ValueRange parentOpParams = parentOp.getAllParams();

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
    ///  1. Both gates must be of the same type, i.e. a quantum.custom can
    ///     only be cancelled with a quantum.custom, not a quantum.unitary
    ///  2. The results of the parent gate must map one-to-one, in order,
    ///     to the operands of the second gate
    ///  3. If there are parameters, both gate must have the same parameters.
    ///     [This pattern assumes the IR is already processed by CSE]
    ///  4. If the gates are controlled, both gates' control wires and values
    ///     must be the same. The control wires must be in the same order
    mlir::LogicalResult matchAndRewrite(OpType op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");

        // llvm::errs() << "visiting " << op << "\n";

        ValueRange InQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<OpType>(InQubits[0].getDefiningOp());

        if (!verifyParentGateType(op, parentOp)) {
            return failure();
        }

        if (!verifyParentGateParams(op, parentOp)) {
            return failure();
        }

        if (!verifyOneAdjoint(op, parentOp)) {
            return failure();
        }

        // llvm::errs() << "matched!\n";
        ValueRange simplifiedVal = parentOp.getInQubits();
        rewriter.replaceOp(op, simplifiedVal);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateSelfInversePatterns(RewritePatternSet &patterns)
{
    patterns.add<ChainedNamedHermitianOpRewritePattern>(patterns.getContext(), 1);
    patterns.add<ChainedUUadjOpRewritePattern<CustomOp>>(patterns.getContext(), 1);
    patterns.add<ChainedUUadjOpRewritePattern<QubitUnitaryOp>>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
