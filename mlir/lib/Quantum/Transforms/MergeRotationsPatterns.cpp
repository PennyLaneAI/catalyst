// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "merge-rotations"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst::quantum;

static const mlir::StringSet<> rotationsSet = {"RX",  "RY",  "RZ",  "PhaseShift",           "Rot",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift", "CRot"};

namespace {

struct MergeRotationsRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
        auto loc = op.getLoc();
        StringRef opGateName = op.getGateName();

        if (!rotationsSet.contains(opGateName))
            return failure();
        ValueRange inQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<CustomOp>(inQubits[0].getDefiningOp());

        if (!parentOp || parentOp.getGateName() != opGateName)
            return failure();

        ValueRange parentOutQubits = parentOp.getOutQubits();
        // Check if the input qubits to the current operation match the output qubits of the parent.
        for (const auto &[Idx, Qubit] : llvm::enumerate(inQubits)) {
            if (Qubit.getDefiningOp<CustomOp>() != parentOp || Qubit != parentOutQubits[Idx])
                return failure();
        }

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        // One param rot case
        auto parentParams = parentOp.getParams();
        auto params = op.getParams();
        std::vector<Value> sumParams;
        for (auto [param, parentParam] : llvm::zip(params, parentParams)) {
            Value sumParam = rewriter.create<arith::AddFOp>(loc, parentParam, param).getResult();
            sumParams.push_back(sumParam);
        };
        auto mergeOp = rewriter.create<CustomOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParams,
                                                 parentInQubits, opGateName, nullptr,
                                                 parentInCtrlQubits, parentInCtrlValues);

        op.replaceAllUsesWith(mergeOp);
        op.erase();
        parentOp.erase();

        return success();
    }
};

struct MergeMultiRZRewritePattern : public mlir::OpRewritePattern<MultiRZOp> {
    using mlir::OpRewritePattern<MultiRZOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MultiRZOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
        auto loc = op.getLoc();
        ValueRange InQubits = op.getInQubits();

        // Check parent op
        auto parentOp = dyn_cast_or_null<MultiRZOp>(InQubits[0].getDefiningOp());

        if (!parentOp)
            return failure();

        // Check the target qubit
        ValueRange parentOutQubits = parentOp.getOutQubits();
        for (const auto &[Idx, Qubit] : llvm::enumerate(InQubits)) {
            if (Qubit.getDefiningOp<CustomOp>() != parentOp || Qubit != parentOutQubits[Idx])
                return failure();
        }

        // Check the control qubits
        ValueRange inCtrlQubits = op.getInCtrlQubits();
        ValueRange parentOutCtrlQubits = parentOp.getOutCtrlQubits();
        for (const auto &[Idx, Qubit] : llvm::enumerate(InQubits)) {
            if (Qubit.getDefiningOp<CustomOp>() != parentOp || Qubit != parentOutCtrlQubits[Idx])
                return failure();
        }
        // Check the control values

        // ...

        // Sum the angles control values

        // Replace operation
        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();
        // op.replaceAllUsesWith(mergeOp);
        // op.erase();
        // parentOp.erase();

        return success();
    }
};
} // namespace

namespace catalyst {
namespace quantum {

void populateMergeRotationsPatterns(RewritePatternSet &patterns)
{
    patterns.add<MergeRotationsRewritePattern>(patterns.getContext(), 1);
    patterns.add<MergeMultiRZRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
