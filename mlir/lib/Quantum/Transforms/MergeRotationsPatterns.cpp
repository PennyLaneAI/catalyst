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
#include "VerifyParentGateAnalysis.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst::quantum;

static const mlir::StringSet<> rotationsSet = {"RX",  "RY",  "RZ",  "PhaseShift",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift"};

namespace {

template <typename OpType>
SmallVector<mlir::Value> convertOpParamsToValues(OpType &op, mlir::PatternRewriter &rewriter);
// Helper function for extracting static or non-static CustomOp parameters as mlir::Values.

template <>
SmallVector<mlir::Value> convertOpParamsToValues<CustomOp>(CustomOp &op,
                                                           mlir::PatternRewriter &rewriter)
{
    // In the case of a (non-static) CustomOp, the parameters are already mlir::Values, so we just
    // collect them into a vector.

    SmallVector<mlir::Value> values;
    auto params = op.getParams();
    for (auto param : params) {
        values.push_back(param);
    }
    return values;
}

template <>
SmallVector<mlir::Value> convertOpParamsToValues<StaticCustomOp>(StaticCustomOp &op,
                                                                 mlir::PatternRewriter &rewriter)
{
    // In the case of a StaticCustomOp, the parameters are doubles, so we need to introduce arith
    // ops to "convert" them into mlir::Values.

    SmallVector<mlir::Value> values;
    auto params = op.getStaticParams();
    for (auto param : params) {
        auto paramAttr = rewriter.getF64FloatAttr(param);
        values.emplace_back(rewriter.create<arith::ConstantOp>(op.getLoc(), paramAttr));
    }
    return values;
}

template <typename ParentOpType, typename OpType>
struct MergeRotationsRewritePattern : public mlir::OpRewritePattern<OpType> {
    // Merge rotation patterns where at least one operand is non-static.
    // The result is a non-static CustomOp, as at least one operand is not known at compile time.
    using mlir::OpRewritePattern<OpType>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(OpType op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
        auto loc = op.getLoc();
        StringRef opGateName = op.getGateName();
        if (!rotationsSet.contains(opGateName))
            return failure();
        ValueRange inQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<ParentOpType>(inQubits[0].getDefiningOp());

        VerifyHeterogeneousParentGateAndNameAnalysis<OpType, ParentOpType> vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        // extract parameters of the op and its parent,
        // promoting the parameters to mlir::Values if necessary
        auto parentParams = convertOpParamsToValues(parentOp, rewriter);
        auto params = convertOpParamsToValues(op, rewriter);
        SmallVector<mlir::Value> sumParams;
        for (auto [param, parentParam] : llvm::zip(params, parentParams)) {
            mlir::Value sumParam =
                rewriter.create<arith::AddFOp>(loc, parentParam, param).getResult();
            sumParams.push_back(sumParam);
        };
        auto mergeOp = rewriter.create<CustomOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParams,
                                                 parentInQubits, opGateName, nullptr,
                                                 parentInCtrlQubits, parentInCtrlValues);

        op.replaceAllUsesWith(mergeOp);

        return success();
    }
};

struct MergeRotationsStaticRewritePattern : public mlir::OpRewritePattern<StaticCustomOp> {
    // Merge rotation patterns where all operands are static
    using mlir::OpRewritePattern<StaticCustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(StaticCustomOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << op << "\n");
        auto loc = op.getLoc();
        StringRef opGateName = op.getGateName();
        if (!rotationsSet.contains(opGateName))
            return failure();
        ValueRange inQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<StaticCustomOp>(inQubits[0].getDefiningOp());

        VerifyParentGateAndNameAnalysis<StaticCustomOp> vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        auto parentParams = parentOp.getStaticParams();
        auto params = op.getStaticParams();
        SmallVector<double> sumParams;
        for (auto [param, parentParam] : llvm::zip(params, parentParams)) {
            sumParams.push_back(parentParam + param);
        };
        auto mergeOp = rewriter.create<StaticCustomOp>(
            loc, outQubitsTypes, outQubitsCtrlTypes, sumParams, parentInQubits, opGateName, nullptr,
            parentInCtrlQubits, parentInCtrlValues);

        op.replaceAllUsesWith(mergeOp);

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

        VerifyParentGateAnalysis<MultiRZOp> vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        ValueRange inQubits = op.getInQubits();
        auto parentOp = dyn_cast_or_null<MultiRZOp>(inQubits[0].getDefiningOp());
        if (!parentOp)
            return failure();

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        auto parentTheta = parentOp.getTheta();
        auto theta = op.getTheta();

        mlir::Value sumParam = rewriter.create<arith::AddFOp>(loc, parentTheta, theta).getResult();

        auto mergeOp = rewriter.create<MultiRZOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParam,
                                                  parentInQubits, nullptr, parentInCtrlQubits,
                                                  parentInCtrlValues);
        op.replaceAllUsesWith(mergeOp);
        return success();
    }
};
} // namespace

namespace catalyst {
namespace quantum {

void populateMergeRotationsPatterns(RewritePatternSet &patterns)
{
    patterns.add<MergeRotationsStaticRewritePattern>(patterns.getContext(), 1);
    patterns.add<MergeRotationsRewritePattern<CustomOp, StaticCustomOp>>(patterns.getContext(), 1);
    patterns.add<MergeRotationsRewritePattern<StaticCustomOp, CustomOp>>(patterns.getContext(), 1);
    patterns.add<MergeRotationsRewritePattern<CustomOp, CustomOp>>(patterns.getContext(), 1);
    patterns.add<MergeMultiRZRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
