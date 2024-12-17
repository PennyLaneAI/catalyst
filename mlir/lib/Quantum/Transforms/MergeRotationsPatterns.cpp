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

        VerifyParentGateAndNameAnalysis vpga(op);
        if (!vpga.getVerifierResult()) {
            return failure();
        }

        TypeRange outQubitsTypes = op.getOutQubits().getTypes();
        TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
        ValueRange parentInQubits = parentOp.getInQubits();
        ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
        ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

        ArrayAttr parentStaticParams = parentOp.getStaticParamsAttr();
        ArrayAttr currentStaticParams = op.getStaticParamsAttr();
        auto parentDynParams = parentOp.getParams();
        auto currentDynParams = op.getParams();

        bool parentHasStatic = parentStaticParams && !parentStaticParams.empty();
        bool currentHasStatic = currentStaticParams && !currentStaticParams.empty();

        SmallVector<mlir::Value> sumParams;
        ArrayAttr SumAttrs = nullptr;

        if (parentHasStatic && currentHasStatic) {
            if (parentStaticParams.size() != currentStaticParams.size()) {
                return failure();
            }
            SmallVector<Attribute, 4> SumAttr;
            for (auto [pAttr, cAttr] : llvm::zip(parentStaticParams, currentStaticParams)) {
                auto pFloat = cast<FloatAttr>(pAttr).getValueAsDouble();
                auto cFloat = cast<FloatAttr>(cAttr).getValueAsDouble();
                SumAttr.push_back(rewriter.getF64FloatAttr(pFloat + cFloat));
            }
            SumAttrs = rewriter.getArrayAttr(SumAttr);
        }
        else if (!parentHasStatic && !currentHasStatic) {
            if (currentDynParams.size() != parentDynParams.size()) {
                return failure();
            }
            for (auto [param, parentParam] : llvm::zip(currentDynParams, parentDynParams)) {
                mlir::Value sumParam = rewriter.create<arith::AddFOp>(loc, parentParam, param);
                sumParams.push_back(sumParam);
            }
        }
        else {
            return failure();
        }

        auto mergeOp = rewriter.create<CustomOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParams,
                                                 parentInQubits, opGateName, nullptr,
                                                 parentInCtrlQubits, parentInCtrlValues, SumAttrs);
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

        ArrayAttr parentStaticParams = parentOp.getStaticParamsAttr();
        ArrayAttr currentStaticParams = op.getStaticParamsAttr();

        bool parentHasStatic = parentStaticParams && parentStaticParams.size() == 1;
        bool currentHasStatic = currentStaticParams && currentStaticParams.size() == 1;

        auto parentTheta = parentOp.getTheta();
        auto theta = op.getTheta();

        mlir::Value sumParam = mlir::Value();
        ArrayAttr SumAttr = nullptr;

        if (parentHasStatic && currentHasStatic) {
            double pVal = cast<FloatAttr>(parentStaticParams[0]).getValueAsDouble();
            double cVal = cast<FloatAttr>(currentStaticParams[0]).getValueAsDouble();
            SumAttr = rewriter.getArrayAttr({rewriter.getF64FloatAttr(pVal + cVal)});
        }
        else if (!parentHasStatic && !currentHasStatic) {
            sumParam = rewriter.create<arith::AddFOp>(loc, parentTheta, theta);
        }
        else {
            return failure();
        }

        auto mergeOp = rewriter.create<MultiRZOp>(loc, outQubitsTypes, outQubitsCtrlTypes, sumParam,
                                                  parentInQubits, nullptr, parentInCtrlQubits,
                                                  parentInCtrlValues, SumAttr);
        op.replaceAllUsesWith(mergeOp);
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
