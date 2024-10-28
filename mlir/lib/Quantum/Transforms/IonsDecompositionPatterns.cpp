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

#define DEBUG_TYPE "ions-decomposition"

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

constexpr double PI = 3.14159265358979323846;

// Define map, name to function creating decomp

void oneQubitDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter, double phi, double theta, double lambda)
{
    TypeRange outQubitsTypes = op.getOutQubits().getTypes();
    TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();

    ValueRange inQubits = op.getInQubits();
    ValueRange inCtrlQubits = op.getInCtrlQubits();
    ValueRange inCtrlValues = op.getInCtrlValues();

    TypedAttr phiAttr = rewriter.getF64FloatAttr(phi);
    mlir::Value phiValue= rewriter.create<arith::ConstantOp>(op.getLoc(), phiAttr);
    TypedAttr thetaAttr = rewriter.getF64FloatAttr(theta);
    mlir::Value thetaValue = rewriter.create<arith::ConstantOp>(op.getLoc(), thetaAttr);
    TypedAttr lambdaAttr = rewriter.getF64FloatAttr(lambda);
    mlir::Value lambdaValue = rewriter.create<arith::ConstantOp>(op.getLoc(), lambdaAttr);

    auto rxPhi = rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes, outQubitsCtrlTypes,
                                                    phiValue, inQubits, "RX", nullptr,
                                                    inCtrlQubits, inCtrlValues);
    auto ryTheta= rewriter.create<CustomOp>(
        op.getLoc(), outQubitsTypes, outQubitsCtrlTypes, thetaValue, rxPhi.getOutQubits(),
        "RY", nullptr, rxPhi.getInCtrlQubits(), rxPhi.getInCtrlValues());
    auto rxLambda = rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes, outQubitsCtrlTypes,
                                               lambdaValue, ryTheta.getOutQubits(), "RX", nullptr,
                                               inCtrlQubits, inCtrlValues);
    op.replaceAllUsesWith(rxLambda);
}

void tDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    oneQubitDecomp(op, rewriter, -PI / 2, PI / 4, PI / 2);
}

std::map<std::string, std::function<void(catalyst::quantum::CustomOp, mlir::PatternRewriter &)>>
    funcMap = {{"T", &tDecomp}};

namespace {

struct IonsDecompositionRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {   
        auto it = funcMap.find(op.getGateName().str());
        if (it != funcMap.end()) {
            auto decompFunc = it->second;
            decompFunc(op, rewriter);
            return success();
        }
        else {
            return failure();
        }
    }
};
} // namespace

namespace catalyst {
namespace quantum {

void populateIonsDecompositionPatterns(RewritePatternSet &patterns)
{
    patterns.add<IonsDecompositionRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
