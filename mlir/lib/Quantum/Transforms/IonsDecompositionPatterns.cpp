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

void tDecomp(catalyst::quantum::CustomOp op, mlir::PatternRewriter &rewriter)
{
    TypeRange outQubitsTypes = op.getOutQubits().getTypes();
    TypeRange outQubitsCtrlTypes = op.getOutCtrlQubits().getTypes();
    ValueRange inQubits = op.getInQubits();

    auto parentOp = dyn_cast_or_null<CustomOp>(inQubits[0].getDefiningOp());
    ValueRange parentInQubits = parentOp.getInQubits();
    ValueRange parentInCtrlQubits = parentOp.getInCtrlQubits();
    ValueRange parentInCtrlValues = parentOp.getInCtrlValues();

    TypedAttr minusPiOver2Attr = rewriter.getF64FloatAttr(-PI / 2);
    mlir::Value minusPiOver2 = rewriter.create<arith::ConstantOp>(op.getLoc(), minusPiOver2Attr);
    TypedAttr piOver2Attr = rewriter.getF64FloatAttr(PI / 2);
    mlir::Value piOver2 = rewriter.create<arith::ConstantOp>(op.getLoc(), piOver2Attr);
    TypedAttr piOver4Attr = rewriter.getF64FloatAttr(PI / 2);
    mlir::Value piOver4 = rewriter.create<arith::ConstantOp>(op.getLoc(), piOver4Attr);

    auto rxMinusPiOver2 = rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes, outQubitsCtrlTypes,
                                                    minusPiOver2, parentInQubits, "RX", nullptr,
                                                    parentInCtrlQubits, parentInCtrlValues);
    auto rzPiOver4 = rewriter.create<CustomOp>(
        op.getLoc(), outQubitsTypes, outQubitsCtrlTypes, piOver4, rxMinusPiOver2.getOutQubits(),
        "RZ", nullptr, rxMinusPiOver2.getInCtrlQubits(), rxMinusPiOver2.getInCtrlValues());
    auto rxPiOver2 = rewriter.create<CustomOp>(op.getLoc(), outQubitsTypes, outQubitsCtrlTypes,
                                               piOver2, rzPiOver4.getOutQubits(), "RX", nullptr,
                                               parentInCtrlQubits, parentInCtrlValues);
    op.replaceAllUsesWith(rxPiOver2);
}

std::map<StringRef, std::function<void(catalyst::quantum::CustomOp, mlir::PatternRewriter &)>>
    funcMap = {{"T", &tDecomp}};

namespace {

struct IonsDecompositionRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        auto it = funcMap.find(op.getGateName());
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
