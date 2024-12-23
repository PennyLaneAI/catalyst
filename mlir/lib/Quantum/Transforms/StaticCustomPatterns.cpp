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

#define DEBUG_TYPE "static-custom"

#include "Quantum/IR/QuantumOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst::quantum;

namespace {

struct LowerStaticCustomOp : public OpConversionPattern<StaticCustomOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(StaticCustomOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override

    {
        LLVM_DEBUG(dbgs() << "Lowering the following static custom operation:\n" << op << "\n");
        SmallVector<Value, 4> paramValues;
        auto staticParams = op.getStaticParams();
        for (auto param : staticParams) {
            auto constant = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getF64Type(),
                                                               rewriter.getF64FloatAttr(param));
            paramValues.push_back(constant);
        }
        if (op.getGateName() == "MultiRZ") {
            if (paramValues.size() != 1) {
                op.emitError() << "MultiRZ gate expects exactly one parameter";
                return failure();
            }
            rewriter.replaceOpWithNewOp<MultiRZOp>(
                op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), paramValues[0],
                op.getInQubits(), op.getAdjointAttr(), op.getInCtrlQubits(), op.getInCtrlValues());
            return success();
        }
        if (op.getGateName() == "GlobalPhase") {
            if (paramValues.size() != 1) {
                op.emitError() << "GlobalPhase gate expects exactly one parameter";
                return failure();
            }
            rewriter.replaceOpWithNewOp<GlobalPhaseOp>(op, op.getOutCtrlQubits().getTypes(),
                                                       paramValues[0], op.getAdjointAttr(),
                                                       op.getInCtrlQubits(), op.getInCtrlValues());
            return success();
        }
        rewriter.replaceOpWithNewOp<CustomOp>(op, op.getGateName(), op.getInQubits(),
                                              op.getInCtrlQubits(), op.getInCtrlValues(),
                                              paramValues, op.getAdjointFlag());
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateStaticCustomPatterns(RewritePatternSet &patterns)
{
    patterns.add<LowerStaticCustomOp>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
