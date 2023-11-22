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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

namespace {

struct BufferizePrintOp : public OpConversionPattern<PrintOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        if (op.getVal()) {
            rewriter.replaceOpWithNewOp<PrintOp>(op, adaptor.getVal(), adaptor.getConstValAttr(),
                                                 adaptor.getPrintDescriptorAttr());
        }
        return success();
    }
};

struct BufferizeCustomCallOp : public OpConversionPattern<CustomCallOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CustomCallOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Check arguments
        SmallVector<Value> bufferArgs;
        auto operands = op.getOperands();
        for (auto operand : operands) {
            auto &newBuffer = bufferArgs.emplace_back();
            auto operandType = operand.getType();
            auto tensorOperandType = operandType.dyn_cast<RankedTensorType>();
            auto memrefType =
                MemRefType::get(tensorOperandType.getShape(), tensorOperandType.getElementType());
            newBuffer =
                rewriter.create<bufferization::ToMemrefOp>(op->getLoc(), memrefType, operand);
        }

        // Allocate returns.
        auto results = op.getResults();
        for (Value result : results) {
            auto &newBuffer = bufferArgs.emplace_back();
            auto resultType = result.getType();
            auto tensorType = resultType.dyn_cast<RankedTensorType>();
            if (!tensorType)
                return failure();
            auto options = bufferization::BufferizationOptions();
            FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
                rewriter, op->getLoc(), result, false, options, false);
            auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            newBuffer =
                rewriter.create<bufferization::ToMemrefOp>(op->getLoc(), memrefType, *tensorAlloc);
        }
        auto numArguments = static_cast<int32_t>(op.getNumOperands());
        auto numArgumentsDenseAttr = rewriter.getDenseI32ArrayAttr({numArguments});
        rewriter.create<CustomCallOp>(op->getLoc(), TypeRange{}, bufferArgs,
                                                   op.getCallTargetName(), numArgumentsDenseAttr);
        size_t startIndex = bufferArgs.size() - op.getNumResults();
        SmallVector<Value> bufferResults(bufferArgs.begin() + startIndex, bufferArgs.end());
        rewriter.replaceOp(op, bufferResults);
        return success();
    }
};

} // namespace

namespace catalyst {

void populateBufferizationPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<BufferizeCustomCallOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizePrintOp>(typeConverter, patterns.getContext());
}

} // namespace catalyst
