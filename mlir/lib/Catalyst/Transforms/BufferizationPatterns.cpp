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
        // Add bufferized arguments
        SmallVector<Value> bufferArgs;
        ValueRange operands = adaptor.getOperands();
        for (Value operand : operands) {
            bufferArgs.push_back(operand);
        }

        // Add bufferized return values to the arguments
        ValueRange results = op.getResults();
        for (Value result : results) {
            Type resultType = result.getType();
            RankedTensorType tensorType = resultType.dyn_cast<RankedTensorType>();
            if (!tensorType) {
                return failure();
            }
            auto options = bufferization::BufferizationOptions();
            FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
                rewriter, op->getLoc(), result, options, false);
            MemRefType memrefType =
                MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            auto newBuffer =
                rewriter.create<bufferization::ToMemrefOp>(op->getLoc(), memrefType, *tensorAlloc);
            bufferArgs.push_back(newBuffer);
        }
        // Add the initial number of arguments
        int32_t numArguments = static_cast<int32_t>(op.getNumOperands());
        DenseI32ArrayAttr numArgumentsDenseAttr = rewriter.getDenseI32ArrayAttr({numArguments});

        // Create an updated custom call operation
        rewriter.create<CustomCallOp>(op->getLoc(), TypeRange{}, bufferArgs, op.getCallTargetName(),
                                      numArgumentsDenseAttr);
        size_t startIndex = bufferArgs.size() - op.getNumResults();
        SmallVector<Value> bufferResults(bufferArgs.begin() + startIndex, bufferArgs.end());
        rewriter.replaceOp(op, bufferResults);
        return success();
    }
};

struct BufferizeInactiveCallbackOp : public OpConversionPattern<InactiveCallbackOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(InactiveCallbackOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Add bufferized arguments
        SmallVector<Value> bufferArgs(adaptor.getOperands().begin(), adaptor.getOperands().end());

        // Add bufferized return values to the arguments
        auto results = op.getResults();

        for (Value result : results) {
            Type resultType = result.getType();
            RankedTensorType tensorType = resultType.dyn_cast<RankedTensorType>();
            if (!tensorType) {
                return failure();
            }
            auto options = bufferization::BufferizationOptions();
            FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
                rewriter, op->getLoc(), result, options, false);
            MemRefType memrefType =
                MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            auto newBuffer =
                rewriter.create<bufferization::ToMemrefOp>(op->getLoc(), memrefType, *tensorAlloc);
            bufferArgs.push_back(newBuffer);
        }

        rewriter.create<InactiveCallbackOp>(op.getLoc(), TypeRange{}, bufferArgs,
                                            adaptor.getIdentifier(), op.getOperands().size());
        size_t startIndex = bufferArgs.size() - op.getNumResults();
        SmallVector<Value> bufferResults(bufferArgs.begin() + startIndex, bufferArgs.end());
        rewriter.replaceOp(op, bufferResults);
        return success();
    }
};

struct BufferizeActiveCallbackOp : public OpConversionPattern<ActiveCallbackOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ActiveCallbackOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        // Add bufferized arguments
        SmallVector<Value> bufferArgs(adaptor.getOperands().begin(), adaptor.getOperands().end());

        // Add bufferized return values to the arguments
        auto results = op.getResults();

        for (Value result : results) {
            Type resultType = result.getType();
            RankedTensorType tensorType = resultType.dyn_cast<RankedTensorType>();
            if (!tensorType) {
                return failure();
            }
            auto options = bufferization::BufferizationOptions();
            FailureOr<Value> tensorAlloc = bufferization::allocateTensorForShapedValue(
                rewriter, op->getLoc(), result, options, false);
            MemRefType memrefType =
                MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            auto newBuffer =
                rewriter.create<bufferization::ToMemrefOp>(op->getLoc(), memrefType, *tensorAlloc);
            bufferArgs.push_back(newBuffer);
        }

        auto originalArgsSize = rewriter.getI64IntegerAttr(op.getOperands().size());
        rewriter.create<ActiveCallbackOp>(op.getLoc(), TypeRange{}, bufferArgs,
                                          adaptor.getIdentifier(), originalArgsSize,
                                          adaptor.getSpecializedAttr(), adaptor.getFwdAttr(), adaptor.getFwdSigAttr(), adaptor.getBwdAttr(), adaptor.getBwdSigAttr());
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
    patterns.add<BufferizeInactiveCallbackOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeActiveCallbackOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizePrintOp>(typeConverter, patterns.getContext());
}

} // namespace catalyst
