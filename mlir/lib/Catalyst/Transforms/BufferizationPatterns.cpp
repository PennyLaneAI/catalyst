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
            RankedTensorType tensorType = dyn_cast<RankedTensorType>(resultType);
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

struct BufferizeCallbackOp : public OpConversionPattern<CallbackOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult match(CallbackOp op) const override
    {
        // Only match here if we have all memref arguments and return values.
        if (llvm::any_of(op.getArgumentTypes(),
                         [](Type argType) { return !isa<MemRefType>(argType); })) {
            return failure();
        }
        if (llvm::any_of(op.getResultTypes(),
                         [](Type argType) { return !isa<MemRefType>(argType); })) {
            return failure();
        }

        // Only match if we have result types.
        return op.getResultTypes().empty() ? failure() : success();
    }

    void rewrite(CallbackOp op, OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter) const override
    {
        auto argTys = op.getArgumentTypes();
        auto retTys = op.getResultTypes();
        SmallVector<Type> emptyRets;
        SmallVector<Type> args(argTys.begin(), argTys.end());
        args.insert(args.end(), retTys.begin(), retTys.end());
        auto callbackTy = rewriter.getFunctionType(args, emptyRets);
        rewriter.modifyOpInPlace(op, [&] { op.setFunctionType(callbackTy); });
    }
};

struct BufferizeCallbackCallOp : public OpConversionPattern<CallbackCallOp> {
    using OpConversionPattern<CallbackCallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CallbackCallOp callOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Type> convertedResults;
        if (failed(typeConverter->convertTypes(callOp.getResultTypes(), convertedResults)))
            return failure();

        if (callOp->getNumResults() != convertedResults.size())
            return failure();

        auto operands = adaptor.getOperands();
        SmallVector<Value> newInputs(operands.begin(), operands.end());
        auto results = callOp.getResults();

        auto loc = callOp->getLoc();
        auto options = bufferization::BufferizationOptions();
        SmallVector<Value> outmemrefs;
        for (auto result : results) {
            FailureOr<Value> tensorAlloc =
                bufferization::allocateTensorForShapedValue(rewriter, loc, result, options, false);
            if (failed(tensorAlloc))
                return failure();

            auto tensor = *tensorAlloc;
            RankedTensorType tensorTy = cast<RankedTensorType>(tensor.getType());
            auto shape = tensorTy.getShape();
            auto elementTy = tensorTy.getElementType();
            auto memrefType = MemRefType::get(shape, elementTy);
            auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType, tensor);
            auto memref = toMemrefOp.getResult();
            outmemrefs.push_back(memref);
            newInputs.push_back(memref);
        }

        SmallVector<Type> emptyRets;
        rewriter.create<CallbackCallOp>(loc, emptyRets, callOp.getCallee(), newInputs);
        rewriter.replaceOp(callOp, outmemrefs);
        return success();
    }
};

} // namespace

namespace catalyst {

void populateBufferizationPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<BufferizeCustomCallOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizePrintOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeCallbackOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeCallbackCallOp>(typeConverter, patterns.getContext());
}

} // namespace catalyst
