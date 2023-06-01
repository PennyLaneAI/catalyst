// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {

class BufferizeAdjointOp : public OpConversionPattern<AdjointOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AdjointOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Type> resTypes;
        if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resTypes)))
            return failure();

        Location loc = op.getLoc();
        Value gradSize = op.getGradSize();
        SmallVector<Value> memrefValues;
        for (Type resType : resTypes) {
            MemRefType memrefType = resType.cast<MemRefType>();
            Value memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, gradSize);
            memrefValues.push_back(memrefValue);
        }

        rewriter.create<AdjointOp>(loc, TypeRange{}, op.getCalleeAttr(), adaptor.getGradSize(),
                                   adaptor.getArgs(), memrefValues);
        rewriter.replaceOp(op, memrefValues);
        return success();
    }
};

class BufferizeBackpropOp : public OpConversionPattern<BackpropOp> {
  public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(BackpropOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<Type> resTypes;
        if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resTypes)))
            return failure();

        Location loc = op.getLoc();

        Value gradSize = op.getGradSize();
        ValueRange args = op.getArgs();

        SmallVector<Value> memrefValues;
        size_t resSize = resTypes.size();

        int argsSize = args.size();

        int numCalleeResults = resSize / argsSize;

        for (size_t i = 0; i < resSize; i++) {
            Type resType = resTypes[i];
            std::vector<Value> dynamicDimSizes;

            int argPos = i % numCalleeResults;

            Type argType = args[argPos].getType();

            if (argType.isa<TensorType>()) {
                RankedTensorType rankedArg = argType.cast<RankedTensorType>();
                int numDynDim = rankedArg.getNumDynamicDims();

                for (int i = 0; i < numDynDim; i++) {
                    int dim = rankedArg.getDynamicDimIndex(i);
                    dynamicDimSizes.push_back(
                        rewriter.create<tensor::DimOp>(loc, args[argPos], dim));
                }
            }

            dynamicDimSizes.push_back(gradSize);

            MemRefType memrefType = resType.cast<MemRefType>();
            Value memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, dynamicDimSizes);
            memrefValues.push_back(memrefValue);
        }

        rewriter.create<BackpropOp>(loc, TypeRange{}, op.getCalleeAttr(), adaptor.getGradSize(),
                                    adaptor.getDiffArgIndices(), adaptor.getArgs(),
                                    adaptor.getQuantumJacobians(), memrefValues);
        rewriter.replaceOp(op, memrefValues);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace gradient {

void populateBufferizationPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<BufferizeAdjointOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeBackpropOp>(typeConverter, patterns.getContext());
}

} // namespace gradient
} // namespace catalyst
