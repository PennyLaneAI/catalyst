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

#include "iostream"
#include "llvm/Support/raw_ostream.h"

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
        ValueRange results = op.getResults();
        SmallVector<Value> memrefValues;
        SmallVector<Value> dynamicDimSizes;
        for (auto [resType, result] : zip(resTypes, results)) {
            if (resType.isa<TensorType>()) {
                RankedTensorType rankedArg = resType.cast<RankedTensorType>();
                int numDynDim = rankedArg.getNumDynamicDims();
                for (int i = 0; i < numDynDim; i++) {
                    int dim = rankedArg.getDynamicDimIndex(i);
                    dynamicDimSizes.push_back(rewriter.create<tensor::DimOp>(loc, result, dim));
                }
            }
            MemRefType memrefType = resType.cast<MemRefType>();
            Value memrefValue;
            if (!dynamicDimSizes.empty()) {
                memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, dynamicDimSizes);
            }
            else {
                memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType);
            }
            memrefValues.push_back(memrefValue);
        }

        DenseIntElementsAttr diffArgIndicesAttr = adaptor.getDiffArgIndices().value_or(nullptr);
        rewriter.create<BackpropOp>(loc, TypeRange{}, adaptor.getCalleeAttr(), adaptor.getArgs(),
                                    adaptor.getQuantumJacobian(), memrefValues, diffArgIndicesAttr);
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
