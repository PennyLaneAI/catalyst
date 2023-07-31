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

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Utils/GradientShape.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {

/// Helper function to generate a set of memref allocations.
///
/// The allocation size and shape is deduced from a list of existing memref values.
///
void generateAllocations(PatternRewriter &rewriter, Location loc,
                         SmallVectorImpl<Value> &allocations, ValueRange referenceValues)
{
    for (Value memref : referenceValues) {
        MemRefType memrefType = cast<MemRefType>(memref.getType());

        // Get dynamic dimension sizes from the provided reference value if necessary.
        SmallVector<Value> dynamicDims;
        if (!memrefType.hasStaticShape()) {
            for (int64_t dim = 0; dim < memrefType.getRank(); dim++) {
                if (memrefType.isDynamicDim(dim)) {
                    Value dimIndex = rewriter.create<index::ConstantOp>(loc, dim);
                    dynamicDims.push_back(rewriter.create<memref::DimOp>(loc, memref, dimIndex));
                }
            }
        }

        Value allocation = rewriter.create<memref::AllocOp>(loc, memrefType, dynamicDims);
        allocations.push_back(allocation);
    }
}

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
        Location loc = op.getLoc();

        // Allocate buffers to place the differentiation results (gradients) into. Enzyme refers to
        // these as shadow arguments. There is one result for each differentiable argument, with a
        // matching shape and type.
        SmallVector<Value> argShadows;
        const std::vector<Value> &diffArgs =
            computeDiffArgs(adaptor.getArgs(), op.getDiffArgIndices());
        generateAllocations(rewriter, loc, argShadows, diffArgs);

        // Enzyme requires buffers for the primal outputs as well, even though we don't need their
        // values. We'll mark them dupNoNeed later on to allow Enzyme to optimize away their
        // computation.
        SmallVector<Value> calleeResults;
        ValueRange resShadows = adaptor.getCotangents();
        generateAllocations(rewriter, loc, calleeResults, resShadows);

        DenseIntElementsAttr diffArgIndicesAttr = adaptor.getDiffArgIndices().value_or(nullptr);
        rewriter.create<BackpropOp>(loc, TypeRange{}, op.getCalleeAttr(), adaptor.getArgs(),
                                    argShadows, calleeResults, resShadows, diffArgIndicesAttr);

        rewriter.replaceOp(op, argShadows);
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
