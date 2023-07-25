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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Utils/GradientShape.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {

Value generateAllocation(OpBuilder &builder, Location loc, Value reference)
{
    auto memrefType = cast<MemRefType>(reference.getType());
    // Get dynamic dimension sizes from the provided reference value if necessary.
    SmallVector<Value> dynamicDims;
    if (!memrefType.hasStaticShape()) {
        for (int64_t dim = 0; dim < memrefType.getRank(); dim++) {
            if (memrefType.isDynamicDim(dim)) {
                Value dimIndex = builder.create<index::ConstantOp>(loc, dim);
                dynamicDims.push_back(builder.create<memref::DimOp>(loc, reference, dimIndex));
            }
        }
    }

    return builder.create<memref::AllocOp>(loc, memrefType, dynamicDims);
}

/// Helper function to generate a set of memref allocations.
///
/// The allocation size and shape is deduced from a list of existing memref values.
///
void generateAllocations(PatternRewriter &rewriter, Location loc,
                         SmallVectorImpl<Value> &allocations, ValueRange referenceValues)
{
    for (Value memref : referenceValues) {
        allocations.push_back(
            generateAllocation(rewriter, loc, cast<TypedValue<MemRefType>>(memref)));
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
        SmallVector<Type> resTypes;
        SmallVector<Type> outTypes;
        if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resTypes))) {
            return failure();
        }
        if (failed(getTypeConverter()->convertTypes(op.getOutputs().getTypes(), outTypes))) {
            return failure();
        }

        Location loc = op.getLoc();
        SmallVector<Value> argShadows;
        SmallVector<Value> outShadows;
        for (Type resType : resTypes) {
            argShadows.push_back(rewriter.create<memref::AllocOp>(loc, cast<MemRefType>(resType)));
        }

        Value one =
            rewriter.create<arith::ConstantFloatOp>(loc, APFloat(1.0), rewriter.getF64Type());
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        for (Type outType : outTypes) {
            Value outShadow = rewriter.create<memref::AllocOp>(loc, cast<MemRefType>(outType));
            rewriter.create<memref::StoreOp>(loc, one, outShadow, c0);
            outShadows.push_back(outShadow);
        }

        rewriter.create<BackpropOp>(loc, TypeRange{}, op.getCalleeAttr(), adaptor.getArgs(),
                                    adaptor.getOutputs(), argShadows, outShadows,
                                    op.getDiffArgIndicesAttr());
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
