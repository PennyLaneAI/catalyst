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
        // Scalar results are returned by Enzyme while shadows are initialized outside and passed
        // in. These values must be split into inputs and outputs, then recombined when the tensor
        // BackpropOp is replaced.
        SmallVector<Value> shadows;
        SmallVector<Value> gradients;
        // Conceptually a map from scalar result index (w.r.t. other scalars) to the position in the
        // overall list of gradients.
        SmallVector<unsigned> scalarIndices;
        SmallVector<Type> scalarReturnTypes;
        for (auto [idx, result] : llvm::enumerate(results)) {
            Type resType = resTypes[idx];
            if (auto memRefType = dyn_cast<MemRefType>(resType)) {
                Value shadow = rewriter.create<memref::AllocOp>(loc, memRefType);
                shadows.push_back(shadow);
                gradients.push_back(shadow);
            }
            else if (isa<FloatType>(resTypes[idx])) {
                scalarReturnTypes.push_back(resType);
                scalarIndices.push_back(idx);
                // Put a null placeholder value that will be filled in with the result of the
                // bufferized BackpropOp.
                gradients.push_back(Value());
            }
        }

        DenseIntElementsAttr diffArgIndicesAttr = adaptor.getDiffArgIndices().value_or(nullptr);
        auto bufferizedBackpropOp = rewriter.create<BackpropOp>(
            loc, scalarReturnTypes, adaptor.getCalleeAttr(), adaptor.getArgs(),
            adaptor.getQuantumJacobian(), shadows, diffArgIndicesAttr);

        // Fill in the null placeholders.
        for (const auto &[idx, scalarResult] : llvm::enumerate(bufferizedBackpropOp.getResults())) {
            gradients[scalarIndices[idx]] = scalarResult;
        }

        rewriter.replaceOp(op, gradients);
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
