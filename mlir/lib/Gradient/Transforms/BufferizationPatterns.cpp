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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
        SmallVector<Value> tensorValues;
        int i = 0;
        for (Type resType : resTypes) {
            MemRefType memrefType = resType.cast<MemRefType>();
            Value memrefValue = rewriter.create<memref::AllocOp>(loc, memrefType, gradSize);
            Value tensorValue = rewriter.create<bufferization::ToTensorOp>(
                loc, op.getResultTypes()[i++], memrefValue);
            memrefValues.push_back(memrefValue);
            tensorValues.push_back(tensorValue);
        }

        op.replaceAllUsesWith(tensorValues);
        rewriter.create<AdjointOp>(loc, TypeRange{}, op.getCalleeAttr(), adaptor.getArgs(),
                                   adaptor.getGradSize(), memrefValues);
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace gradient {

void populateBufferizationPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<BufferizeAdjointOp>(typeConverter, patterns.getContext());
}

} // namespace gradient
} // namespace catalyst
