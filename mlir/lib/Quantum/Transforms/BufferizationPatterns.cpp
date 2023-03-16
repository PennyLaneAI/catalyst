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
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

struct BufferizeQubitUnitaryOp : public OpConversionPattern<QubitUnitaryOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(QubitUnitaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<QubitUnitaryOp>(op, op.getResultTypes(), adaptor.getMatrix(),
                                                    adaptor.getInQubits());
        return success();
    }
};

struct BufferizeHermitianOp : public OpConversionPattern<HermitianOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(HermitianOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<HermitianOp>(op, op.getType(), adaptor.getMatrix(),
                                                 adaptor.getQubits());
        return success();
    }
};

struct BufferizeHamiltonianOp : public OpConversionPattern<HamiltonianOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(HamiltonianOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<HamiltonianOp>(op, op.getType(), adaptor.getCoeffs(),
                                                   adaptor.getTerms());
        return success();
    }
};

struct BufferizeSampleOp : public OpConversionPattern<SampleOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(SampleOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MemRefType resultType = getTypeConverter()->convertType(op.getType()).cast<MemRefType>();
        Location loc = op.getLoc();
        auto allocOp = rewriter.create<memref::AllocOp>(loc, resultType);
        rewriter.replaceOpWithNewOp<SampleOp>(op, resultType, adaptor.getObs(),
                                              allocOp->getResult(0), adaptor.getShotsAttr());
        return success();
    }
};

struct BufferizeStateOp : public OpConversionPattern<StateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(StateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {

        MemRefType resultType = getTypeConverter()->convertType(op.getType()).cast<MemRefType>();
        Location loc = op.getLoc();
        auto allocOp = rewriter.create<memref::AllocOp>(loc, resultType);
        rewriter.replaceOpWithNewOp<StateOp>(op, TypeRange{resultType},
                                             ValueRange{adaptor.getObs(), allocOp->getResult(0)});
        return success();
    }
};

struct BufferizeProbsOp : public OpConversionPattern<ProbsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ProbsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        MemRefType resultType = getTypeConverter()->convertType(op.getType()).cast<MemRefType>();
        Location loc = op.getLoc();
        auto allocOp = rewriter.create<memref::AllocOp>(loc, resultType);
        rewriter.replaceOpWithNewOp<ProbsOp>(op, TypeRange{resultType},
                                             ValueRange{adaptor.getObs(), allocOp->getResult(0)});
        return success();
    }
};

struct BufferizeCountsOp : public OpConversionPattern<CountsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CountsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        auto resultType0 = getTypeConverter()->convertType(op.getType(0));
        auto resultType1 = getTypeConverter()->convertType(op.getType(1));
        SmallVector<Type, 2> resultType = {resultType0, resultType1};
        rewriter.replaceOpWithNewOp<CountsOp>(op, resultType, adaptor.getObs(),
                                              adaptor.getShotsAttr());
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateBufferizationLegality(TypeConverter &typeConverter, ConversionTarget &target)
{
    // Default to operations being legal with the exception of the ones below.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    // Quantum ops which return arrays need to be marked illegal when the type is a tensor.
    target.addDynamicallyLegalOp<QubitUnitaryOp>(
        [&](QubitUnitaryOp op) { return typeConverter.isLegal(op.getMatrix().getType()); });
    target.addDynamicallyLegalOp<HermitianOp>(
        [&](HermitianOp op) { return typeConverter.isLegal(op.getMatrix().getType()); });
    target.addDynamicallyLegalOp<HamiltonianOp>(
        [&](HamiltonianOp op) { return typeConverter.isLegal(op.getCoeffs().getType()); });
    target.addDynamicallyLegalOp<SampleOp>(
        [&](SampleOp op) { return typeConverter.isLegal(op.getType()); });
    target.addDynamicallyLegalOp<StateOp>(
        [&](StateOp op) { return typeConverter.isLegal(op.getType()); });
    target.addDynamicallyLegalOp<ProbsOp>(
        [&](ProbsOp op) { return typeConverter.isLegal(op.getType()); });
    target.addDynamicallyLegalOp<CountsOp>([&](CountsOp op) {
        return typeConverter.isLegal(op.getType(0)) and typeConverter.isLegal(op.getType(1));
    });
}

void populateBufferizationPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns)
{
    patterns.add<BufferizeQubitUnitaryOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeHermitianOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeHamiltonianOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeSampleOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeStateOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeProbsOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeCountsOp>(typeConverter, patterns.getContext());
}

} // namespace quantum
} // namespace catalyst
