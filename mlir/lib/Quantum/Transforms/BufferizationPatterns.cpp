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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
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
        rewriter.replaceOpWithNewOp<QubitUnitaryOp>(
            op, op.getOutQubits().getTypes(), op.getOutCtrlQubits().getTypes(), adaptor.getMatrix(),
            adaptor.getInQubits(), adaptor.getAdjointAttr(), adaptor.getInCtrlQubits(),
            adaptor.getInCtrlValues());
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
        Type tensorType = op.getType(0);
        MemRefType resultType = cast<MemRefType>(getTypeConverter()->convertType(tensorType));
        Location loc = op.getLoc();

        // SampleOp's result shape is (shots, num_qubits)
        // shots is a SSA argument to the device and can potentially be dynamic
        // so we need to memref alloc from the shots SSA value
        auto shape = cast<mlir::RankedTensorType>(tensorType).getShape();
        SmallVector<Value> allocSizes;

        if (shape[0] == ShapedType::kDynamic) {
            auto parentFunc = op->getParentOfType<func::FuncOp>();
            SmallVector<DeviceInitOp> DeviceInitOpPool;
            parentFunc->walk(
                [&](DeviceInitOp deviceInitOp) { DeviceInitOpPool.push_back(deviceInitOp); });
            assert(DeviceInitOpPool.size() == 1 &&
                   "quantum.sample operation is only valid when either inside a function with "
                   "exactly one shot-ful device init operation, or have allocated memref as input");

            auto shots = rewriter.create<index::CastSOp>(loc, rewriter.getIndexType(),
                                                         DeviceInitOpPool[0].getShots());

            allocSizes.push_back(shots);
        }

        Value allocVal = rewriter.replaceOpWithNewOp<memref::AllocOp>(op, resultType, allocSizes);
        rewriter.create<SampleOp>(loc, TypeRange{}, ValueRange{adaptor.getObs(), allocVal},
                                  op->getAttrs());

        return success();
    }
};

struct BufferizeStateOp : public OpConversionPattern<StateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(StateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Type tensorType = op.getType(0);
        MemRefType resultType = cast<MemRefType>(getTypeConverter()->convertType(tensorType));
        Location loc = op.getLoc();
        Value allocVal = rewriter.replaceOpWithNewOp<memref::AllocOp>(op, resultType);
        rewriter.create<StateOp>(loc, TypeRange{}, ValueRange{adaptor.getObs(), allocVal});
        return success();
    }
};

struct BufferizeProbsOp : public OpConversionPattern<ProbsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ProbsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Type tensorType = op.getType(0);
        MemRefType resultType = cast<MemRefType>(getTypeConverter()->convertType(tensorType));
        Location loc = op.getLoc();
        Value allocVal = rewriter.replaceOpWithNewOp<memref::AllocOp>(op, resultType);
        rewriter.create<ProbsOp>(loc, TypeRange{}, ValueRange{adaptor.getObs(), allocVal});
        return success();
    }
};

struct BufferizeCountsOp : public OpConversionPattern<CountsOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CountsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Location loc = op.getLoc();
        Type tensorType0 = op.getType(0);
        Type tensorType1 = op.getType(1);
        MemRefType resultType0 = cast<MemRefType>(getTypeConverter()->convertType(tensorType0));
        MemRefType resultType1 = cast<MemRefType>(getTypeConverter()->convertType(tensorType1));
        Value allocVal0 = rewriter.create<memref::AllocOp>(loc, resultType0);
        Value allocVal1 = rewriter.create<memref::AllocOp>(loc, resultType1);
        rewriter.replaceOp(op, ValueRange{allocVal0, allocVal1});
        rewriter.create<CountsOp>(loc, nullptr, nullptr, adaptor.getObs(), allocVal0, allocVal1);

        return success();
    }
};

struct BufferizeSetStateOp : public OpConversionPattern<SetStateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(SetStateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Type tensorType = op.getInState().getType();
        MemRefType memrefType = cast<MemRefType>(getTypeConverter()->convertType(tensorType));
        auto toMemrefOp =
            rewriter.create<bufferization::ToMemrefOp>(op->getLoc(), memrefType, op.getInState());
        auto memref = toMemrefOp.getResult();
        rewriter.replaceOpWithNewOp<SetStateOp>(op, op.getOutQubits().getTypes(), memref,
                                                adaptor.getInQubits());
        return success();
    }
};

struct BufferizeSetBasisStateOp : public OpConversionPattern<SetBasisStateOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(SetBasisStateOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        Type tensorType = op.getBasisState().getType();
        MemRefType memrefType = cast<MemRefType>(getTypeConverter()->convertType(tensorType));
        auto toMemrefOp = rewriter.create<bufferization::ToMemrefOp>(op->getLoc(), memrefType,
                                                                     op.getBasisState());
        auto memref = toMemrefOp.getResult();
        rewriter.replaceOpWithNewOp<SetBasisStateOp>(op, op.getOutQubits().getTypes(), memref,
                                                     adaptor.getInQubits());
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
    target.addDynamicallyLegalOp<SampleOp>([&](SampleOp op) { return op.isBufferized(); });
    target.addDynamicallyLegalOp<StateOp>([&](StateOp op) { return op.isBufferized(); });
    target.addDynamicallyLegalOp<ProbsOp>([&](ProbsOp op) { return op.isBufferized(); });
    target.addDynamicallyLegalOp<CountsOp>([&](CountsOp op) { return op.isBufferized(); });
    target.addDynamicallyLegalOp<SetStateOp>([&](SetStateOp op) { return op.isBufferized(); });
    target.addDynamicallyLegalOp<SetBasisStateOp>(
        [&](SetBasisStateOp op) { return op.isBufferized(); });
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
    patterns.add<BufferizeSetStateOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeSetBasisStateOp>(typeConverter, patterns.getContext());
}

} // namespace quantum
} // namespace catalyst
