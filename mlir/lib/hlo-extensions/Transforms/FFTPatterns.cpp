// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "fft"

#include <cmath>
#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace catalyst {
namespace hlo_extensions {

namespace {

// The multidimensional DFT is separable so stablehlo.fft is lowered to a
// chain of batched 1D stages with one stage per transform axis.
enum class StageKind {
    C2CForward, // complex to complex with twiddle exp(-2*pi*i*j*k/L) and no scale
    C2CInverse, // complex to complex with twiddle exp(+2*pi*i*j*k/L) scaled by 1/L
    R2C,        // real to complex forward producing bins k = 0..floor(L/2)
    C2R         // complex to real inverse from the floor(L/2)+1 stored bins scaled by 1/L
};

bool isForwardStage(StageKind kind)
{
    return kind == StageKind::C2CForward || kind == StageKind::R2C;
}

Value createFloatConst(OpBuilder &builder, Location loc, FloatType type, double value)
{
    APFloat apValue(value);
    bool losesInfo = false;
    apValue.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
    return arith::ConstantFloatOp::create(builder, loc, type, apValue);
}

// Emit a batched 1D DFT stage along `axis` of `input` as a linalg.generic
// reduction in destination passing style:
//
//   out[b..., k, b'...] = scale * sum_j in[b..., j, b'...] * T(j, k)
//
// The twiddle T(j, k) = exp(sign * 2*pi*i * ((j*k) mod L) / L) is computed per
// index with L given by `modulus`. The exponent is reduced mod L in exact
// integer arithmetic before the float conversion so the twiddle phase error
// stays O(eps) independent of the magnitude of j*k. The reduction extent is
// the input extent along `axis` and `outLen` is the output extent along it.
// Inverse direction stages fold the 1/L normalization into the twiddles.
//
// All arithmetic is componentwise on the real and imaginary parts to avoid
// complex.mul whose default lowering carries inf and nan fixup branches.
// Angles are evaluated in f64 and truncated at the end so f32 transforms do
// not lose twiddle accuracy.
Value emitDFTStage(OpBuilder &builder, Location loc, Value input, int64_t axis, int64_t modulus,
                   int64_t outLen, StageKind kind, Type outElemType)
{
    MLIRContext *ctx = builder.getContext();
    auto inType = cast<RankedTensorType>(input.getType());
    int64_t rank = inType.getRank();
    int64_t numBins = inType.getDimSize(axis);

    SmallVector<int64_t> outShape(inType.getShape());
    outShape[axis] = outLen;
    auto outType = RankedTensorType::get(outShape, outElemType);

    FloatType realType;
    if (auto complexType = dyn_cast<ComplexType>(outElemType)) {
        realType = cast<FloatType>(complexType.getElementType());
    }
    else {
        realType = cast<FloatType>(outElemType);
    }

    // Batch dims may be dynamic. The transform axis is static by construction.
    SmallVector<Value> dynSizes;
    for (int64_t dim = 0; dim < rank; ++dim) {
        if (outType.isDynamicDim(dim)) {
            dynSizes.push_back(tensor::DimOp::create(builder, loc, input, dim));
        }
    }
    Value initTensor = tensor::EmptyOp::create(builder, loc, outShape, outElemType, dynSizes);

    Value zero;
    if (isa<ComplexType>(outElemType)) {
        auto zeroAttr = builder.getFloatAttr(realType, 0.0);
        zero = complex::ConstantOp::create(builder, loc, outElemType,
                                           builder.getArrayAttr({zeroAttr, zeroAttr}));
    }
    else {
        zero = createFloatConst(builder, loc, realType, 0.0);
    }
    Value accInit = linalg::FillOp::create(builder, loc, zero, initTensor).getResult(0);

    // Loop dims d0..d_{rank-1} index the output and d_rank is the reduction.
    // The input map substitutes the reduction dim for the transform axis.
    SmallVector<AffineExpr> inExprs, outExprs;
    for (int64_t dim = 0; dim < rank; ++dim) {
        inExprs.push_back(getAffineDimExpr(dim == axis ? rank : dim, ctx));
        outExprs.push_back(getAffineDimExpr(dim, ctx));
    }
    SmallVector<AffineMap> indexingMaps = {AffineMap::get(rank + 1, 0, inExprs, ctx),
                                           AffineMap::get(rank + 1, 0, outExprs, ctx)};
    SmallVector<utils::IteratorType> iteratorTypes(rank, utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);

    double sign = isForwardStage(kind) ? -1.0 : 1.0;
    double angleFactor = sign * 2.0 * M_PI / static_cast<double>(modulus);
    double scale = isForwardStage(kind) ? 1.0 : 1.0 / static_cast<double>(modulus);

    auto bodyBuilder = [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
        Value x = args[0];
        Value acc = args[1];
        auto f64Type = b.getF64Type();

        Value k = linalg::IndexOp::create(b, bodyLoc, axis);
        Value j = linalg::IndexOp::create(b, bodyLoc, rank);

        // Twiddle angle with exact integer reduction of the exponent.
        Value modulusCst = arith::ConstantIndexOp::create(b, bodyLoc, modulus);
        Value jk = arith::MulIOp::create(b, bodyLoc, j, k);
        Value m = arith::RemUIOp::create(b, bodyLoc, jk, modulusCst);
        Value mInt = arith::IndexCastOp::create(b, bodyLoc, b.getI64Type(), m);
        Value mFloat = arith::SIToFPOp::create(b, bodyLoc, f64Type, mInt);
        Value angleCst = createFloatConst(b, bodyLoc, f64Type, angleFactor);
        Value theta = arith::MulFOp::create(b, bodyLoc, mFloat, angleCst);

        Value twiddleRe = math::CosOp::create(b, bodyLoc, theta);
        Value twiddleIm = math::SinOp::create(b, bodyLoc, theta);
        if (scale != 1.0) {
            Value scaleCst = createFloatConst(b, bodyLoc, f64Type, scale);
            twiddleRe = arith::MulFOp::create(b, bodyLoc, twiddleRe, scaleCst);
            twiddleIm = arith::MulFOp::create(b, bodyLoc, twiddleIm, scaleCst);
        }
        if (!realType.isF64()) {
            twiddleRe = arith::TruncFOp::create(b, bodyLoc, realType, twiddleRe);
            twiddleIm = arith::TruncFOp::create(b, bodyLoc, realType, twiddleIm);
        }

        switch (kind) {
        case StageKind::C2CForward:
        case StageKind::C2CInverse: {
            Value xRe = complex::ReOp::create(b, bodyLoc, realType, x);
            Value xIm = complex::ImOp::create(b, bodyLoc, realType, x);
            Value accRe = complex::ReOp::create(b, bodyLoc, realType, acc);
            Value accIm = complex::ImOp::create(b, bodyLoc, realType, acc);
            // (xRe + i*xIm) * (tRe + i*tIm)
            Value prodRe =
                arith::SubFOp::create(b, bodyLoc, arith::MulFOp::create(b, bodyLoc, xRe, twiddleRe),
                                      arith::MulFOp::create(b, bodyLoc, xIm, twiddleIm));
            Value prodIm =
                arith::AddFOp::create(b, bodyLoc, arith::MulFOp::create(b, bodyLoc, xRe, twiddleIm),
                                      arith::MulFOp::create(b, bodyLoc, xIm, twiddleRe));
            Value newRe = arith::AddFOp::create(b, bodyLoc, accRe, prodRe);
            Value newIm = arith::AddFOp::create(b, bodyLoc, accIm, prodIm);
            Value result = complex::CreateOp::create(b, bodyLoc, outElemType, newRe, newIm);
            linalg::YieldOp::create(b, bodyLoc, result);
            break;
        }
        case StageKind::R2C: {
            Value accRe = complex::ReOp::create(b, bodyLoc, realType, acc);
            Value accIm = complex::ImOp::create(b, bodyLoc, realType, acc);
            Value newRe = arith::AddFOp::create(b, bodyLoc, accRe,
                                                arith::MulFOp::create(b, bodyLoc, x, twiddleRe));
            Value newIm = arith::AddFOp::create(b, bodyLoc, accIm,
                                                arith::MulFOp::create(b, bodyLoc, x, twiddleIm));
            Value result = complex::CreateOp::create(b, bodyLoc, outElemType, newRe, newIm);
            linalg::YieldOp::create(b, bodyLoc, result);
            break;
        }
        case StageKind::C2R: {
            Value xRe = complex::ReOp::create(b, bodyLoc, realType, x);
            Value xIm = complex::ImOp::create(b, bodyLoc, realType, x);
            // Re(x * e^{i*theta}) with the 1/L scale already in the twiddle.
            Value base =
                arith::SubFOp::create(b, bodyLoc, arith::MulFOp::create(b, bodyLoc, xRe, twiddleRe),
                                      arith::MulFOp::create(b, bodyLoc, xIm, twiddleIm));
            // Hermitian symmetry weight. Interior bins stand in for their
            // conjugate mirror as well and count twice. The DC bin at j = 0
            // and for even L the Nyquist bin at j = L/2 are self conjugate
            // and count once.
            Value one = createFloatConst(b, bodyLoc, realType, 1.0);
            Value two = createFloatConst(b, bodyLoc, realType, 2.0);
            Value zeroIdx = arith::ConstantIndexOp::create(b, bodyLoc, 0);
            Value isDC = arith::CmpIOp::create(b, bodyLoc, arith::CmpIPredicate::eq, j, zeroIdx);
            Value weight = arith::SelectOp::create(b, bodyLoc, isDC, one, two);
            if (modulus % 2 == 0) {
                Value nyquistIdx = arith::ConstantIndexOp::create(b, bodyLoc, numBins - 1);
                Value isNyquist =
                    arith::CmpIOp::create(b, bodyLoc, arith::CmpIPredicate::eq, j, nyquistIdx);
                weight = arith::SelectOp::create(b, bodyLoc, isNyquist, one, weight);
            }
            Value contrib = arith::MulFOp::create(b, bodyLoc, base, weight);
            Value result = arith::AddFOp::create(b, bodyLoc, acc, contrib);
            linalg::YieldOp::create(b, bodyLoc, result);
            break;
        }
        }
    };

    auto genericOp =
        linalg::GenericOp::create(builder, loc, TypeRange{outType}, ValueRange{input},
                                  ValueRange{accInit}, indexingMaps, iteratorTypes, bodyBuilder);
    return genericOp.getResult(0);
}

struct FftOpRewritePattern : public OpRewritePattern<stablehlo::FftOp> {
    using OpRewritePattern<stablehlo::FftOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::FftOp op, PatternRewriter &rewriter) const override
    {
        auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
        auto resultType = dyn_cast<RankedTensorType>(op.getType());
        if (!operandType || !resultType) {
            return rewriter.notifyMatchFailure(op, "expected ranked tensor operand and result");
        }

        ArrayRef<int64_t> fftLength = op.getFftLength();
        int64_t rank = operandType.getRank();
        int64_t numAxes = static_cast<int64_t>(fftLength.size());
        if (numAxes < 1 || numAxes > rank) {
            return rewriter.notifyMatchFailure(op, "invalid fft_length rank");
        }
        for (int64_t i = 0; i < numAxes; ++i) {
            int64_t axis = rank - numAxes + i;
            if (fftLength[i] <= 0) {
                return rewriter.notifyMatchFailure(op, "non-positive fft length");
            }
            if (operandType.isDynamicDim(axis) || resultType.isDynamicDim(axis)) {
                return rewriter.notifyMatchFailure(op, "dynamic extent on a transform axis");
            }
        }
        // The stablehlo verifier guarantees the shape relations below. Check
        // them anyway so a violation degrades into an op that is not lowered
        // rather than silently wrong IR.
        stablehlo::FftType fftType = op.getFftType();
        int64_t lastLength = fftLength[numAxes - 1];
        for (int64_t i = 0; i < numAxes; ++i) {
            int64_t axis = rank - numAxes + i;
            int64_t expected = fftLength[i];
            if (fftType == stablehlo::FftType::IRFFT && i == numAxes - 1) {
                expected = lastLength / 2 + 1;
            }
            if (operandType.getDimSize(axis) != expected) {
                return rewriter.notifyMatchFailure(op, "operand shape inconsistent w/ fft_length");
            }
        }

        Location loc = op.getLoc();
        Value current = op.getOperand();

        switch (fftType) {
        case stablehlo::FftType::FFT:
        case stablehlo::FftType::IFFT: {
            Type complexElemType = operandType.getElementType();
            StageKind kind =
                fftType == stablehlo::FftType::FFT ? StageKind::C2CForward : StageKind::C2CInverse;
            for (int64_t i = 0; i < numAxes; ++i) {
                int64_t axis = rank - numAxes + i;
                current = emitDFTStage(rewriter, loc, current, axis, fftLength[i], fftLength[i],
                                       kind, complexElemType);
            }
            break;
        }
        case stablehlo::FftType::RFFT: {
            Type complexElemType = resultType.getElementType();
            // Real to complex along the last axis first which produces the
            // halved axis and then forward transforms along the other axes.
            current = emitDFTStage(rewriter, loc, current, rank - 1, lastLength, lastLength / 2 + 1,
                                   StageKind::R2C, complexElemType);
            for (int64_t i = 0; i < numAxes - 1; ++i) {
                int64_t axis = rank - numAxes + i;
                current = emitDFTStage(rewriter, loc, current, axis, fftLength[i], fftLength[i],
                                       StageKind::C2CForward, complexElemType);
            }
            break;
        }
        case stablehlo::FftType::IRFFT: {
            Type complexElemType = operandType.getElementType();
            // Mirror image of RFFT. Inverse transforms along the leading axes
            // first and complex to real reconstruction along the last axis.
            // Each stage folds its own 1/L factor and the product is 1/N.
            for (int64_t i = 0; i < numAxes - 1; ++i) {
                int64_t axis = rank - numAxes + i;
                current = emitDFTStage(rewriter, loc, current, axis, fftLength[i], fftLength[i],
                                       StageKind::C2CInverse, complexElemType);
            }
            current = emitDFTStage(rewriter, loc, current, rank - 1, lastLength, lastLength,
                                   StageKind::C2R, resultType.getElementType());
            break;
        }
        }

        if (current.getType() != resultType) {
            current = tensor::CastOp::create(rewriter, loc, resultType, current);
        }
        rewriter.replaceOp(op, current);
        return success();
    }
};

} // namespace

void populateFFTPatterns(RewritePatternSet &patterns)
{
    patterns.add<FftOpRewritePattern>(patterns.getContext());
}

} // namespace hlo_extensions
} // namespace catalyst
