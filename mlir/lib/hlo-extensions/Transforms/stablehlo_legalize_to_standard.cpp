// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file is taken from the
//   tensorflow/mlir-hlo
// repository, under the Apache 2.0 License, at
//   https://github.com/tensorflow/mlir-hlo/blob/a5529d99fc4d1132b0c282a053d26c11e6636b3a/mhlo/transforms/legalize_to_standard/legalize_to_standard.cc
// with the following copyright notice:

/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// The modifications are porting the pass from the upstream stablehlo namespace to
// catalyst namespace.

// This file implements logic for lowering Stablehlo dialect to Standard dialect.

#include <memory>
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

using namespace mlir;
using namespace stablehlo;

namespace {
class CompareIConvert : public OpRewritePattern<stablehlo::CompareOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::CompareOp op, PatternRewriter &rewriter) const override
    {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto lhsType = mlir::cast<TensorType>(lhs.getType());
        auto rhsType = mlir::cast<TensorType>(rhs.getType());

        // Broadcasting not supported by this rewrite.
        if (lhsType.getShape() != rhsType.getShape())
            return failure();

        if (!lhsType.getElementType().isSignlessInteger() ||
            !rhsType.getElementType().isSignlessInteger())
            return failure();

        std::optional<arith::CmpIPredicate> comparePredicate = std::nullopt;
        switch (op.getComparisonDirection()) {
        case ComparisonDirection::EQ:
            comparePredicate = arith::CmpIPredicate::eq;
            break;
        case ComparisonDirection::NE:
            comparePredicate = arith::CmpIPredicate::ne;
            break;
        case ComparisonDirection::LT:
            comparePredicate = arith::CmpIPredicate::slt;
            break;
        case ComparisonDirection::LE:
            comparePredicate = arith::CmpIPredicate::sle;
            break;
        case ComparisonDirection::GT:
            comparePredicate = arith::CmpIPredicate::sgt;
            break;
        case ComparisonDirection::GE:
            comparePredicate = arith::CmpIPredicate::sge;
            break;
        }

        if (!comparePredicate.has_value())
            return failure();

        rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, comparePredicate.value(), lhs, rhs);
        return success();
    }
};

class CompareFConvert : public OpRewritePattern<stablehlo::CompareOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::CompareOp op, PatternRewriter &rewriter) const override
    {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto lhsType = mlir::cast<TensorType>(lhs.getType());
        auto rhsType = mlir::cast<TensorType>(rhs.getType());

        // Broadcasting not supported by this rewrite.
        if (lhsType.getShape() != rhsType.getShape())
            return failure();

        if (!mlir::isa<FloatType>(lhsType.getElementType()) ||
            !mlir::isa<FloatType>(rhsType.getElementType()))
            return failure();

        std::optional<arith::CmpFPredicate> comparePredicate = std::nullopt;
        switch (op.getComparisonDirection()) {
        case ComparisonDirection::EQ:
            comparePredicate = arith::CmpFPredicate::OEQ;
            break;
        case ComparisonDirection::NE:
            comparePredicate = arith::CmpFPredicate::UNE;
            break;
        case ComparisonDirection::LT:
            comparePredicate = arith::CmpFPredicate::OLT;
            break;
        case ComparisonDirection::LE:
            comparePredicate = arith::CmpFPredicate::OLE;
            break;
        case ComparisonDirection::GT:
            comparePredicate = arith::CmpFPredicate::OGT;
            break;
        case ComparisonDirection::GE:
            comparePredicate = arith::CmpFPredicate::OGE;
            break;
        }

        if (!comparePredicate.has_value())
            return failure();

        rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, comparePredicate.value(), lhs, rhs);
        return success();
    }
};

// Replace IotaOp with an integer constant. A ConvertOp is added to
// convert the integer constant to iota result type. For complex types, the real
// part is replaced with the generated constant and the imaginary part is
// replaced with zero tensor.
class ConvertIotaOp : public OpRewritePattern<stablehlo::IotaOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(stablehlo::IotaOp op, PatternRewriter &rewriter) const override
    {
        auto outputType = mlir::cast<ShapedType>(op.getType());
        auto outputSize = outputType.getNumElements();
        auto dimension = op.getIotaDimension();
        auto maxDimSize = outputType.getDimSize(dimension);

        auto elementType = outputType.getElementType();
        int bitwidth;

        auto complexTy = mlir::dyn_cast<ComplexType>(elementType);
        Type intOrFloatTy = elementType;
        if (complexTy)
            intOrFloatTy = complexTy.getElementType();

        bitwidth = intOrFloatTy.getIntOrFloatBitWidth();
        llvm::SmallVector<APInt, 10> values;
        values.reserve(outputSize);

        int64_t increaseStride = outputSize;
        for (uint64_t i = 0; i <= dimension; i++) {
            increaseStride /= outputType.getDimSize(i);
        }

        int64_t currentValue = 0;
        for (int i = 0; i < outputSize; i++) {
            int64_t value = (currentValue / increaseStride) % maxDimSize;
            values.push_back(APInt(bitwidth, value));
            ++currentValue;
        }

        auto intShapeType = RankedTensorType::get(
            outputType.getShape(), IntegerType::get(rewriter.getContext(), bitwidth));
        auto loc = op.getLoc();
        auto integerConst = mlir::arith::ConstantOp::create(rewriter,
            loc, DenseIntElementsAttr::get(intShapeType, values));

        auto intOrFloatShapeTy = RankedTensorType::get(outputType.getShape(), intOrFloatTy);

        auto iotaConst = ConvertOp::create(rewriter, loc, intOrFloatShapeTy, integerConst);

        // For int/float types we are done, replace op and return.
        if (!complexTy) {
            rewriter.replaceOp(op, iotaConst.getResult());
            return success();
        }

        // For complex types, generate a constant tensor of zeroes for the imaginary
        // part and use iota_const for real part.
        auto zeroes = mlir::arith::ConstantOp::create(rewriter,
            loc, DenseIntElementsAttr::get(intShapeType, APInt(bitwidth, 0)));
        auto imagZeroes = ConvertOp::create(rewriter, loc, intOrFloatShapeTy, zeroes);
        rewriter.replaceOpWithNewOp<stablehlo::ComplexOp>(op, iotaConst, imagZeroes);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace hlo_extensions {

#define GEN_PASS_DEF_STABLEHLOLEGALIZETOSTANDARDPASS
#include "hlo-extensions/Transforms/Passes.h.inc"
#include "hlo-extensions/Transforms/generated_stablehlo_legalize_to_standard.cpp.inc"

void populateStablehloToStdPatterns(RewritePatternSet *patterns, mlir::MLIRContext *ctx)
{
    populateWithGenerated(*patterns);
    patterns->add<CompareFConvert, CompareIConvert, ConvertIotaOp>(ctx);
}

struct StablehloLegalizeToStandardPass
    : public impl::StablehloLegalizeToStandardPassBase<StablehloLegalizeToStandardPass> {
    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect, math::MathDialect, func::FuncDialect>();
    }

    /// Perform the lowering to Standard dialect.
    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());
        populateStablehloToStdPatterns(&patterns, &getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace hlo_extensions
} // namespace catalyst
