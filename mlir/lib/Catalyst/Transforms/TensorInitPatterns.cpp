// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "tensorinit"

#include "Catalyst/IR/CatalystOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/IR/BuiltinTypes.h"

#include "mhlo/IR/hlo_ops.h"

using namespace mlir;
using namespace catalyst;

namespace {

// Extract and return the type and a value of the initializer.
std::tuple<Type, std::optional<Value>> getInitTypeValue(TensorInitOp op, PatternRewriter &rewriter)
{
    using llvm::cast, llvm::isa;
    using std::tuple, std::optional, std::nullopt;

    auto loc = op.getLoc();
    auto empty = op.getEmpty().has_value();
    auto initializer = op.getInitializerAttr();
    auto type = initializer.getType();
    std::optional<Value> emptyValue;

    std::optional<Value> value =
        empty ? emptyValue : rewriter.create<arith::ConstantOp>(loc, type, initializer);
    return tuple(type, value);
}

// Rewrite catalyst.tensor_init as tensor.empty optionally followed by linalg.fill
struct TensorInitOpRewritePattern : public OpRewritePattern<TensorInitOp> {
    using OpRewritePattern<TensorInitOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TensorInitOp op, PatternRewriter &rewriter) const override
    {
        // Statically-known part of the shape, with possible kDynamic placeholders
        // Dynamic part
        SmallVector<Value> dynamicShapeValues;
        auto staticShape = cast<RankedTensorType>(op.getResult().getType()).getShape();
        auto loc = op.getLoc();
        {
            auto ctx = op.getContext();
            auto shape = op.getShape();
            auto shapeType =
                shape ? shape.getType() : RankedTensorType::get({0}, IndexType::get(ctx));
            auto elementType = shapeType.getElementType();
            auto dynShapeShape = shapeType.getShape();
            auto staticShapeShape = shapeType.getShape();

            assert(staticShapeShape.size() == 1 && "static shape argument must have '1xi..' shape");

            int64_t d = 0;
            for (size_t s = 0; s < staticShape.size(); s++) {
                if (staticShape[s] < 0) {
                    assert(dynShapeShape.size() == 1 && "dynamic shape argument must be '1xi..'");
                    assert(d < dynShapeShape[0] && "not enough dynamic shape arguments");
                    Value axis = rewriter.create<arith::ConstantIndexOp>(loc, s);
                    Value val = rewriter.create<tensor::ExtractOp>(loc, elementType, shape, axis);
                    Value ival = rewriter.create<arith::IndexCastOp>(loc, IndexType::get(ctx), val);
                    dynamicShapeValues.push_back(ival);
                    d++;
                }
            }
            assert(d == 0 ||
                   d == dynShapeShape[0] &&
                       "Number of dynamic items must match the number of static placeholders");
        }

        auto [type, value] = getInitTypeValue(op, rewriter);
        auto tensorType = RankedTensorType::get(staticShape, type);
        Value result = rewriter.create<tensor::EmptyOp>(loc, tensorType, dynamicShapeValues);
        if (value.has_value()) {
            result = rewriter.create<linalg::FillOp>(loc, value.value(), result).getResult(0);
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

} // namespace

namespace catalyst {

void populateTensorInitPatterns(RewritePatternSet &patterns)
{
    patterns.add<TensorInitOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace catalyst
