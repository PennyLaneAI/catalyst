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

#include <algorithm>
#include <iostream> // FIXME
#include <vector>

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

struct TensorInitOpRewritePattern : public OpRewritePattern<TensorInitOp> {
    using OpRewritePattern<TensorInitOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TensorInitOp op, PatternRewriter &rewriter) const override
    {
        auto loc = op.getLoc();
        SmallVector<int64_t, 4> dimensions;
        SmallVector<Value> shapeValues;
        {
            auto ctx = op.getContext();
            auto shape = op.getShape();
            auto shapeType = shape.getType();
            auto elementType = shapeType.getElementType();

            auto shapeShape = shapeType.getShape();
            assert(shapeShape.size() == 1 && "shape argument must have '1xi..' shape");
            assert(shapeShape[0] >= 0 && "dynamic elements in shapes are not supported!");

            for (int64_t i = 0; i < shapeShape[0]; i++) {
                std::cerr << i << "\n";
                Value axis = rewriter.create<arith::ConstantIndexOp>(loc, i);
                Value val = rewriter.create<tensor::ExtractOp>(loc, elementType, shape, axis);
                Value ival = rewriter.create<arith::IndexCastOp>(loc, IndexType::get(ctx), val);
                shapeValues.push_back(ival);
                dimensions.push_back(ShapedType::kDynamic);
            }
        }

        auto [type, value] = getInitTypeValue(op, rewriter);
        auto tensorType = RankedTensorType::get(dimensions, type);
        Value result = rewriter.create<tensor::EmptyOp>(loc, tensorType, shapeValues);
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
