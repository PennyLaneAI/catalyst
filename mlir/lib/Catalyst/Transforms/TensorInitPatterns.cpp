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
#include <vector>
#include <iostream> // FIXME

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Catalyst/IR/CatalystOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinTypes.h"

#include "mhlo/IR/hlo_ops.h"


using namespace mlir;
using namespace catalyst;

namespace {

struct TensorInitOpRewritePattern : public OpRewritePattern<TensorInitOp> {
    using OpRewritePattern<TensorInitOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TensorInitOp op,
                                  PatternRewriter &rewriter) const override
    {
        auto ctx = op.getContext();
        auto loc = op.getLoc();
        auto method = op.getMethod();

        if (method == "ones") {
        }
        else if (method == "zeros") {
        }
        else if (method == "empty") {
            auto shape = op.getShape();
            auto shapeType = shape.getType();
            auto elementType = shapeType.getElementType(); // FIXME: query from parameter

            SmallVector<Value> shapeValues;
            auto shapeShape = shapeType.getShape();
            assert(shapeShape.size() == 1 && "shape argument must have '1xi..' shape");
            assert(shapeShape[0] >= 0 && "dynamic elements in shapes are not supported!");

            SmallVector<int64_t, 4> dimensions;
            for (int64_t i = 0; i < shapeShape[0]; i++) {
                std::cerr << i << "\n";
                Value axis = rewriter.create<arith::ConstantIndexOp>(loc, i);
                Value val = rewriter.create<tensor::ExtractOp>(loc, elementType, shape, axis);
                Value val2 = rewriter.create<arith::IndexCastOp>(loc, IndexType::get(ctx), val);
                shapeValues.push_back(val2);
                dimensions.push_back(ShapedType::kDynamic);
            }

            auto tt = RankedTensorType::get(dimensions, elementType);
            Value result = rewriter.create<tensor::EmptyOp>(loc, tt, shapeValues);

            rewriter.replaceOp(op, result);
            return success();
        }
        else {
            assert(false && "invalid method attribute");
        }
        return failure();
    }

};

} // namespace


namespace catalyst {

void populateTensorInitPatterns(RewritePatternSet &patterns)
{
    patterns.add<TensorInitOpRewritePattern>(patterns.getContext(), 1);
}

}
