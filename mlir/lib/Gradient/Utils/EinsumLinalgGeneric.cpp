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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"

#include "Gradient/Utils/EinsumLinalgGeneric.h"

using namespace mlir;
using namespace llvm;

namespace catalyst {

Value einsumLinalgGeneric(OpBuilder &ob, Location loc, ArrayRef<int64_t> axisCodesA,
                          ArrayRef<int64_t> axisCodesB, ArrayRef<int64_t> axisCodesResult, Value a,
                          Value b, std::optional<Value> bufferOut)
{
    bool useBufferSemantics = bufferOut.has_value();
    if (useBufferSemantics) {
        assert(isa<MemRefType>(a.getType()) && isa<MemRefType>(b.getType()) &&
               isa<MemRefType>(bufferOut->getType()) &&
               "einsumLinalgGeneric with buffer output expects operands and output to have "
               "MemRefType");
    }
    else {
        assert(
            isa<RankedTensorType>(a.getType()) && isa<RankedTensorType>(b.getType()) &&
            "einsumLinalgGeneric with no buffer output expects operands to have RankedTensorType");
    }

    auto ta = cast<ShapedType>(a.getType());
    auto tb = cast<ShapedType>(b.getType());
    assert(ta.getElementType() == tb.getElementType() && "element types should match");

    std::map<int64_t, int64_t> axisDims;
    {
        for (size_t i = 0; i < ta.getShape().size(); i++)
            axisDims[axisCodesA[i]] = ta.getShape()[i];
        for (size_t i = 0; i < tb.getShape().size(); i++)
            axisDims[axisCodesB[i]] = tb.getShape()[i];
    }

    std::vector<int64_t> shapeR;
    {
        for (auto i : axisCodesResult) {
            shapeR.push_back(axisDims[i]);
        }
    }

    ShapedType resultType;
    if (useBufferSemantics) {
        resultType = MemRefType::get(shapeR, ta.getElementType());
    }
    else {
        resultType = RankedTensorType::get(shapeR, ta.getElementType());
    }

    SmallVector<AffineMap> maps;
    {
        for (const auto axis : {axisCodesA, axisCodesB, axisCodesResult}) {
            SmallVector<AffineExpr> aexprs;
            for (const auto a : axis) {
                aexprs.push_back(getAffineDimExpr(a, ob.getContext()));
            }
            maps.push_back(AffineMap::get(axisDims.size(), 0, aexprs, ob.getContext()));
        };
    }

    SmallVector<utils::IteratorType, 4> iteratorTypes;
    {
        SmallSetVector<int64_t, 4> ua(axisCodesA.begin(), axisCodesA.end());
        SmallSetVector<int64_t, 4> ub(axisCodesB.begin(), axisCodesB.end());
        for (const auto a : axisDims) {
            iteratorTypes.push_back((ua.contains(a.first) && ub.contains(a.first))
                                        ? utils::IteratorType::reduction
                                        : utils::IteratorType::parallel);
        }
    }

    Value r;
    if (useBufferSemantics) {
        r = bufferOut.value();
    }
    else {
        Value empty =
            ob.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
        Value zero =
            ob.create<arith::ConstantOp>(loc, resultType.getElementType(), ob.getF64FloatAttr(0.0));
        r = ob.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    }

    SmallVector<Value> operands = {a, b};
    auto bodyBuilder = [](OpBuilder &builder, Location loc, ValueRange args) {
        builder.create<linalg::YieldOp>(
            loc, Value(builder.create<arith::AddFOp>(
                     loc, args[2], builder.create<arith::MulFOp>(loc, args[0], args[1]))));
    };
    if (useBufferSemantics) {
        ob.create<linalg::GenericOp>(loc, operands, r, maps, iteratorTypes, bodyBuilder);
        return r;
    }
    else {
        auto genOp = ob.create<linalg::GenericOp>(loc, resultType, operands, r, maps, iteratorTypes,
                                                  bodyBuilder);
        return genOp.getResult(0);
    }
}

} // namespace catalyst
