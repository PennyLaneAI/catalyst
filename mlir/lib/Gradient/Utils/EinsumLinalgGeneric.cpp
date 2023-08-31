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

namespace catalyst {

Value buildBufferLinalgGeneric(OpBuilder &builder, Location loc, ValueRange operands, Value output,
                               ArrayRef<AffineMap> indexingMaps,
                               ArrayRef<utils::IteratorType> iteratorTypes,
                               function_ref<void(OpBuilder &, Location, ValueRange)> buildBody)
{
    builder.create<linalg::GenericOp>(loc, operands, output, indexingMaps, iteratorTypes,
                                      buildBody);
    return output;
}

Value buildTensorLinalgGeneric(OpBuilder &builder, Location loc, ValueRange operands,
                               RankedTensorType resultType, ArrayRef<AffineMap> indexingMaps,
                               ArrayRef<utils::IteratorType> iteratorTypes,
                               function_ref<void(OpBuilder &, Location, ValueRange)> buildBody)
{
    // Initialize the result tensor
    FloatType elementType = cast<FloatType>(resultType.getElementType());
    Value zero = builder.create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(elementType.getFloatSemantics()), elementType);
    Value result =
        builder.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    result = builder.create<linalg::FillOp>(loc, zero, result).getResult(0);

    auto genericOp = builder.create<linalg::GenericOp>(loc, resultType, operands, result,
                                                       indexingMaps, iteratorTypes, buildBody);
    return genericOp.getResult(0);
}

void inferIndexingMaps(MLIRContext *ctx, unsigned numDims, ArrayRef<int64_t> axisCodesA,
                       ArrayRef<int64_t> axisCodesB, ArrayRef<int64_t> axisCodesResult,
                       SmallVectorImpl<AffineMap> &indexingMaps)
{
    for (const auto axis : {axisCodesA, axisCodesB, axisCodesResult}) {
        SmallVector<AffineExpr> aexprs;
        for (const auto a : axis) {
            aexprs.push_back(getAffineDimExpr(a, ctx));
        }
        indexingMaps.push_back(AffineMap::get(numDims, 0, aexprs, ctx));
    };
}

void inferIteratorTypes(const std::map<int64_t, int64_t> &axisDims,
                        ArrayRef<int64_t> axisCodesResult,
                        SmallVectorImpl<utils::IteratorType> &iteratorTypes)
{
    DenseSet<int64_t> outCodes{axisCodesResult.begin(), axisCodesResult.end()};
    for (const auto &[code, _size] : axisDims) {
        iteratorTypes.push_back(outCodes.contains(code) ? utils::IteratorType::reduction
                                                        : utils::IteratorType::parallel);
    }
}

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

    // Create an ordered map from axis codes to the size of the corresponding dimension
    std::map<int64_t, int64_t> axisDims;
    {
        for (size_t i = 0; i < ta.getShape().size(); i++)
            axisDims[axisCodesA[i]] = ta.getShape()[i];
        for (size_t i = 0; i < tb.getShape().size(); i++)
            axisDims[axisCodesB[i]] = tb.getShape()[i];
    }

    SmallVector<AffineMap> maps;
    SmallVector<utils::IteratorType> iteratorTypes;
    inferIndexingMaps(ob.getContext(), axisDims.size(), axisCodesA, axisCodesB, axisCodesResult,
                      maps);
    inferIteratorTypes(axisDims, axisCodesResult, iteratorTypes);
    auto bodyBuilder = [](OpBuilder &builder, Location loc, ValueRange args) {
        builder.create<linalg::YieldOp>(
            loc, Value(builder.create<arith::AddFOp>(
                     loc, args[2], builder.create<arith::MulFOp>(loc, args[0], args[1]))));
    };

    if (useBufferSemantics) {
        return buildBufferLinalgGeneric(ob, loc, {a, b}, *bufferOut, maps, iteratorTypes,
                                        bodyBuilder);
    }

    SmallVector<int64_t> resultShape(axisCodesResult.size());
    llvm::transform(axisCodesResult, resultShape.begin(),
                    [&](int64_t code) { return axisDims[code]; });
    auto resultType = RankedTensorType::get(resultShape, ta.getElementType());
    return buildTensorLinalgGeneric(ob, loc, {a, b}, resultType, maps, iteratorTypes, bodyBuilder);
}

} // namespace catalyst
