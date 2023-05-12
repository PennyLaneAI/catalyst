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

Value einsumLinalgGeneric(OpBuilder &ob, Location loc, ArrayRef<size_t> a_axis,
                          ArrayRef<size_t> b_axis, ArrayRef<size_t> r_axis, Value a, Value b)
{
    auto ta = a.getType().cast<TensorType>();
    auto tb = b.getType().cast<TensorType>();
    assert(ta.getElementType() == tb.getElementType() && "element types should match");

    auto axis_dims = ({
        std::map<size_t, size_t> out;
        for (size_t i = 0; i < ta.getShape().size(); i++)
            out[a_axis[i]] = ta.getShape()[i];
        for (size_t i = 0; i < tb.getShape().size(); i++)
            out[b_axis[i]] = tb.getShape()[i];
        out;
    });

    auto r_shape = ({
        std::vector<int64_t> out;
        for (auto i : r_axis)
            out.push_back(axis_dims[i]);
        out;
    });

    auto tr = ta.cloneWith(ArrayRef<int64_t>(r_shape), ta.getElementType());

    auto maps = ({
        SmallVector<AffineMap> out;
        for (const auto axis : {a_axis, b_axis, r_axis}) {
            SmallVector<AffineExpr> aexprs;
            for (const auto a : axis) {
                aexprs.push_back(getAffineDimExpr(a, ob.getContext()));
            }
            out.push_back(AffineMap::get(axis_dims.size(), 0, aexprs, ob.getContext()));
        };
        out;
    });

    auto attrs = ({
        SmallVector<utils::IteratorType, 4> out;
        SmallSetVector<size_t, 4> ua(a_axis.begin(), a_axis.end());
        SmallSetVector<size_t, 4> ub(b_axis.begin(), b_axis.end());
        for (const auto a : axis_dims) {
            out.push_back((ua.contains(a.first) && ub.contains(a.first))
                              ? utils::IteratorType::reduction
                              : utils::IteratorType::parallel);
        }
        out;
    });

    Value r = ({
        Value empty = ob.create<tensor::EmptyOp>(loc, tr.getShape(), tr.getElementType());
        Value zero =
            ob.create<arith::ConstantOp>(loc, tr.getElementType(), ob.getF64FloatAttr(0.0));
        ob.create<linalg::FillOp>(loc, zero, empty).getResult(0);
    });

    SmallVector<Value> operands = {a, b};
    SmallVector<NamedAttribute> nattrs = {};
    auto genOp = ob.create<linalg::GenericOp>(
        loc, tr, operands, r, maps, attrs,
        [](OpBuilder &ob2, Location loc2, ValueRange args) {
            ob2.create<linalg::YieldOp>(
                loc2, Value(ob2.create<arith::AddFOp>(
                          loc2, args[2], ob2.create<arith::MulFOp>(loc2, args[0], args[1]))));
        },
        nattrs);

    assert(genOp.getResults().size() == 1);
    return genOp.getResults()[0];
}

} // namespace catalyst
