#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "Gradient/Utils/EinsumLinalgGeneric.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace llvm;

namespace catalyst {

Value einsumLinalgGeneric(
  OpBuilder& ob,
  Location loc,
  ArrayRef<size_t> a_axis,
  ArrayRef<size_t> b_axis,
  ArrayRef<size_t> r_axis,
  Value a,
  Value b
  )
{
  auto ta = a.getType().cast<TensorType>();
  auto tb = b.getType().cast<TensorType>();
  assert(ta.getElementType() == tb.getElementType() && "element types should match");

  auto axis_dims = ({
    std::map<size_t, size_t> out;
    for(size_t i=0; i<ta.getShape().size(); i++) out[a_axis[i]] = ta.getShape()[i];
    for(size_t i=0; i<tb.getShape().size(); i++) out[b_axis[i]] = tb.getShape()[i];
    out;
  });

  std::vector<int64_t> _r;
  auto tr = ta.cloneWith(({
      for(auto i : r_axis) _r.push_back(axis_dims[i]);
      ArrayRef<int64_t> x(_r);
      x;
    }), ta.getElementType());

  auto maps = ({
    SmallVector<AffineMap> out;
    for (const auto axis : {a_axis, b_axis, r_axis}) {
      SmallVector<AffineExpr> aexprs;
      for(const auto a : axis) {
        aexprs.push_back(getAffineDimExpr(a, ob.getContext()));
      }
      assert(aexprs.size()>0 && "affine expression set should be non-empty");
      out.push_back(AffineMap::get(axis_dims.size(), 0, aexprs, ob.getContext()));
    };
    out;
  });

  auto attrs = ({
    SmallVector<utils::IteratorType, 4> out;
    SmallSetVector<size_t, 4> ua(a_axis.begin(), a_axis.end());
    SmallSetVector<size_t, 4> ub(b_axis.begin(), b_axis.end());
    for (const auto a : axis_dims) {
      out.push_back(
        (ua.contains(a.first) && ub.contains(a.first)) ?
          utils::IteratorType::reduction : utils::IteratorType::parallel
      );
    }
    out;
  });

  Value r = ob.create<tensor::EmptyOp>(loc, tr.getShape(), tr.getElementType());
  SmallVector<Value> operands = {a,b};
  SmallVector<NamedAttribute> nattrs = {};
  auto genOp = ob.create<linalg::GenericOp>(
    loc, tr, operands, r, maps, attrs,
    [](OpBuilder& ob2, Location loc2, ValueRange args) {
      ob2.create<linalg::YieldOp>(loc2,
        Value(
          ob2.create<arith::AddFOp>(loc2, args[2],
            ob2.create<arith::MulFOp>(loc2, args[0], args[1])))
        );
    },
    nattrs);

  assert(genOp.getResults().size() == 1);
  return genOp.getResults()[0];
}



}
