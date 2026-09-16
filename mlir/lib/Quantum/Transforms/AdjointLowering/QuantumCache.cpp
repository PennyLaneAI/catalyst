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

#include "QuantumCache.hpp"

#include <cstdint>

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace quantum {

LogicalResult verifyTypeIsCacheable(Type ty, Operation *op) {
    // Sanitizing inputs. Every rejection must return failure: `emitOpError` only records a
    // diagnostic, it does not unwind, so falling through to the casts below would abort the
    // compiler on an unsupported type instead of reporting it.
    if (ty.isF64()) {
        return success();
    }

    // TODO: Generalize to unranked tensors
    if (!isa<RankedTensorType>(ty)) {
        return op->emitOpError() << "Caching only supports F64 and tensors of complex F64, got "
                                 << ty;
    }

    auto aTensorType = cast<RankedTensorType>(ty);
    ArrayRef<int64_t> shape = aTensorType.getShape();
    Type elementType = aTensorType.getElementType();

    // Real-valued scalar/rank-1 tensors (e.g. `quantum.operator` angle tensors) are cached
    // element-wise as plain f64 values.
    if (elementType.isF64()) {
        if (shape.size() > 1) {
            return op->emitOpError()
                   << "Caching only supports scalar or rank-1 real F64 tensors, got " << ty;
        }
        return success();
    }

    // TODO: Generalize to arbitrary dimensions
    if (2 != shape.size()) {
        return op->emitOpError() << "Caching only supports rank-2 tensors of complex F64, got "
                                 << ty;
    }
    // TODO: Generalize to other types
    auto complexType = dyn_cast<ComplexType>(elementType);
    if (!complexType) {
        return op->emitOpError() << "Caching only supports tensors of complex F64, got " << ty;
    }
    // TODO: Generalize to other types
    if (!complexType.getElementType().isF64()) {
        return op->emitOpError() << "Caching only supports tensors of complex F64, got " << ty;
    }
    return success();
}

bool isAvailableToReversePass(Value param, Region &adjointRegion) {
    Region *definingRegion = param.getParentRegion();
    // Defined outside the adjoint region: dominates the adjoint operation itself.
    if (!definingRegion || !adjointRegion.isAncestor(definingRegion)) {
        return true;
    }
    // Defined at the immediate top level of the adjoint region: cloned by the forward pass to the
    // insertion point the reverse pass continues from. Anything deeper lives inside control flow
    // that the forward pass rebuilds, and must be recorded.
    return definingRegion == &adjointRegion;
}

QuantumCache QuantumCache::initialize(Region &region, OpBuilder &builder, Location loc) {
    MLIRContext *ctx = builder.getContext();
    auto paramVectorType = ArrayListType::get(ctx, builder.getF64Type());
    auto wireVectorType = ArrayListType::get(ctx, builder.getI64Type());
    auto controlFlowTapeType = ArrayListType::get(ctx, builder.getIndexType());
    auto paramVector = ListInitOp::create(builder, loc, paramVectorType);
    auto wireVector = ListInitOp::create(builder, loc, wireVectorType);

    // Initialize the tapes that store the structure of control flow.
    DenseMap<Operation *, TypedValue<ArrayListType>> controlFlowTapes;
    region.walk([&](Operation *op) {
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp>(op)) {
            auto tape = catalyst::ListInitOp::create(builder, loc, controlFlowTapeType);
            controlFlowTapes.insert({op, tape});
        }
    });
    return quantum::QuantumCache{
        .paramVector = paramVector, .wireVector = wireVector, .controlFlowTapes = controlFlowTapes};
}

void QuantumCache::emitDealloc(OpBuilder &builder, Location loc) {
    ListDeallocOp::create(builder, loc, paramVector);
    ListDeallocOp::create(builder, loc, wireVector);
    for (const auto &[_key, controlFlowTape] : controlFlowTapes) {
        ListDeallocOp::create(builder, loc, controlFlowTape);
    }
}

} // namespace quantum
} // namespace catalyst
