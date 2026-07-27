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

void verifyTypeIsCacheable(Type ty, Operation *op)
{
    // Sanitizing inputs.
    // Technically we know for a fact that none of this will ever issue an
    // error. This is because QubitUnitary is guaranteed to have a
    // tensor<NxNxcomplex<f64>> But this code in the future may be extended to
    // support other types. Hence the sanitization.
    if (ty.isF64()) {
        return;
    }

    // TODO: Generalize to unranked tensors
    if (!isa<RankedTensorType>(ty)) {
        op->emitOpError() << "Caching only supports tensors complex F64";
    }

    auto aTensorType = cast<RankedTensorType>(ty);
    ArrayRef<int64_t> shape = aTensorType.getShape();

    // TODO: Generalize to arbitrary dimensions
    if (2 != shape.size()) {
        op->emitOpError() << "Caching only supports tensors complex F64";
    }
    // TODO: Generalize to other types
    Type elementType = aTensorType.getElementType();
    if (!isa<ComplexType>(elementType)) {
        op->emitOpError() << "Caching only supports tensors complex F64";
    }
    // TODO: Generalize to other types
    Type f64 = cast<ComplexType>(elementType).getElementType();
    if (!f64.isF64()) {
        op->emitOpError() << "Caching only supports tensors complex F64";
    }
}

QuantumCache QuantumCache::initialize(Region &region, OpBuilder &builder, Location loc)
{
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

void QuantumCache::emitDealloc(OpBuilder &builder, Location loc)
{
    ListDeallocOp::create(builder, loc, paramVector);
    ListDeallocOp::create(builder, loc, wireVector);
    for (const auto &[_key, controlFlowTape] : controlFlowTapes) {
        ListDeallocOp::create(builder, loc, controlFlowTape);
    }
}

} // namespace quantum
} // namespace catalyst
