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

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Catalyst/IR/CatalystOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace quantum {

bool isAvailableToReversePass(Value param, Region &adjointRegion) {
    Region *definingRegion = param.getParentRegion();

    // Defined outside the adjoint region: dominates the adjoint operation itself, reverse pass
    // sees it from above directly.
    if (!definingRegion || !adjointRegion.isAncestor(definingRegion)) {
        return true;
    }

    // Defined at the immediate top level of the adjoint region, not in nested control flow
    return definingRegion == &adjointRegion;
}

QuantumCache QuantumCache::initialize(Region &region, OpBuilder &builder, Location loc) {
    MLIRContext *ctx = builder.getContext();

    Type byteSizeType = builder.getI8Type();
    uint32_t defaultSize = 2048; // just some default size for now
    auto paramVector =
        memref::AllocOp::create(builder, loc, MemRefType::get({defaultSize}, byteSizeType))
            .getMemref();

    auto currentOffset =
        memref::AllocOp::create(builder, loc, MemRefType::get({}, builder.getIndexType()))
            .getMemref();
    auto zero = index::ConstantOp::create(builder, loc, 0);
    memref::StoreOp::create(builder, loc, zero, currentOffset, ValueRange{});
    auto offsetVectorType = ArrayListType::get(ctx, builder.getIndexType());
    auto offsetVector = ListInitOp::create(builder, loc, offsetVectorType);

    auto wireVectorType = ArrayListType::get(ctx, builder.getI64Type());
    auto wireVector = ListInitOp::create(builder, loc, wireVectorType);

    // Initialize the tapes that store the structure of control flow.
    auto controlFlowTapeType = ArrayListType::get(ctx, builder.getIndexType());
    DenseMap<Operation *, TypedValue<ArrayListType>> controlFlowTapes;
    region.walk([&](Operation *op) {
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp>(op)) {
            auto tape = catalyst::ListInitOp::create(builder, loc, controlFlowTapeType);
            controlFlowTapes.insert({op, tape});
        }
    });
    return quantum::QuantumCache{.paramVector = paramVector,
                                 .currentOffset = currentOffset,
                                 .offsetVector = offsetVector,
                                 .wireVector = wireVector,
                                 .controlFlowTapes = controlFlowTapes};
}

void QuantumCache::emitDealloc(OpBuilder &builder, Location loc) {
    memref::DeallocOp::create(builder, loc, paramVector);
    memref::DeallocOp::create(builder, loc, currentOffset);
    ListDeallocOp::create(builder, loc, offsetVector);
    ListDeallocOp::create(builder, loc, wireVector);
    for (const auto &[_key, controlFlowTape] : controlFlowTapes) {
        ListDeallocOp::create(builder, loc, controlFlowTape);
    }
}

} // namespace quantum
} // namespace catalyst
