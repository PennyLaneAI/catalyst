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

// Integer/boolean parameters are recorded in a dedicated i64 buffer (`cache.intVector`): each value
// is zero-extended to i64 on push (`arith.extui`) and truncated back on pop (`arith.trunci`), which
// is lossless for every element width <= 64 bits. This covers every integer gate parameter seen in
// practice (a MultiX `tensor<Nxi1>` bitstring, wire indices, control counts, a QROM
// `tensor<Nxi64>` bitstring, ...). Wider integers (e.g. i128) are not supported.
static bool isCacheableInteger(Type ty) {
    auto intType = dyn_cast<IntegerType>(ty);
    return intType && intType.getWidth() <= 64;
}

LogicalResult verifyTypeIsCacheable(Type ty, Operation *op) {
    // Sanitizing inputs.
    // TODO: although OperatorOp params can be arbitrary types, currently only caching of f64s,
    // narrow (<= 53-bit) integers, and complex (and tensors of them) are implemented.
    if (ty.isF64() || isCacheableInteger(ty)) {
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

    // Real-valued tensors of any rank (e.g. `quantum.operator` angle tensors or a BasisRotation
    // matrix) are cached element-wise as plain f64 values. Integer/boolean tensors (e.g. the
    // `tensor<Nxi1>` bitstring of a MultiX gate) are cached the same way via an f64 round-trip,
    // exact only for element widths <= 53 bits (see `isCacheableInteger`; wider ones are rejected).
    if (elementType.isF64() || isCacheableInteger(elementType)) {
        return success();
    }

    // TODO: Generalize to arbitrary dimensions
    if (shape.size() != 2) {
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
