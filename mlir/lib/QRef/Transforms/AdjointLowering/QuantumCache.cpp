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

#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Catalyst/IR/CatalystOps.h"
#include "PBC/IR/PBCOps.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace qref {

bool hasQuantumOperations(Operation *op) {
    llvm::SmallPtrSet<Operation *, 8> visited;
    auto inspect = [&](auto &&inspect, Operation *root) -> bool {
        if (!visited.insert(root).second) {
            return false;
        }
        return root
            ->walk([&](Operation *nested) {
                if ((isa<QRefDialect>(nested->getDialect()) && !isa<GetOp>(nested)) ||
                    isa<pbc::RefPPRotationOp>(nested)) {
                    return WalkResult::interrupt();
                }
                if (auto call = dyn_cast<func::CallOp>(nested)) {
                    if (llvm::any_of(call.getOperandTypes(), llvm::IsaPred<QubitType, QuregType>) ||
                        llvm::any_of(call.getResultTypes(), llvm::IsaPred<QubitType, QuregType>)) {
                        return WalkResult::interrupt();
                    }
                    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                        call, call.getCalleeAttr());
                    if (callee && inspect(inspect, callee)) {
                        return WalkResult::interrupt();
                    }
                }
                return WalkResult::advance();
            })
            .wasInterrupted();
    };
    return inspect(inspect, op);
}

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
    // OperatorOp parameters can have arbitrary types; the cache supports the types below.
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

    // Real-valued tensors of any rank (e.g. `qref.operator` angle tensors or a BasisRotation
    // matrix) are cached element-wise as plain f64 values. Integer/boolean tensors (e.g. the
    // `tensor<Nxi1>` bitstring of a MultiX gate) use lossless integer extension and truncation.
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
    auto paramVectorType = ArrayListType::get(ctx, builder.getF64Type());
    auto intVectorType = ArrayListType::get(ctx, builder.getI64Type());
    auto wireVectorType = ArrayListType::get(ctx, builder.getI64Type());
    auto controlValueVectorType = ArrayListType::get(ctx, builder.getI1Type());
    auto controlFlowTapeType = ArrayListType::get(ctx, builder.getIndexType());
    auto paramVector = ListInitOp::create(builder, loc, paramVectorType);
    auto intVector = ListInitOp::create(builder, loc, intVectorType);
    auto wireVector = ListInitOp::create(builder, loc, wireVectorType);
    auto controlValueVector = ListInitOp::create(builder, loc, controlValueVectorType);

    // Initialize the tapes that store the structure of control flow.
    DenseMap<Operation *, TypedValue<ArrayListType>> controlFlowTapes;
    region.walk([&](Operation *op) {
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp>(op)) {
            auto tape = catalyst::ListInitOp::create(builder, loc, controlFlowTapeType);
            controlFlowTapes.insert({op, tape});
        }
    });
    return QuantumCache{.paramVector = paramVector,
                        .intVector = intVector,
                        .wireVector = wireVector,
                        .controlValueVector = controlValueVector,
                        .controlFlowTapes = controlFlowTapes};
}

void QuantumCache::emitDealloc(OpBuilder &builder, Location loc) {
    ListDeallocOp::create(builder, loc, paramVector);
    ListDeallocOp::create(builder, loc, intVector);
    ListDeallocOp::create(builder, loc, wireVector);
    ListDeallocOp::create(builder, loc, controlValueVector);
    for (const auto &[_key, controlFlowTape] : controlFlowTapes) {
        ListDeallocOp::create(builder, loc, controlFlowTape);
    }
}

} // namespace qref
} // namespace catalyst
