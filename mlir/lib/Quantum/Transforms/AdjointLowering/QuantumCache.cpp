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

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

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

LogicalResult verifyTypeIsCacheable(Type ty, Operation *op) {
    auto isIntOrFloatOrComplex = [](Type ty) -> bool {
        if (ty.isIntOrFloat()) {
            return true;
        }
        if (auto complexTy = dyn_cast<mlir::ComplexType>(ty)) {
            return complexTy.getElementType().isIntOrFloat();
        }
        return false;
    };

    if (isIntOrFloatOrComplex(ty)) {
        return success();
    }

    if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
        if (!tensorTy.hasStaticShape()) {
            return op->emitError()
                   << "Caching does not support dynamic shape tensors yet, got " << ty;
        }
        if (isIntOrFloatOrComplex(tensorTy.getElementType())) {
            return success();
        }
    }

    return op->emitOpError() << "Caching only supports scalar and tensor types, got " << ty;
}

QuantumCache QuantumCache::initialize(Region &region, OpBuilder &builder, Location loc) {
    MLIRContext *ctx = builder.getContext();

    Type byteSizeType = builder.getI8Type();
    uint32_t defaultSize = 2048; // just some default size for now
    auto paramVector =
        memref::AllocOp::create(builder, loc, MemRefType::get({defaultSize}, byteSizeType),
                                /*alignment =*/builder.getI64IntegerAttr(64))
            .getMemref();

    auto currentOffset =
        memref::AllocOp::create(builder, loc, MemRefType::get({}, builder.getIndexType()))
            .getMemref();
    auto zero = index::ConstantOp::create(builder, loc, 0);
    memref::StoreOp::create(builder, loc, zero, currentOffset, ValueRange{});
    auto offsetVectorType = ArrayListType::get(ctx, builder.getIndexType());
    auto offsetVector = ListInitOp::create(builder, loc, offsetVectorType);

    std::string funcName = "__adjoint_lowering_roundup_offset_to_alignment";
    auto moduleOp = region.getParentOfType<ModuleOp>();
    auto offsetRoundupFunc = moduleOp.lookupSymbol(funcName);
    // Check if the helper already exists in the module
    if (!offsetRoundupFunc) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());

        Type indexType = builder.getIndexType();
        auto offsetRoundupFuncType = FunctionType::get(ctx, /*inputs=*/{indexType, indexType},
                                                       /*outputs=*/{indexType});

        offsetRoundupFunc = func::FuncOp::create(builder, loc, funcName, offsetRoundupFuncType);
        func::FuncOp offsetRoundupFuncOp = cast<func::FuncOp>(offsetRoundupFunc);
        offsetRoundupFuncOp.setPrivate();

        // The formula to round up current offset (O) to intended alignment (A) is
        // O_aligned = (O + A - 1) & ~(A - 1)
        // given that A is a power of 2
        Block *entryBlock = offsetRoundupFuncOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        BlockArgument rawOffset = offsetRoundupFuncOp.getArgument(0);
        BlockArgument alignment = offsetRoundupFuncOp.getArgument(1);

        Value one = index::ConstantOp::create(builder, loc, 1);
        Value a_minus_one = index::SubOp::create(builder, loc, alignment, one);
        Value o_plus_a_minus_one = index::AddOp::create(builder, loc, rawOffset, a_minus_one);
        // mlir doesn't have an instruction for bitwise NOT
        // need to XOR with a mask of all 1
        Value all_bit_ones = index::ConstantOp::create(builder, loc, -1);
        Value not_a_minus_one = index::XOrOp::create(builder, loc, a_minus_one, all_bit_ones);
        Value offsetAligned =
            index::AndOp::create(builder, loc, o_plus_a_minus_one, not_a_minus_one);
        func::ReturnOp::create(builder, loc, offsetAligned);
    }

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
                                 .offsetRoundupFunc = cast<func::FuncOp>(offsetRoundupFunc),
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
