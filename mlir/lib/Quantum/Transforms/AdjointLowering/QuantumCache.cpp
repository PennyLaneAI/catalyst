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

#include "llvm/ADT/StringRef.h"
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

namespace {

// Initial byte capacity, and the alignment, of the parameter cache.
// We assume the param data that needs to be cached will be aligned at "nice" boundaries,
// i.e. all alignments are factors of 64.
// In other words we assume there's no weird param types like i23, f17, i9, ...
constexpr int64_t initialParamVectorCapacity = 2048;
constexpr int64_t paramVectorAlignment = 64;
constexpr llvm::StringLiteral offsetRoundupFuncName =
    "__adjoint_lowering_roundup_offset_to_alignment";
constexpr llvm::StringLiteral ensureCapacityFuncName =
    "__adjoint_lowering_ensure_param_vector_capacity";

// `memref<?xi8>`: the growable raw byte buffer that gate parameters are recorded into.
MemRefType getParamVectorDataType(OpBuilder &builder) {
    return MemRefType::get({ShapedType::kDynamic}, builder.getI8Type());
}

// `memref<memref<?xi8>>`: the rank-0 indirection that holds the byte buffer.
MemRefType getParamVectorType(OpBuilder &builder) {
    return MemRefType::get({}, getParamVectorDataType(builder));
}

// Get or create the helper rounding a raw byte offset up to a required alignment.
func::FuncOp getOrInsertOffsetRoundupFunc(ModuleOp moduleOp, OpBuilder &builder, Location loc) {
    if (auto existing = moduleOp.lookupSymbol<func::FuncOp>(offsetRoundupFuncName)) {
        return existing;
    }

    MLIRContext *ctx = builder.getContext();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    Type indexType = builder.getIndexType();
    auto offsetRoundupFuncType = FunctionType::get(ctx, /*inputs=*/{indexType, indexType},
                                                   /*outputs=*/{indexType});

    auto offsetRoundupFuncOp =
        func::FuncOp::create(builder, loc, offsetRoundupFuncName, offsetRoundupFuncType);
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
    Value offsetAligned = index::AndOp::create(builder, loc, o_plus_a_minus_one, not_a_minus_one);
    func::ReturnOp::create(builder, loc, offsetAligned);

    return offsetRoundupFuncOp;
}

// Get or create the helper that grows the parameter byte buffer.
func::FuncOp getOrInsertEnsureCapacityFunc(ModuleOp moduleOp, OpBuilder &builder, Location loc) {
    if (auto existing = moduleOp.lookupSymbol<func::FuncOp>(ensureCapacityFuncName)) {
        return existing;
    }

    MLIRContext *ctx = builder.getContext();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    Type indexType = builder.getIndexType();
    MemRefType dataType = getParamVectorDataType(builder);
    auto ensureCapacityFuncType =
        FunctionType::get(ctx,
                          /*inputs=*/
                          {getParamVectorType(builder), MemRefType::get({}, indexType), indexType},
                          /*outputs=*/{});

    auto ensureCapacityFuncOp =
        func::FuncOp::create(builder, loc, ensureCapacityFuncName, ensureCapacityFuncType);
    ensureCapacityFuncOp.setPrivate();

    Block *entryBlock = ensureCapacityFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    BlockArgument dataField = ensureCapacityFuncOp.getArgument(0);
    BlockArgument capacityField = ensureCapacityFuncOp.getArgument(1);
    BlockArgument requiredNumBytes = ensureCapacityFuncOp.getArgument(2);

    Value capacity = memref::LoadOp::create(builder, loc, capacityField, ValueRange{});
    Value needsGrowth =
        index::CmpOp::create(builder, loc, builder.getI1Type(), index::IndexCmpPredicate::ULT,
                             capacity, requiredNumBytes);

    scf::IfOp::create(builder, loc, needsGrowth, [&](OpBuilder &thenBuilder, Location loc) {
        // Double the capacity to keep the amortized cost of caching a parameter constant, but
        // never grow to less than what the caller asked for: a single parameter can be larger
        // than the whole current buffer.
        Value two = index::ConstantOp::create(thenBuilder, loc, 2);
        Value doubledCapacity = index::MulOp::create(thenBuilder, loc, capacity, two);
        Value newCapacity =
            index::MaxUOp::create(thenBuilder, loc, doubledCapacity, requiredNumBytes);

        Value oldData = memref::LoadOp::create(thenBuilder, loc, dataField, ValueRange{});
        Value newData = memref::ReallocOp::create(
            thenBuilder, loc, dataType, oldData, newCapacity,
            /*alignment=*/thenBuilder.getI64IntegerAttr(paramVectorAlignment));

        memref::StoreOp::create(thenBuilder, loc, newData, dataField, ValueRange{});
        memref::StoreOp::create(thenBuilder, loc, newCapacity, capacityField, ValueRange{});
        scf::YieldOp::create(thenBuilder, loc);
    });

    func::ReturnOp::create(builder, loc);

    return ensureCapacityFuncOp;
}

} // namespace

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

    Type indexType = builder.getIndexType();

    // The byte buffer that parameters get recorded into. It is dynamically sized and held behind a
    // rank-0 memref so that it can be reallocated in place as the cache fills up; see the comment
    // on QuantumCache::paramVector.
    Value initialCapacity = index::ConstantOp::create(builder, loc, initialParamVectorCapacity);
    auto paramVectorData =
        memref::AllocOp::create(builder, loc, getParamVectorDataType(builder),
                                /*dynamicSizes=*/ValueRange{initialCapacity},
                                /*alignment=*/builder.getI64IntegerAttr(paramVectorAlignment))
            .getMemref();
    auto paramVector =
        memref::AllocOp::create(builder, loc, getParamVectorType(builder)).getMemref();
    memref::StoreOp::create(builder, loc, paramVectorData, paramVector, ValueRange{});

    auto paramVectorCapacity =
        memref::AllocOp::create(builder, loc, MemRefType::get({}, indexType)).getMemref();
    memref::StoreOp::create(builder, loc, initialCapacity, paramVectorCapacity, ValueRange{});

    auto currentOffset =
        memref::AllocOp::create(builder, loc, MemRefType::get({}, indexType)).getMemref();
    auto zero = index::ConstantOp::create(builder, loc, 0);
    memref::StoreOp::create(builder, loc, zero, currentOffset, ValueRange{});
    auto offsetVectorType = ArrayListType::get(ctx, indexType);
    auto offsetVector = ListInitOp::create(builder, loc, offsetVectorType);

    auto moduleOp = region.getParentOfType<ModuleOp>();
    func::FuncOp offsetRoundupFunc = getOrInsertOffsetRoundupFunc(moduleOp, builder, loc);
    func::FuncOp ensureCapacityFunc = getOrInsertEnsureCapacityFunc(moduleOp, builder, loc);

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
                                 .paramVectorCapacity = paramVectorCapacity,
                                 .currentOffset = currentOffset,
                                 .offsetVector = offsetVector,
                                 .offsetRoundupFunc = offsetRoundupFunc,
                                 .ensureCapacityFunc = ensureCapacityFunc,
                                 .wireVector = wireVector,
                                 .controlFlowTapes = controlFlowTapes};
}

void QuantumCache::emitEnsureCapacity(OpBuilder &builder, Location loc,
                                      Value requiredNumBytes) const {
    func::CallOp::create(builder, loc, ensureCapacityFunc,
                         ValueRange{paramVector, paramVectorCapacity, requiredNumBytes});
}

Value QuantumCache::emitLoadParamVectorData(OpBuilder &builder, Location loc) const {
    return memref::LoadOp::create(builder, loc, paramVector, ValueRange{}).getResult();
}

void QuantumCache::emitDealloc(OpBuilder &builder, Location loc) {
    memref::DeallocOp::create(builder, loc, emitLoadParamVectorData(builder, loc));
    memref::DeallocOp::create(builder, loc, paramVector);
    memref::DeallocOp::create(builder, loc, paramVectorCapacity);
    memref::DeallocOp::create(builder, loc, currentOffset);
    ListDeallocOp::create(builder, loc, offsetVector);
    ListDeallocOp::create(builder, loc, wireVector);
    for (const auto &[_key, controlFlowTape] : controlFlowTapes) {
        ListDeallocOp::create(builder, loc, controlFlowTape);
    }
}

} // namespace quantum
} // namespace catalyst
