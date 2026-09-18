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

#include <cassert>
#include <cstdint>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace quantum {

FailureOr<ParamStorageInfo> classifyParamStorage(Type ty) {
    if (isa<TensorType>(ty)) {
        auto rankedType = dyn_cast<RankedTensorType>(ty);
        if (!rankedType) {
            // An unranked tensor carries no shape for the reverse pass to rebuild from.
            return failure();
        }

        Type elementType = rankedType.getElementType();
        if (!MemRefType::isValidElementType(elementType)) {
            return failure();
        }

        // A rank-0 tensor holds a single element and has no rank-1 buffer to append, so it is
        // cached as one scalar.
        if (rankedType.getRank() == 0) {
            return ParamStorageInfo{ParamStorage::Scalar, elementType, /*hasDynamicShape=*/false};
        }

        // Every other tensor is cached as a single block of elements, whatever its rank, element
        // type or dynamic dimensions.
        return ParamStorageInfo{ParamStorage::WholeTensor, elementType,
                                /*hasDynamicShape=*/!rankedType.hasStaticShape()};
    }

    // Any non-shaped type an array list can store, which covers every scalar the frontend produces:
    // integers, indices, floats and complex values.
    if (!MemRefType::isValidElementType(ty)) {
        return failure();
    }
    return ParamStorageInfo{ParamStorage::Scalar, ty, /*hasDynamicShape=*/false};
}

LogicalResult verifyTypeIsCacheable(Type ty, Operation *op) {
    if (failed(classifyParamStorage(ty))) {
        return op->emitOpError()
               << "adjoint lowering cannot cache a parameter of type " << ty
               << ": expected a scalar (integer, index, float or complex) or a ranked tensor of "
                  "those";
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
    auto wireVectorType = ArrayListType::get(ctx, builder.getI64Type());
    auto controlFlowTapeType = ArrayListType::get(ctx, builder.getIndexType());
    auto wireVector = ListInitOp::create(builder, loc, wireVectorType);

    // Create one list per parameter layout the forward pass will use. Deciding this through
    // `classifyParamStorage`, with the same "already available" filter the two passes apply,
    // guarantees that every push and every pop finds the list it needs.
    llvm::MapVector<Type, TypedValue<ArrayListType>> paramVectors;
    bool hasDynamicallyShapedParam = false;
    region.walk([&](quantum::ParametrizedGate gate) {
        for (Value param : gate.getAllParams()) {
            if (isAvailableToReversePass(param, region)) {
                continue;
            }
            FailureOr<ParamStorageInfo> storage = classifyParamStorage(param.getType());
            if (failed(storage)) {
                // The forward pass reports this through `verifyTypeIsCacheable` and aborts the
                // lowering; no list is needed for a parameter that is never pushed.
                continue;
            }
            if (!paramVectors.contains(storage->listElementType)) {
                auto paramVector = ListInitOp::create(
                    builder, loc, ArrayListType::get(ctx, storage->listElementType));
                paramVectors.insert({storage->listElementType, paramVector});
            }
            hasDynamicallyShapedParam |= storage->hasDynamicShape;
        }
    });

    TypedValue<ArrayListType> shapeVector;
    if (hasDynamicallyShapedParam) {
        auto shapeList = ListInitOp::create(builder, loc, controlFlowTapeType);
        shapeVector = shapeList;
    }

    // Initialize the tapes that store the structure of control flow.
    DenseMap<Operation *, TypedValue<ArrayListType>> controlFlowTapes;
    region.walk([&](Operation *op) {
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp>(op)) {
            auto tape = catalyst::ListInitOp::create(builder, loc, controlFlowTapeType);
            controlFlowTapes.insert({op, tape});
        }
    });
    return quantum::QuantumCache{.paramVectors = paramVectors,
                                 .shapeVector = shapeVector,
                                 .wireVector = wireVector,
                                 .controlFlowTapes = controlFlowTapes};
}

void QuantumCache::pushParam(Value param, OpBuilder &builder, Location loc) {
    FailureOr<ParamStorageInfo> storage = classifyParamStorage(param.getType());
    assert(succeeded(storage) &&
           "parameter type should have been checked by verifyTypeIsCacheable");
    TypedValue<ArrayListType> paramVector = paramVectors.lookup(storage->listElementType);
    assert(paramVector && "the cache holds no list for this parameter type");

    if (storage->kind == ParamStorage::Scalar) {
        // Either a plain scalar or a rank-0 tensor holding a single element.
        Value element = param;
        if (isa<RankedTensorType>(param.getType())) {
            element = tensor::ExtractOp::create(builder, loc, param, ValueRange{});
        }
        ListPushOp::create(builder, loc, element, paramVector);
        return;
    }

    auto tensorType = cast<RankedTensorType>(param.getType());

    // Record the dynamic dimensions so that the reverse pass can rebuild the shape without
    // referring to any value of the region being replaced. These go to `shapeVector`, a list
    // distinct from every `paramVectors` entry, so the two sequences never interleave.
    if (storage->hasDynamicShape) {
        assert(shapeVector && "the cache holds no list for dynamic dimensions");
        for (int64_t dim = 0; dim < tensorType.getRank(); ++dim) {
            if (tensorType.isDynamicDim(dim)) {
                Value dimSize = tensor::DimOp::create(builder, loc, param, dim);
                ListPushOp::create(builder, loc, dimSize, shapeVector);
            }
        }
    }

    // Cache all the elements in ascending row-major order as a single block.
    ListPushBlockOp::create(builder, loc, param, paramVector);
}

Value QuantumCache::popParam(Type paramType, OpBuilder &builder, Location loc) {
    FailureOr<ParamStorageInfo> storage = classifyParamStorage(paramType);
    assert(succeeded(storage) &&
           "parameter type should have been checked by verifyTypeIsCacheable");
    TypedValue<ArrayListType> paramVector = paramVectors.lookup(storage->listElementType);
    assert(paramVector && "the cache holds no list for this parameter type");

    if (storage->kind == ParamStorage::Scalar) {
        Value element = ListPopOp::create(builder, loc, paramVector);
        if (auto tensorType = dyn_cast<RankedTensorType>(paramType)) {
            // A rank-0 tensor: wrap the single element back up.
            return tensor::FromElementsOp::create(builder, loc, tensorType, element);
        }
        return element;
    }

    auto tensorType = cast<RankedTensorType>(paramType);

    // Dimensions were pushed in ascending order, so pop them in descending order.
    SmallVector<Value> dimSizes(tensorType.getRank());
    for (int64_t dim = tensorType.getRank() - 1; dim >= 0; --dim) {
        if (tensorType.isDynamicDim(dim)) {
            assert(shapeVector && "the cache holds no list for dynamic dimensions");
            dimSizes[dim] = ListPopOp::create(builder, loc, shapeVector);
        }
    }
    SmallVector<Value> dynamicSizes;
    for (int64_t dim = 0; dim < tensorType.getRank(); ++dim) {
        if (tensorType.isDynamicDim(dim)) {
            dynamicSizes.push_back(dimSizes[dim]);
        }
    }

    // The whole tensor was cached as one block: pop it into a fresh tensor of the recorded shape.
    Value destination = tensor::EmptyOp::create(builder, loc, tensorType, dynamicSizes);
    return ListPopBlockOp::create(builder, loc, /*resultTypes=*/TypeRange{tensorType},
                                  /*operands=*/ValueRange{paramVector, destination})
        .getResult();
}

void QuantumCache::emitDealloc(OpBuilder &builder, Location loc) {
    for (const auto &[_elementType, paramVector] : paramVectors) {
        ListDeallocOp::create(builder, loc, paramVector);
    }
    if (shapeVector) {
        ListDeallocOp::create(builder, loc, shapeVector);
    }
    ListDeallocOp::create(builder, loc, wireVector);
    for (const auto &[_key, controlFlowTape] : controlFlowTapes) {
        ListDeallocOp::create(builder, loc, controlFlowTape);
    }
}

} // namespace quantum
} // namespace catalyst
