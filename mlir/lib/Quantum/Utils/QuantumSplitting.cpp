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

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/QuantumSplitting.h"

using namespace mlir;
using namespace catalyst;

namespace {
bool isQuantumType(Type type) { return isa<quantum::QuantumDialect>(type.getDialect()); }

void populateArgIdxMapping(TypeRange types, DenseMap<unsigned, unsigned> &argIdxMapping)
{
    unsigned newIdx = 0;
    for (const auto &[oldIdx, type] : llvm::enumerate(types)) {
        if (!isQuantumType(type)) {
            argIdxMapping.insert({oldIdx, newIdx++});
        }
    }
}

} // namespace

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
    auto paramVector = builder.create<ListInitOp>(loc, paramVectorType);
    auto wireVector = builder.create<ListInitOp>(loc, wireVectorType);

    // Initialize the tapes that store the structure of control flow.
    DenseMap<Operation *, TypedValue<ArrayListType>> controlFlowTapes;
    region.walk([&](Operation *op) {
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
            auto tape = builder.create<catalyst::ListInitOp>(loc, controlFlowTapeType);
            controlFlowTapes.insert({op, tape});
        }
    });
    return quantum::QuantumCache{
        .paramVector = paramVector, .wireVector = wireVector, .controlFlowTapes = controlFlowTapes};
}

void QuantumCache::emitDealloc(OpBuilder &builder, Location loc)
{
    builder.create<ListDeallocOp>(loc, paramVector);
    builder.create<ListDeallocOp>(loc, wireVector);
    for (const auto &[_key, controlFlowTape] : controlFlowTapes) {
        builder.create<ListDeallocOp>(loc, controlFlowTape);
    }
}

void AugmentedCircuitGenerator::cacheGate(quantum::ParametrizedGate gate, OpBuilder &builder)
{
    ValueRange params = gate.getAllParams();

    for (Value param : params) {
        Location loc = gate.getLoc();
        Value clonedParam = oldToCloned.lookupOrDefault(param);
        Type paramType = clonedParam.getType();
        Operation *op = gate;
        verifyTypeIsCacheable(paramType, op);

        if (paramType.isF64()) {
            builder.create<ListPushOp>(loc, clonedParam, cache.paramVector);
            continue;
        }

        // Sanitizing inputs.
        // Technically we know for a fact that none of this will ever issue an error.
        // This is because QubitUnitary is guaranteed to have a tensor<NxNxcomplex<f64>>
        // But this code in the future may be extended to support other types.
        // Hence the sanitization.
        if (!isa<RankedTensorType>(paramType)) {
            gate.emitOpError() << "Unexpected type.";
        }

        auto aTensor = cast<RankedTensorType>(paramType);
        ArrayRef<int64_t> shape = aTensor.getShape();
        Value c0 = builder.create<index::ConstantOp>(loc, 0);
        Value c1 = builder.create<index::ConstantOp>(loc, 1);
        bool isDim0Static = ShapedType::kDynamic != shape[0];
        bool isDim1Static = ShapedType::kDynamic != shape[1];
        Value dim0Length = isDim0Static ? (Value)builder.create<index::ConstantOp>(loc, shape[0])
                                        : (Value)builder.create<tensor::DimOp>(loc, param, c0);
        Value dim1Length = isDim1Static ? (Value)builder.create<index::ConstantOp>(loc, shape[1])
                                        : (Value)builder.create<tensor::DimOp>(loc, param, c1);

        Value lowerBoundDim0 = c0;
        Value upperBoundDim0 = dim0Length;
        Value stepDim0 = c1;
        Value lowerBoundDim1 = c0;
        Value upperBoundDim1 = dim1Length;
        Value stepDim1 = c1;
        Value matrix = clonedParam;

        scf::ForOp iForLoop =
            builder.create<scf::ForOp>(loc, lowerBoundDim0, upperBoundDim0, stepDim0);
        {
            OpBuilder::InsertionGuard afterIForLoop(builder);
            builder.setInsertionPointToStart(iForLoop.getBody());
            Value i_index = iForLoop.getInductionVar();

            scf::ForOp jForLoop =
                builder.create<scf::ForOp>(loc, lowerBoundDim1, upperBoundDim1, stepDim1);
            {
                OpBuilder::InsertionGuard afterJForLoop(builder);
                builder.setInsertionPointToStart(jForLoop.getBody());
                Value j_index = jForLoop.getInductionVar();
                SmallVector<Value> indices = {i_index, j_index};
                Value element = builder.create<tensor::ExtractOp>(loc, matrix, indices);
                // element is complex!
                // So we need to convert into {f64, f64}
                Value real = builder.create<complex::ReOp>(loc, element);
                Value imag = builder.create<complex::ImOp>(loc, element);
                // Again, take note of the order.
                builder.create<ListPushOp>(loc, real, cache.paramVector);
                builder.create<ListPushOp>(loc, imag, cache.paramVector);
            }
        }
    }
}

void AugmentedCircuitGenerator::generate(Region &region, OpBuilder &builder)
{
    assert(region.hasOneBlock() &&
           "Expected only structured control flow (each region should have a single block)");
    auto isClassicalSCFOp = [](Operation &op) {
        return isa<scf::SCFDialect>(op.getDialect()) &&
               llvm::none_of(op.getResultTypes(), isQuantumType);
    };

    for (Operation &op : region.front().without_terminator()) {
        if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
            cacheDynamicWire(insertOp, builder);
        }
        else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
            cacheDynamicWire(extractOp, builder);
        }
        else if (auto gate = dyn_cast<quantum::ParametrizedGate>(op)) {
            cacheGate(gate, builder);
        }
        else if (isa<QuantumDialect>(op.getDialect())) {
            // Any quantum op other than a parametrized gate/insert/extract is ignored.
        }
        else if (isClassicalSCFOp(op)) {
            // Purely classical SCF ops should be treated as any other purely classical op, but
            // quantum SCF ops need to be recursively visited.
            builder.clone(op, oldToCloned);
        }
        else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            visitOperation(forOp, builder);
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            visitOperation(ifOp, builder);
        }
        else if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
            visitOperation(whileOp, builder);
        }
        else if (auto callOp = dyn_cast<func::CallOp>(op)) {
            auto results = callOp.getResultTypes();

            bool multiReturns = results.size() > 1;

            bool quantum = std::any_of(results.begin(), results.end(),
                                       [](const auto &value) { return isa<QuregType>(value); });

            // Classical call operations are cloned for the backward pass
            if (!quantum) {
                builder.clone(op, oldToCloned);
            }

            assert(!(quantum && multiReturns) && "Adjoint does not support functions with multiple "
                                                 "returns that contain a quantum register.");
        }
        else {
            // Purely classical ops are deeply cloned as-is.
            builder.clone(op, oldToCloned);
        }
    }
}

void AugmentedCircuitGenerator::visitOperation(scf::ForOp forOp, OpBuilder &builder)
{
    DenseMap<unsigned, unsigned> argIdxMapping;
    SmallVector<Value> classicalInits;
    populateArgIdxMapping(forOp.getResultTypes(), argIdxMapping);
    unsigned newIdx = 0;
    for (const auto &[oldIdx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (!isQuantumType(initArg.getType())) {
            argIdxMapping.insert({oldIdx, newIdx++});
            classicalInits.push_back(oldToCloned.lookupOrDefault(initArg));
        }
    }

    // Store the start, stop, and step to this op's control flow tape.
    Value tape = cache.controlFlowTapes.at(forOp);
    for (Value param : {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()}) {
        builder.create<ListPushOp>(forOp.getLoc(), oldToCloned.lookupOrDefault(param), tape);
    }

    auto newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(), oldToCloned.lookupOrDefault(forOp.getLowerBound()),
        oldToCloned.lookupOrDefault(forOp.getUpperBound()),
        oldToCloned.lookupOrDefault(forOp.getStep()), classicalInits,
        [&](OpBuilder &builder, Location loc, Value inductionVar, ValueRange iterArgs) {
            oldToCloned.map(forOp.getInductionVar(), inductionVar);
            for (const auto &[oldIdx, newIdx] : argIdxMapping) {
                oldToCloned.map(forOp.getRegionIterArg(oldIdx), iterArgs[newIdx]);
            }

            generate(forOp.getRegion(), builder);
            cloneTerminatorClassicalOperands(forOp.getBody()->getTerminator(), builder);
        });

    mapResults(forOp, newForOp, argIdxMapping);
}

void AugmentedCircuitGenerator::visitOperation(scf::WhileOp whileOp, OpBuilder &builder)
{
    SmallVector<Type> classicalResultTypes;
    SmallVector<Value> classicalInits;
    DenseMap<unsigned, unsigned> argIdxMapping;
    populateArgIdxMapping(whileOp.getResultTypes(), argIdxMapping);
    unsigned newIdx = 0;
    for (const auto &[oldIdx, init] : llvm::enumerate(whileOp.getInits())) {
        if (!isQuantumType(init.getType())) {
            classicalInits.push_back(oldToCloned.lookupOrDefault(init));
            classicalResultTypes.push_back(init.getType());
            argIdxMapping.insert({oldIdx, newIdx++});
        }
    }

    // Augment the classical loop by counting the number of iterations.
    auto counterType = MemRefType::get({}, builder.getIndexType());
    Location loc = whileOp.getLoc();
    Value idx0 = builder.create<index::ConstantOp>(loc, 0);
    Value idx1 = builder.create<index::ConstantOp>(loc, 1);
    Value counter = builder.create<memref::AllocaOp>(loc, counterType);
    builder.create<memref::StoreOp>(loc, idx0, counter);

    auto getRegionBuilder = [&](Region &oldRegion, bool incrementCounter) {
        return [&, incrementCounter](OpBuilder &builder, Location loc, ValueRange newRegionArgs) {
            for (const auto &[oldIdx, newIdx] : argIdxMapping) {
                oldToCloned.map(oldRegion.getArgument(oldIdx), newRegionArgs[newIdx]);
            }

            if (incrementCounter) {
                Value countVal = builder.create<memref::LoadOp>(loc, counter);
                countVal = builder.create<index::AddOp>(loc, countVal, idx1);
                builder.create<memref::StoreOp>(loc, countVal, counter);
            }

            // Recursively clone the region
            generate(oldRegion, builder);
            cloneTerminatorClassicalOperands(oldRegion.front().getTerminator(), builder);
        };
    };

    auto newWhileOp = builder.create<scf::WhileOp>(
        whileOp.getLoc(), classicalResultTypes, classicalInits,
        getRegionBuilder(whileOp.getBefore(), /*incrementCounter=*/false),
        // We only care about the number of times the "After" region executes. The frontend
        // does not support putting quantum operations in the "Before" region, which only
        // computes the iteration condition.
        getRegionBuilder(whileOp.getAfter(), /*incrementCounter=*/true));

    mapResults(whileOp, newWhileOp, argIdxMapping);

    Value numIters = builder.create<memref::LoadOp>(whileOp.getLoc(), counter);
    Value tape = cache.controlFlowTapes.at(whileOp);
    builder.create<ListPushOp>(whileOp.getLoc(), numIters, tape);
}

void AugmentedCircuitGenerator::visitOperation(scf::IfOp ifOp, OpBuilder &builder)
{
    auto getRegionBuilder = [&](Region &oldRegion) {
        return [&](OpBuilder &builder, Location loc) {
            generate(oldRegion, builder);
            cloneTerminatorClassicalOperands(oldRegion.front().getTerminator(), builder);
        };
    };
    DenseMap<unsigned, unsigned> argIdxMapping;
    populateArgIdxMapping(ifOp.getResultTypes(), argIdxMapping);

    // Store the condition to this op's control flow tape
    Value condition = oldToCloned.lookupOrDefault(ifOp.getCondition());
    Value tape = cache.controlFlowTapes.at(ifOp);
    Value castedCondition =
        builder.create<index::CastSOp>(ifOp.getLoc(), builder.getIndexType(), condition);
    builder.create<ListPushOp>(ifOp.getLoc(), castedCondition, tape);

    auto newIfOp =
        builder.create<scf::IfOp>(ifOp.getLoc(), condition, getRegionBuilder(ifOp.getThenRegion()),
                                  getRegionBuilder(ifOp.getElseRegion()));

    mapResults(ifOp, newIfOp, argIdxMapping);
}

void AugmentedCircuitGenerator::cloneTerminatorClassicalOperands(Operation *terminator,
                                                                 OpBuilder &builder)
{
    SmallVector<Value> newYieldOperands;
    for (Value operand : terminator->getOperands()) {
        if (!isQuantumType(operand.getType())) {
            newYieldOperands.push_back(oldToCloned.lookupOrDefault(operand));
        }
    }
    Operation *newTerminator = builder.clone(*terminator, oldToCloned);
    newTerminator->setOperands(newYieldOperands);
}

void AugmentedCircuitGenerator::mapResults(Operation *oldOp, Operation *clonedOp,
                                           const DenseMap<unsigned, unsigned> &argIdxMapping)
{
    for (const auto &[oldIdx, oldResult] : llvm::enumerate(oldOp->getResults())) {
        if (argIdxMapping.contains(oldIdx)) {
            unsigned newIdx = argIdxMapping.at(oldIdx);
            oldToCloned.map(oldOp->getResult(oldIdx), clonedOp->getResult(newIdx));
        }
    }
}

} // namespace quantum
} // namespace catalyst
