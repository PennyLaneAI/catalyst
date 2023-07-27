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

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

void AugmentedCircuitGenerator::generate(Region &region)
{
    assert(region.hasOneBlock() &&
           "Expected only structured control flow (each region should have a single block)");
    auto isClassicalSCFOp = [](Operation &op) {
        return isa<scf::SCFDialect>(op.getDialect()) &&
               llvm::none_of(op.getResultTypes(), isQuantumType);
    };

    for (Operation &op : region.front().without_terminator()) {
        if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
            cacheDynamicWire(insertOp);
        }
        else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
            cacheDynamicWire(extractOp);
        }
        else if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
            ValueRange diffParams = gate.getDiffParams();
            if (!diffParams.empty()) {
                for (Value param : diffParams) {
                    rewriter.create<ListPushOp>(gate.getLoc(), oldToCloned.lookupOrDefault(param),
                                                cache.paramVector);
                }
            }
        }
        else if (isa<QuantumDialect>(op.getDialect())) {
            continue;
        }
        else if (isClassicalSCFOp(op)) {
            rewriter.clone(op, oldToCloned);
            continue;
        }
        else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            visitOperation(forOp);
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            visitOperation(ifOp);
        }
        else if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
            visitOperation(whileOp);
        }
        else {
            rewriter.clone(op, oldToCloned);
        }
    }
}

void AugmentedCircuitGenerator::visitOperation(scf::ForOp forOp)
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
        rewriter.create<ListPushOp>(forOp.getLoc(), oldToCloned.lookupOrDefault(param), tape);
    }

    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), oldToCloned.lookupOrDefault(forOp.getLowerBound()),
        oldToCloned.lookupOrDefault(forOp.getUpperBound()),
        oldToCloned.lookupOrDefault(forOp.getStep()), classicalInits,
        [&](OpBuilder &builder, Location loc, Value inductionVar, ValueRange iterArgs) {
            oldToCloned.map(forOp.getInductionVar(), inductionVar);
            for (const auto &[oldIdx, newIdx] : argIdxMapping) {
                oldToCloned.map(forOp.getRegionIterArg(oldIdx), iterArgs[newIdx]);
            }

            PatternRewriter::InsertionGuard insertionGuard(rewriter);
            rewriter.restoreInsertionPoint(builder.saveInsertionPoint());
            generate(forOp.getRegion());
            cloneTerminatorClassicalOperands(forOp.getBody()->getTerminator());
        });

    mapResults(forOp, newForOp, argIdxMapping);
}

void AugmentedCircuitGenerator::visitOperation(scf::WhileOp whileOp)
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
    auto counterType = MemRefType::get({}, rewriter.getIndexType());
    Location loc = whileOp.getLoc();
    Value idx0 = rewriter.create<index::ConstantOp>(loc, 0);
    Value idx1 = rewriter.create<index::ConstantOp>(loc, 1);
    Value counter = rewriter.create<memref::AllocaOp>(loc, counterType);
    rewriter.create<memref::StoreOp>(loc, idx0, counter);

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
            PatternRewriter::InsertionGuard insertionGuard(rewriter);
            rewriter.restoreInsertionPoint(builder.saveInsertionPoint());
            generate(oldRegion);
            cloneTerminatorClassicalOperands(oldRegion.front().getTerminator());
        };
    };

    auto newWhileOp = rewriter.create<scf::WhileOp>(
        whileOp.getLoc(), classicalResultTypes, classicalInits,
        getRegionBuilder(whileOp.getBefore(), /*incrementCounter=*/false),
        // We only care about the number of times the "After" region executes. The frontend
        // does not support putting quantum operations in the "Before" region, which only
        // computes the iteration condition.
        getRegionBuilder(whileOp.getAfter(), /*incrementCounter=*/true));

    mapResults(whileOp, newWhileOp, argIdxMapping);

    Value numIters = rewriter.create<memref::LoadOp>(whileOp.getLoc(), counter);
    Value tape = cache.controlFlowTapes.at(whileOp);
    rewriter.create<ListPushOp>(whileOp.getLoc(), numIters, tape);
}

void AugmentedCircuitGenerator::visitOperation(scf::IfOp ifOp)
{
    auto getRegionBuilder = [&](Region &oldRegion) {
        return [&](OpBuilder &builder, Location loc) {
            PatternRewriter::InsertionGuard insertionGuard(rewriter);
            rewriter.restoreInsertionPoint(builder.saveInsertionPoint());
            generate(oldRegion);
            cloneTerminatorClassicalOperands(oldRegion.front().getTerminator());
        };
    };
    DenseMap<unsigned, unsigned> argIdxMapping;
    populateArgIdxMapping(ifOp.getResultTypes(), argIdxMapping);

    // Store the condition to this op's control flow tape
    Value condition = oldToCloned.lookupOrDefault(ifOp.getCondition());
    Value tape = cache.controlFlowTapes.at(ifOp);
    Value castedCondition =
        rewriter.create<index::CastSOp>(ifOp.getLoc(), rewriter.getIndexType(), condition);
    rewriter.create<ListPushOp>(ifOp.getLoc(), castedCondition, tape);

    auto newIfOp =
        rewriter.create<scf::IfOp>(ifOp.getLoc(), condition, getRegionBuilder(ifOp.getThenRegion()),
                                   getRegionBuilder(ifOp.getElseRegion()));

    mapResults(ifOp, newIfOp, argIdxMapping);
}

void AugmentedCircuitGenerator::cloneTerminatorClassicalOperands(Operation *terminator)
{
    SmallVector<Value> newYieldOperands;
    for (Value operand : terminator->getOperands()) {
        if (!isQuantumType(operand.getType())) {
            newYieldOperands.push_back(oldToCloned.lookupOrDefault(operand));
        }
    }
    Operation *newTerminator = rewriter.clone(*terminator, oldToCloned);
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
