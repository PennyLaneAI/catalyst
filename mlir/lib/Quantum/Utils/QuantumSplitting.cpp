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

quantum::QuantumCache quantum::QuantumCache::initialize(Region &region, OpBuilder &builder,
                                                        Location loc)
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

void quantum::cloneClassical(Region &region, IRMapping &oldToCloned, PatternRewriter &rewriter,
                             QuantumCache &cache)
{
    assert(region.hasOneBlock() &&
           "Expected only structured control flow (each region should have a single block)");
    auto isQuantumType = [](Type type) { return isa<QuantumDialect>(type.getDialect()); };
    auto hasQuantumType = [&isQuantumType](Value value) { return isQuantumType(value.getType()); };

    for (Operation &op : region.front().without_terminator()) {
        auto isQuantumSCFOp = [](Operation *op) {
            return llvm::any_of(op->getResultTypes(), [](Type resultType) {
                return isa<quantum::QuregType>(resultType);
            });
        };
        // Cache dynamic wires
        if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
            if (!insertOp.getIdxAttr().has_value()) {
                rewriter.create<ListPushOp>(insertOp.getLoc(),
                                            oldToCloned.lookupOrDefault(insertOp.getIdx()),
                                            cache.wireVector);
            }
        }
        if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
            if (!extractOp.getIdxAttr().has_value()) {
                rewriter.create<ListPushOp>(extractOp.getLoc(),
                                            oldToCloned.lookupOrDefault(extractOp.getIdx()),
                                            cache.wireVector);
            }
        }
        else if (auto gate = dyn_cast<quantum::DifferentiableGate>(op)) {
            ValueRange diffParams = gate.getDiffParams();
            if (!diffParams.empty()) {
                for (Value param : diffParams) {
                    rewriter.create<catalyst::ListPushOp>(
                        gate.getLoc(), oldToCloned.lookupOrDefault(param), cache.paramVector);
                }
            }
        }
        else if (isa<QuantumDialect>(op.getDialect())) {
            continue;
        }
        // TODO: Reduce code duplication with different structured control flow constructs
        else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            if (!isQuantumSCFOp(forOp)) {
                auto clonedFor = cast<scf::ForOp>(rewriter.clone(*forOp, oldToCloned));
                oldToCloned.map(forOp.getResults(), clonedFor.getResults());
                continue;
            }
            DenseMap<unsigned, unsigned> argIdxMapping;
            SmallVector<Value> classicalInits;
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
                rewriter.create<ListPushOp>(forOp.getLoc(), oldToCloned.lookupOrDefault(param),
                                            tape);
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

                    ConversionPatternRewriter rewriter(builder.getContext());
                    rewriter.restoreInsertionPoint(builder.saveInsertionPoint());
                    cloneClassical(forOp.getRegion(), oldToCloned, rewriter, cache);

                    // Clone the classical operands of the terminator.
                    Operation *terminator = forOp.getRegion().front().getTerminator();
                    SmallVector<Value> newYieldOperands;
                    for (Value operand : terminator->getOperands()) {
                        if (!hasQuantumType(operand)) {
                            newYieldOperands.push_back(oldToCloned.lookupOrDefault(operand));
                        }
                    }
                    Operation *newTerminator = builder.clone(*terminator, oldToCloned);
                    newTerminator->setOperands(newYieldOperands);
                });
            // Replace uses of the old for loop's results
            for (const auto &[oldIdx, oldResult] : llvm::enumerate(forOp.getResults())) {
                if (argIdxMapping.contains(oldIdx)) {
                    unsigned newIdx = argIdxMapping.at(oldIdx);
                    oldToCloned.map(forOp.getResult(oldIdx), newForOp.getResult(newIdx));
                }
            }
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            if (!isQuantumSCFOp(ifOp)) {
                auto clonedOp = rewriter.clone(*ifOp, oldToCloned);
                oldToCloned.map(ifOp.getResults(), clonedOp->getResults());
                continue;
            }
            auto getRegionBuilder = [&](Region &oldRegion) {
                return [&](OpBuilder &builder, Location loc) {
                    // Recursively clone the region
                    ConversionPatternRewriter rewriter(builder.getContext());
                    rewriter.restoreInsertionPoint(builder.saveInsertionPoint());
                    cloneClassical(oldRegion, oldToCloned, rewriter, cache);

                    // Clone the classical operands of the terminator.
                    Operation *terminator = oldRegion.front().getTerminator();
                    SmallVector<Value> newYieldOperands;
                    for (Value operand : terminator->getOperands()) {
                        if (!hasQuantumType(operand)) {
                            newYieldOperands.push_back(oldToCloned.lookupOrDefault(operand));
                        }
                    }
                    Operation *newTerminator = rewriter.clone(*terminator, oldToCloned);
                    newTerminator->setOperands(newYieldOperands);
                };
            };
            DenseMap<unsigned, unsigned> argIdxMapping;
            unsigned newIdx = 0;
            for (const auto &[oldIdx, resultType] : llvm::enumerate(ifOp.getResultTypes())) {
                if (!isQuantumType(resultType)) {
                    argIdxMapping.insert({oldIdx, newIdx++});
                }
            }

            // Store the condition to this op's control flow tape
            Value condition = oldToCloned.lookupOrDefault(ifOp.getCondition());
            Value tape = cache.controlFlowTapes.at(ifOp);
            Value castedCondition =
                rewriter.create<index::CastSOp>(ifOp.getLoc(), rewriter.getIndexType(), condition);
            rewriter.create<ListPushOp>(ifOp.getLoc(), castedCondition, tape);

            auto newIfOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), condition,
                                                      getRegionBuilder(ifOp.getThenRegion()),
                                                      getRegionBuilder(ifOp.getElseRegion()));

            // Replace uses of the old if op's results
            for (const auto &[oldIdx, oldResult] : llvm::enumerate(ifOp.getResults())) {
                if (argIdxMapping.contains(oldIdx)) {
                    unsigned newIdx = argIdxMapping.at(oldIdx);
                    oldToCloned.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
                }
            }
        }
        else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
            if (!isQuantumSCFOp(whileOp)) {
                auto clonedOp = rewriter.clone(*whileOp, oldToCloned);
                oldToCloned.map(whileOp.getResults(), clonedOp->getResults());
                continue;
            }
            SmallVector<Type> classicalResultTypes;
            SmallVector<Value> classicalInits;
            DenseMap<unsigned, unsigned> argIdxMapping;
            unsigned newIdx = 0;
            for (const auto &[oldIdx, init] : llvm::enumerate(whileOp.getInits())) {
                if (!hasQuantumType(init)) {
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
                return [&, incrementCounter](OpBuilder &builder, Location loc,
                                             ValueRange newRegionArgs) {
                    for (const auto &[oldIdx, newIdx] : argIdxMapping) {
                        oldToCloned.map(oldRegion.getArgument(oldIdx), newRegionArgs[newIdx]);
                    }

                    if (incrementCounter) {
                        Value countVal = builder.create<memref::LoadOp>(loc, counter);
                        countVal = builder.create<index::AddOp>(loc, countVal, idx1);
                        builder.create<memref::StoreOp>(loc, countVal, counter);
                    }

                    // Recursively clone the region
                    ConversionPatternRewriter rewriter(builder.getContext());
                    rewriter.restoreInsertionPoint(builder.saveInsertionPoint());
                    cloneClassical(oldRegion, oldToCloned, rewriter, cache);

                    // Clone the classical operands of the terminator.
                    Operation *terminator = oldRegion.front().getTerminator();
                    SmallVector<Value> newYieldOperands;
                    for (Value operand : terminator->getOperands()) {
                        if (!hasQuantumType(operand)) {
                            newYieldOperands.push_back(oldToCloned.lookupOrDefault(operand));
                        }
                    }
                    Operation *newTerminator = builder.clone(*terminator, oldToCloned);
                    newTerminator->setOperands(newYieldOperands);
                };
            };

            // We only care about the number of times the "After" region executes. The frontend does
            // not support putting quantum operations in the "Before" region, which only computes
            // the iteration condition.
            auto newWhileOp = rewriter.create<scf::WhileOp>(
                whileOp.getLoc(), classicalResultTypes, classicalInits,
                getRegionBuilder(whileOp.getBefore(), /*incrementCounter=*/false),
                getRegionBuilder(whileOp.getAfter(), /*incrementCounter=*/true));

            // Replace uses of the old while loop's results
            for (const auto &[oldIdx, oldResult] : llvm::enumerate(whileOp.getResults())) {
                if (argIdxMapping.contains(oldIdx)) {
                    unsigned newIdx = argIdxMapping.at(oldIdx);
                    rewriter.replaceAllUsesWith(whileOp.getResult(oldIdx),
                                                newWhileOp.getResult(newIdx));
                }
            }

            Value numIters = rewriter.create<memref::LoadOp>(whileOp.getLoc(), counter);
            Value tape = cache.controlFlowTapes.at(whileOp);
            rewriter.create<catalyst::ListPushOp>(whileOp.getLoc(), numIters, tape);
        }
        else {
            rewriter.clone(op, oldToCloned);
        }
    }
}
