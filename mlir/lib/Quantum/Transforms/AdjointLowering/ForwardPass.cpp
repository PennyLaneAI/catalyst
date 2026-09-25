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

#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumOps.h"

#include "AdjointLowering.hpp"
#include "QuantumCache.hpp"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {
bool isQuantumType(Type type) { return isa<quantum::QuantumDialect>(type.getDialect()); }

void populateArgIdxMapping(TypeRange types, DenseMap<unsigned, unsigned> &argIdxMapping) {
    unsigned newIdx = 0;
    for (const auto &[oldIdx, type] : llvm::enumerate(types)) {
        if (!isQuantumType(type)) {
            argIdxMapping.insert({oldIdx, newIdx++});
        }
    }
}

/// Generates the forward "augmented circuit" of the adjoint operation: classical preprocessing is
/// cloned as-is, while gate parameters, dynamic wires, and control-flow structure are recorded into
/// the cache for the reverse pass to replay.
class AugmentedCircuitGenerator {
  public:
    AugmentedCircuitGenerator(IRMapping &oldToCloned, QuantumCache &cache, Region &adjointRegion)
        : oldToCloned(oldToCloned), cache(cache), adjointRegion(adjointRegion) {}

    /// Given a `region` containing classical preprocessing and quantum operations, generate an
    /// augmented version that caches all the parameters required to deterministically re-execute
    /// the circuit (gate params, classical control flow, and dynamic wires).
    void generate(Region &region, OpBuilder &builder);

    bool hasFailed() const { return generationFailed; }

  private:
    IRMapping &oldToCloned;
    QuantumCache &cache;

    /// The top level region of the adjoint operation being lowered.
    Region &adjointRegion;
    bool generationFailed = false;

    void visitOperation(scf::ForOp forOp, OpBuilder &builder);
    void visitOperation(scf::WhileOp whileOp, OpBuilder &builder);
    void visitOperation(scf::IfOp ifOp, OpBuilder &builder);
    void visitOperation(scf::IndexSwitchOp indexSwitchOp, OpBuilder &builder);

    void cloneTerminatorClassicalOperands(Operation *terminator, OpBuilder &builder);

    /// Update the internal mapping of the results of `oldOp` to the results of `clonedOp` using the
    /// given result remapping.
    void mapResults(Operation *oldOp, Operation *clonedOp,
                    const DenseMap<unsigned, unsigned> &argIdxMapping);

    // Emit an operation to cache a dynamic wire for quantum.insert/extract ops.
    template <typename IndexingOp> void cacheDynamicWire(IndexingOp op, OpBuilder &builder) {
        if (!op.getIdxAttr().has_value()) {
            ListPushOp::create(builder, op.getLoc(), oldToCloned.lookupOrDefault(op.getIdx()),
                               cache.wireVector);
        }
    }

    void cacheGate(quantum::ParametrizedGate gate, OpBuilder &builder);
};

void AugmentedCircuitGenerator::cacheGate(quantum::ParametrizedGate gate, OpBuilder &builder) {
    ValueRange params = gate.getAllParams();

    for (Value param : params) {
        Location loc = gate.getLoc();

        // Params that the reverse pass can already see do not need to be recorded.
        if (isAvailableToReversePass(param, adjointRegion)) {
            continue;
        }

        Value clonedParam = oldToCloned.lookupOrDefault(param);
        Type paramType = clonedParam.getType();
        Operation *op = gate;

        DataLayout dataLayout = DataLayout::closest(op);
        auto zero = arith::ConstantIndexOp::create(builder, loc, 0);

        // 1. Load the current offset index from the currentOffset memref
        Value currentOffsetIndex =
            memref::LoadOp::create(builder, loc, cache.currentOffset, ValueRange{}).getResult();

        // 2. Store the param value into the cache at the current offset
        // The cache is a raw <some_size x i8> byte memref, so need to view it with the
        // param type size
        if (isa<RankedTensorType>(paramType)) {
            // Param is a tensor, need to convert to memrefs via bufferization ops
            auto tensorType = cast<RankedTensorType>(paramType);
            assert(tensorType.hasStaticShape() &&
                   "Dynamically sized tensor params not supported yet");

            MemRefType memrefType =
                MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            auto buffer = bufferization::ToBufferOp::create(builder, loc, memrefType, clonedParam)
                              .getBuffer();
            Value view =
                memref::ViewOp::create(builder, loc, memrefType, cache.paramVector,
                                       currentOffsetIndex, ValueRange{} // Empty dynamic sizes
                                       )
                    ->getResult(0);
            memref::CopyOp::create(builder, loc, buffer, view);
        } else {
            // Param not a tensor, just use a raw memref without buffers
            auto targetViewType = MemRefType::get({1}, paramType);
            Value view =
                memref::ViewOp::create(builder, loc, targetViewType, cache.paramVector,
                                       currentOffsetIndex, ValueRange{} // Empty dynamic sizes
                                       )
                    ->getResult(0);
            memref::StoreOp::create(builder, loc, clonedParam, view, ValueRange{zero});
        }

        // 3. Push the current offset onto the offset stack
        ListPushOp::create(builder, loc, currentOffsetIndex, cache.offsetVector);

        // 4. Increment the current offset with the byte size of this param
        int64_t paramNumBytes = 0;
        if (isa<RankedTensorType>(paramType)) {
            auto tensorType = cast<RankedTensorType>(paramType);
            int64_t elementTypeNumBytes =
                dataLayout.getTypeSize(tensorType.getElementType()).getFixedValue();

            int64_t numElements = 1;
            for (int64_t dim : tensorType.getShape()) {
                numElements *= dim;
            }
            paramNumBytes = numElements * elementTypeNumBytes;
        } else {
            paramNumBytes = dataLayout.getTypeSize(paramType).getFixedValue();
        }

        Value paramNumBytesValue =
            index::ConstantOp::create(builder, loc, paramNumBytes).getResult();
        Value newOffset =
            index::AddOp::create(builder, loc, currentOffsetIndex, paramNumBytesValue).getResult();
        memref::StoreOp::create(builder, loc, newOffset, cache.currentOffset, ValueRange{});
    }
}

void AugmentedCircuitGenerator::generate(Region &region, OpBuilder &builder) {
    assert(region.hasOneBlock() &&
           "Expected only structured control flow (each region should have a single block)");
    auto isClassicalSCFOp = [](Operation &op) {
        return isa<scf::SCFDialect>(op.getDialect()) &&
               llvm::none_of(op.getResultTypes(), isQuantumType);
    };

    for (Operation &op : region.front().without_terminator()) {
        if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
            cacheDynamicWire(insertOp, builder);
        } else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
            cacheDynamicWire(extractOp, builder);
        } else if (auto gate = dyn_cast<quantum::ParametrizedGate>(op)) {
            cacheGate(gate, builder);
        } else if (isa<QuantumDialect>(op.getDialect())) {
            // Any quantum op other than a parametrized gate/insert/extract is ignored.
        } else if (isa<pbc::PPRotationOp>(op)) {
            // PPRs are ignored
        } else if (isClassicalSCFOp(op)) {
            // Purely classical SCF ops should be treated as any other purely classical op, but
            // quantum SCF ops need to be recursively visited.
            builder.clone(op, oldToCloned);
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            visitOperation(forOp, builder);
        } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            visitOperation(ifOp, builder);
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
            visitOperation(whileOp, builder);
        } else if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
            visitOperation(switchOp, builder);
        } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
            auto results = callOp.getResultTypes();
            bool quantum = std::any_of(results.begin(), results.end(), [](const auto &value) {
                return isa<QuregType, QubitType>(value);
            });

            // Classical call operations are cloned for the backward pass
            if (!quantum) {
                builder.clone(op, oldToCloned);
            }
        } else {
            // Purely classical ops are deeply cloned as-is.
            builder.clone(op, oldToCloned);
        }
    }
}

void AugmentedCircuitGenerator::visitOperation(scf::ForOp forOp, OpBuilder &builder) {
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

    // Store the start, stop, and step to this op's control flow tape, but only when dynamic.
    // Constant values can be rematerialized directly in the backward pass, which preverses the
    // static info.
    Value tape = cache.controlFlowTapes.at(forOp);
    for (Value param : {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()}) {
        if (getConstantIntValue(param).has_value()) {
            continue;
        }
        ListPushOp::create(builder, forOp.getLoc(), oldToCloned.lookupOrDefault(param), tape);
    }

    auto newForOp = scf::ForOp::create(
        builder, forOp.getLoc(), oldToCloned.lookupOrDefault(forOp.getLowerBound()),
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

void AugmentedCircuitGenerator::visitOperation(scf::WhileOp whileOp, OpBuilder &builder) {
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
    Value idx0 = index::ConstantOp::create(builder, loc, 0);
    Value idx1 = index::ConstantOp::create(builder, loc, 1);
    Value counter = memref::AllocaOp::create(builder, loc, counterType);
    memref::StoreOp::create(builder, loc, idx0, counter);

    auto getRegionBuilder = [&](Region &oldRegion, bool incrementCounter) {
        return [&, incrementCounter](OpBuilder &builder, Location loc, ValueRange newRegionArgs) {
            for (const auto &[oldIdx, newIdx] : argIdxMapping) {
                oldToCloned.map(oldRegion.getArgument(oldIdx), newRegionArgs[newIdx]);
            }

            if (incrementCounter) {
                Value countVal = memref::LoadOp::create(builder, loc, counter);
                countVal = index::AddOp::create(builder, loc, countVal, idx1);
                memref::StoreOp::create(builder, loc, countVal, counter);
            }

            // Recursively clone the region
            generate(oldRegion, builder);
            cloneTerminatorClassicalOperands(oldRegion.front().getTerminator(), builder);
        };
    };

    auto newWhileOp =
        scf::WhileOp::create(builder, whileOp.getLoc(), classicalResultTypes, classicalInits,
                             getRegionBuilder(whileOp.getBefore(), /*incrementCounter=*/false),
                             // We only care about the number of times the "After" region executes.
                             // The frontend does not support putting quantum operations in the
                             // "Before" region, which only computes the iteration condition.
                             getRegionBuilder(whileOp.getAfter(), /*incrementCounter=*/true));

    mapResults(whileOp, newWhileOp, argIdxMapping);

    Value numIters = memref::LoadOp::create(builder, whileOp.getLoc(), counter);
    Value tape = cache.controlFlowTapes.at(whileOp);
    ListPushOp::create(builder, whileOp.getLoc(), numIters, tape);
}

void AugmentedCircuitGenerator::visitOperation(scf::IndexSwitchOp switchOp, OpBuilder &builder) {
    auto getRegionBuilder = [&](Region &oldRegion) {
        return [&](OpBuilder &builder, Location loc) {
            generate(oldRegion, builder);
            cloneTerminatorClassicalOperands(oldRegion.front().getTerminator(), builder);
        };
    };
    DenseMap<unsigned, unsigned> argIdxMapping;
    populateArgIdxMapping(switchOp.getResultTypes(), argIdxMapping);

    // Cache the switch index to the current control flow tape
    Value arg = oldToCloned.lookupOrDefault(switchOp.getArg());
    Value tape = cache.controlFlowTapes.at(switchOp.getOperation());
    ListPushOp::create(builder, switchOp.getLoc(), oldToCloned.lookupOrDefault(switchOp.getArg()),
                       tape);

    SmallVector<Type> classicalResultTypes;
    for (Type ty : switchOp.getResultTypes()) {
        if (!isQuantumType(ty)) {
            classicalResultTypes.push_back(ty);
        }
    }

    auto newSwitchOp = scf::IndexSwitchOp::create(builder, switchOp.getLoc(), classicalResultTypes,
                                                  arg, switchOp.getCases(), switchOp.getNumCases());

    // Case and default regions are gotten by different APIs
    // Here we handle them separately.

    // Case regions:
    for (auto [oldCaseRegion, newCaseRegion] :
         llvm::zip_equal(switchOp.getCaseRegions(), newSwitchOp.getCaseRegions())) {
        OpBuilder::InsertionGuard guard(builder);
        newCaseRegion.push_back(new Block());
        builder.setInsertionPointToStart(&newCaseRegion.front());
        getRegionBuilder(oldCaseRegion)(builder, switchOp.getLoc());
    }

    // Default region:
    {
        OpBuilder::InsertionGuard guard(builder);
        newSwitchOp.getDefaultRegion().push_back(new Block());
        builder.setInsertionPointToStart(&newSwitchOp.getDefaultRegion().front());
        getRegionBuilder(switchOp.getDefaultRegion())(builder, switchOp.getLoc());
    }

    mapResults(switchOp, newSwitchOp, argIdxMapping);
}

void AugmentedCircuitGenerator::visitOperation(scf::IfOp ifOp, OpBuilder &builder) {
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
        index::CastSOp::create(builder, ifOp.getLoc(), builder.getIndexType(), condition);
    ListPushOp::create(builder, ifOp.getLoc(), castedCondition, tape);

    auto newIfOp =
        scf::IfOp::create(builder, ifOp.getLoc(), condition, getRegionBuilder(ifOp.getThenRegion()),
                          getRegionBuilder(ifOp.getElseRegion()));

    mapResults(ifOp, newIfOp, argIdxMapping);
}

void AugmentedCircuitGenerator::cloneTerminatorClassicalOperands(Operation *terminator,
                                                                 OpBuilder &builder) {
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
                                           const DenseMap<unsigned, unsigned> &argIdxMapping) {
    for (const auto &[oldIdx, oldResult] : llvm::enumerate(oldOp->getResults())) {
        if (argIdxMapping.contains(oldIdx)) {
            unsigned newIdx = argIdxMapping.at(oldIdx);
            oldToCloned.map(oldOp->getResult(oldIdx), clonedOp->getResult(newIdx));
        }
    }
}

} // namespace

namespace catalyst {
namespace quantum {

LogicalResult generateAdjointForwardPass(Region &region, OpBuilder &builder, IRMapping &oldToCloned,
                                         QuantumCache &cache) {
    AugmentedCircuitGenerator generator{oldToCloned, cache, region};
    generator.generate(region, builder);
    return failure(generator.hasFailed());
}

} // namespace quantum
} // namespace catalyst
