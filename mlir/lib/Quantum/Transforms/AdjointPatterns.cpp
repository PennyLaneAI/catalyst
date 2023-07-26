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

#define DEBUG_TYPE "adjoint"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

/// Clone the region of the adjoint operation `op` to the insertion point specified by the
/// `builder`. Build and return the value mapping `mapping`.
Value cloneAdjointRegion(AdjointOp op, OpBuilder &builder, IRMapping &mapping)
{
    Block &block = op.getRegion().front();
    for (Operation &op : block.without_terminator()) {
        builder.clone(op, mapping);
    }
    auto yieldOp = cast<quantum::YieldOp>(block.getTerminator());
    return mapping.lookupOrDefault(yieldOp.getOperand(0));
}

/// Recursively clone the given region, removing all quantum ops.
void cloneOnlyClassical(IRMapping &oldToCloned, Region &region, PatternRewriter &rewriter,
                        Value paramVector,
                        DenseMap<Operation *, TypedValue<ArrayListType>> &controlFlowTapes)
{
    assert(region.hasOneBlock() &&
           "Expected only structured control flow (each region should have a single block)");
    auto isQuantumType = [](Type type) { return isa<QuantumDialect>(type.getDialect()); };
    auto hasQuantumType = [&isQuantumType](Value value) { return isQuantumType(value.getType()); };

    for (Operation &op : region.front().without_terminator()) {
        if (auto gate = dyn_cast<catalyst::quantum::DifferentiableGate>(op)) {
            ValueRange diffParams = gate.getDiffParams();
            if (!diffParams.empty()) {
                for (Value param : diffParams) {
                    rewriter.create<catalyst::ListPushOp>(
                        gate.getLoc(), oldToCloned.lookupOrDefault(param), paramVector);
                }
            }
        }
        else if (isa<QuantumDialect>(op.getDialect())) {
            continue;
        }
        else if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
            SmallVector<Type> classicalResultTypes;
            SmallVector<Value> classicalInits;
            DenseMap<unsigned, unsigned> argIdxMapping;
            IRMapping quantumOldToCloned;
            unsigned newIdx = 0;
            for (const auto &[oldIdx, init] : llvm::enumerate(whileOp.getInits())) {
                if (hasQuantumType(init)) {
                    quantumOldToCloned.map(whileOp.getResult(oldIdx), init);
                }
                else {
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
                    cloneOnlyClassical(oldToCloned, oldRegion, rewriter, paramVector,
                                       controlFlowTapes);

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
            Value tape = controlFlowTapes.at(whileOp);
            rewriter.create<catalyst::ListPushOp>(whileOp.getLoc(), numIters, tape);
        }
        else {
            rewriter.clone(op, oldToCloned);
        }
    }
}

/// Generate the quantum "backwards pass" of the adjoint operation using the stored gate parameters
/// and cached control flow.
void generateReversedQuantum(IRMapping &oldToCloned, Region &region, OpBuilder &builder,
                             Value paramVector,
                             DenseMap<Operation *, TypedValue<ArrayListType>> &controlFlowTapes)
{
    assert(region.hasOneBlock() &&
           "Expected only structured control flow (each region should have a single block)");
    for (Operation &op : llvm::reverse(region.front().without_terminator())) {
        if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
            Value tape = controlFlowTapes.at(whileOp);
            Value numIterations = builder.create<ListPopOp>(whileOp.getLoc(), tape);
            Value c0 = builder.create<index::ConstantOp>(whileOp.getLoc(), 0);
            Value c1 = builder.create<index::ConstantOp>(whileOp.getLoc(), 1);
            auto getQuantumReg = [](ValueRange values) -> Value {
                for (Value value : values) {
                    if (isa<quantum::QuregType>(value.getType())) {
                        return value;
                    }
                }
                assert(false &&
                       "getResultQuantumReg called on a value range that did not contain a "
                       "quantum register");
            };

            Value iterArgInit = oldToCloned.lookup(getQuantumReg(whileOp.getResults()));
            auto replacedWhile = builder.create<scf::ForOp>(
                whileOp.getLoc(), /*start=*/c0, /*stop=*/numIterations, /*step=*/c1, iterArgInit,
                /*bodyBuilder=*/
                [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
                    Value yieldedQureg =
                        getQuantumReg(whileOp.getAfter().front().getTerminator()->getOperands());
                    oldToCloned.map(yieldedQureg, iterArgs[0]);
                    generateReversedQuantum(oldToCloned, whileOp.getAfter(), builder, paramVector,
                                            controlFlowTapes);
                    builder.create<scf::YieldOp>(
                        loc, oldToCloned.lookup(
                                 getQuantumReg(whileOp.getAfter().front().getArguments())));
                });
            oldToCloned.map(getQuantumReg(whileOp.getInits()), replacedWhile.getResult(0));
        }
        else if (auto insertOp = dyn_cast<quantum::InsertOp>(&op)) {
            auto extractOp = builder.create<quantum::ExtractOp>(
                insertOp.getLoc(), insertOp.getQubit().getType(),
                oldToCloned.lookup(insertOp.getOutQreg()),
                oldToCloned.lookupOrDefault(insertOp.getIdx()), insertOp.getIdxAttrAttr());
            oldToCloned.map(insertOp.getQubit(), extractOp.getResult());
            oldToCloned.map(insertOp.getInQreg(), oldToCloned.lookup(insertOp.getOutQreg()));
        }
        else if (QuantumGate gate = dyn_cast<quantum::QuantumGate>(&op)) {
            for (const auto &[qubitResult, qubitOperand] :
                 llvm::zip(gate.getQubitResults(), gate.getQubitOperands())) {
                oldToCloned.map(qubitOperand, oldToCloned.lookup(qubitResult));
            }

            auto clone = cast<QuantumGate>(builder.clone(*gate, oldToCloned));
            clone.setAdjointFlag(!gate.getAdjointFlag());

            // Read cached differentiable parameters from the recorded parameter vector.
            if (auto differentiableGate = dyn_cast<quantum::DifferentiableGate>(&op)) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.setInsertionPoint(clone);
                SmallVector<Value> cachedParams;
                ValueRange diffParams = differentiableGate.getDiffParams();
                for (unsigned i = 0; i < diffParams.size(); i++) {
                    cachedParams.push_back(
                        builder.create<ListPopOp>(differentiableGate.getLoc(), paramVector));
                }
                MutableOperandRange(clone, differentiableGate.getDiffOperandIdx(),
                                    diffParams.size())
                    .assign(cachedParams);
            }

            for (const auto &[qubitResult, qubitOperand] :
                 llvm::zip(clone.getQubitResults(), gate.getQubitOperands())) {
                oldToCloned.map(qubitOperand, qubitResult);
            }
        }
        else if (auto extractOp = dyn_cast<quantum::ExtractOp>(&op)) {
            auto insertOp = builder.create<quantum::InsertOp>(
                extractOp.getLoc(), extractOp.getQreg().getType(),
                oldToCloned.lookup(extractOp.getQreg()),
                oldToCloned.lookupOrDefault(extractOp.getIdx()), extractOp.getIdxAttrAttr(),
                oldToCloned.lookup(extractOp.getQubit()));
            oldToCloned.map(extractOp.getQreg(), insertOp.getResult());
        }
        else if (auto adjointOp = dyn_cast<AdjointOp>(&op)) {
            BlockArgument regionArg = adjointOp.getRegion().getArgument(0);
            Value result = adjointOp.getResult();
            oldToCloned.map(regionArg, oldToCloned.lookup(result));
            Value reversedResult = cloneAdjointRegion(adjointOp, builder, oldToCloned);
            oldToCloned.map(adjointOp.getQreg(), reversedResult);
        }
        if (!isa<QuantumDialect>(op.getDialect())) {
            continue;
        }
    }
}

struct AdjointSingleOpRewritePattern : public mlir::OpRewritePattern<AdjointOp> {
    using mlir::OpRewritePattern<AdjointOp>::OpRewritePattern;

    /// We build a map from values mentioned in the source data flow to the values of
    /// the program where quantum control flow is reversed. Most of the time, there is a 1-to-1
    /// correspondence with a notable exception caused by `insert`/`extract` API asymmetry.
    mlir::LogicalResult matchAndRewrite(AdjointOp adjoint,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Adjointing the following:\n" << adjoint << "\n");
        Location loc = adjoint.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        auto paramVectorType = ArrayListType::get(ctx, rewriter.getF64Type());
        auto controlFlowTapeType = ArrayListType::get(ctx, rewriter.getIndexType());
        Value paramVector = rewriter.create<ListInitOp>(loc, paramVectorType);
        // Initialize the tapes that store the structure of control flow.
        DenseMap<Operation *, TypedValue<ArrayListType>> controlFlowTapes;
        adjoint.walk([&](Operation *op) {
            if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
                auto tape = rewriter.create<catalyst::ListInitOp>(loc, controlFlowTapeType);
                controlFlowTapes.insert({op, tape});
            }
        });

        // First, copy the classical computations directly to the target insertion point.
        IRMapping oldToCloned;
        cloneOnlyClassical(oldToCloned, adjoint.getRegion(), rewriter, paramVector,
                           controlFlowTapes);

        // Initialize the backward pass with the operand of the quantum.yield
        auto yieldOp = cast<quantum::YieldOp>(adjoint.getRegion().front().getTerminator());
        assert(yieldOp.getNumOperands() == 1 && "Expected quantum.yield to have one operand");
        oldToCloned.map(yieldOp.getOperands().front(), adjoint.getQreg());

        // Emit the adjoint quantum operations and reversed control flow, using cached values.
        generateReversedQuantum(oldToCloned, adjoint.getRegion(), rewriter, paramVector,
                                controlFlowTapes);

        // The final register is the re-mapped region argument of the original adjoint op.
        SmallVector<Value> reversedOutputs;
        for (BlockArgument arg : adjoint.getRegion().getArguments()) {
            reversedOutputs.push_back(oldToCloned.lookup(arg));
        }
        rewriter.replaceOp(adjoint, reversedOutputs);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateAdjointPatterns(RewritePatternSet &patterns)
{
    patterns.add<AdjointSingleOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
