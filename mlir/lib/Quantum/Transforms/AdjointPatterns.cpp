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
#include "Quantum/Utils/QuantumSplitting.h"

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

/// Generate the quantum "backwards pass" of the adjoint operation using the stored gate parameters
/// and cached control flow.
void generateReversedQuantum(IRMapping &oldToCloned, Region &region, OpBuilder &builder,
                             QuantumCache &cache)
{
    assert(region.hasOneBlock() &&
           "Expected only structured control flow (each region should have a single block)");

    auto getQuantumReg = [](ValueRange values) -> std::optional<Value> {
        for (Value value : values) {
            if (isa<quantum::QuregType>(value.getType())) {
                return value;
            }
        }
        return std::nullopt;
    };
    for (Operation &op : llvm::reverse(region.front().without_terminator())) {
        LLVM_DEBUG(dbgs() << "generating adjoint for: " << op << "\n");
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            std::optional<Value> yieldedQureg =
                getQuantumReg(forOp.getBody()->getTerminator()->getOperands());
            if (!yieldedQureg.has_value()) {
                // This operation is purely classical
                continue;
            }

            Value tape = cache.controlFlowTapes.at(forOp);
            // Popping the start, stop, and step implies that these are backwards relative to the
            // order they were pushed.
            Value step = builder.create<ListPopOp>(forOp.getLoc(), tape);
            Value stop = builder.create<ListPopOp>(forOp.getLoc(), tape);
            Value start = builder.create<ListPopOp>(forOp.getLoc(), tape);

            Value reversedResult = oldToCloned.lookup(getQuantumReg(forOp.getResults()).value());
            auto replacedFor = builder.create<scf::ForOp>(
                forOp.getLoc(), start, stop, step, reversedResult,
                [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
                    oldToCloned.map(yieldedQureg.value(), iterArgs[0]);
                    generateReversedQuantum(oldToCloned, forOp.getBodyRegion(), builder, cache);
                    builder.create<scf::YieldOp>(
                        loc, oldToCloned.lookup(getQuantumReg(forOp.getRegionIterArgs()).value()));
                });
            oldToCloned.map(getQuantumReg(forOp.getInitArgs()).value(), replacedFor.getResult(0));
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            std::optional<Value> qureg = getQuantumReg(ifOp.getResults());
            if (!qureg.has_value()) {
                // This operation is purely classical
                continue;
            }

            Value tape = cache.controlFlowTapes.at(ifOp);
            Value condition = builder.create<ListPopOp>(ifOp.getLoc(), tape);
            condition =
                builder.create<index::CastSOp>(ifOp.getLoc(), builder.getI1Type(), condition);
            Value reversedResult = oldToCloned.lookup(getQuantumReg(ifOp.getResults()).value());

            // The quantum register is captured from outside rather than passed in through a
            // basic block argument. We thus need to traverse the region to look for it.
            auto findOldestQuregInRegion = [&](Region &region) {
                for (Operation &innerOp : region.getOps()) {
                    for (Value operand : innerOp.getOperands()) {
                        if (isa<quantum::QuregType>(operand.getType())) {
                            return operand;
                        }
                    }
                }
                llvm_unreachable("failed to find qureg in scf.if region");
            };
            auto getRegionBuilder = [&](Region &oldRegion) {
                return [&](OpBuilder &builder, Location loc) {
                    std::optional<Value> yieldedQureg =
                        getQuantumReg(oldRegion.front().getTerminator()->getOperands());
                    oldToCloned.map(yieldedQureg.value(), reversedResult);
                    generateReversedQuantum(oldToCloned, oldRegion, builder, cache);
                    builder.create<scf::YieldOp>(
                        loc, oldToCloned.lookup(findOldestQuregInRegion(oldRegion)));
                };
            };
            auto reversedIf = builder.create<scf::IfOp>(ifOp.getLoc(), condition,
                                                        getRegionBuilder(ifOp.getThenRegion()),
                                                        getRegionBuilder(ifOp.getElseRegion()));
            Value startingThenQureg = findOldestQuregInRegion(ifOp.getThenRegion());
            Value startingElseQureg = findOldestQuregInRegion(ifOp.getElseRegion());
            assert(startingThenQureg == startingElseQureg &&
                   "Expected the same input register for both scf.if branches");
            oldToCloned.map(startingThenQureg, reversedIf.getResult(0));
        }
        else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
            std::optional<Value> yieldedQureg =
                getQuantumReg(whileOp.getAfter().front().getTerminator()->getOperands());
            if (!yieldedQureg.has_value()) {
                // This operation is purely classical
                continue;
            }

            Value tape = cache.controlFlowTapes.at(whileOp);
            Value numIterations = builder.create<ListPopOp>(whileOp.getLoc(), tape);
            Value c0 = builder.create<index::ConstantOp>(whileOp.getLoc(), 0);
            Value c1 = builder.create<index::ConstantOp>(whileOp.getLoc(), 1);

            Value iterArgInit = oldToCloned.lookup(getQuantumReg(whileOp.getResults()).value());
            auto replacedWhile = builder.create<scf::ForOp>(
                whileOp.getLoc(), /*start=*/c0, /*stop=*/numIterations, /*step=*/c1, iterArgInit,
                /*bodyBuilder=*/
                [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
                    oldToCloned.map(yieldedQureg.value(), iterArgs[0]);
                    generateReversedQuantum(oldToCloned, whileOp.getAfter(), builder, cache);
                    builder.create<scf::YieldOp>(
                        loc, oldToCloned.lookup(
                                 getQuantumReg(whileOp.getAfter().front().getArguments()).value()));
                });
            oldToCloned.map(getQuantumReg(whileOp.getInits()).value(), replacedWhile.getResult(0));
        }
        else if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
            Value dynamicWire;
            if (!insertOp.getIdxAttr().has_value()) {
                dynamicWire = builder.create<ListPopOp>(insertOp.getLoc(), cache.wireVector);
            }
            auto extractOp = builder.create<quantum::ExtractOp>(
                insertOp.getLoc(), insertOp.getQubit().getType(),
                oldToCloned.lookup(insertOp.getOutQreg()), dynamicWire, insertOp.getIdxAttrAttr());
            oldToCloned.map(insertOp.getQubit(), extractOp.getResult());
            oldToCloned.map(insertOp.getInQreg(), oldToCloned.lookup(insertOp.getOutQreg()));
        }
        else if (auto gate = dyn_cast<quantum::QuantumGate>(op)) {
            for (const auto &[qubitResult, qubitOperand] :
                 llvm::zip(gate.getQubitResults(), gate.getQubitOperands())) {
                oldToCloned.map(qubitOperand, oldToCloned.lookup(qubitResult));
            }

            auto clone = cast<QuantumGate>(builder.clone(*gate, oldToCloned));
            clone.setAdjointFlag(!gate.getAdjointFlag());

            // Read cached differentiable parameters from the recorded parameter vector.
            if (auto differentiableGate = dyn_cast<quantum::DifferentiableGate>(op)) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.setInsertionPoint(clone);
                SmallVector<Value> cachedParams;
                ValueRange diffParams = differentiableGate.getDiffParams();
                for (unsigned i = 0; i < diffParams.size(); i++) {
                    cachedParams.push_back(
                        builder.create<ListPopOp>(differentiableGate.getLoc(), cache.paramVector));
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
        else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
            Value dynamicWire;
            if (!extractOp.getIdxAttr().has_value()) {
                dynamicWire = builder.create<ListPopOp>(extractOp.getLoc(), cache.wireVector);
            }
            auto insertOp = builder.create<quantum::InsertOp>(
                extractOp.getLoc(), extractOp.getQreg().getType(),
                oldToCloned.lookup(extractOp.getQreg()), dynamicWire, extractOp.getIdxAttrAttr(),
                oldToCloned.lookup(extractOp.getQubit()));
            oldToCloned.map(extractOp.getQreg(), insertOp.getResult());
        }
        else if (auto adjointOp = dyn_cast<quantum::AdjointOp>(&op)) {
            BlockArgument regionArg = adjointOp.getRegion().getArgument(0);
            Value result = adjointOp.getResult();
            oldToCloned.map(regionArg, oldToCloned.lookup(result));
            Value reversedResult = cloneAdjointRegion(adjointOp, builder, oldToCloned);
            oldToCloned.map(adjointOp.getQreg(), reversedResult);
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
        auto cache = QuantumCache::initialize(adjoint.getRegion(), rewriter, adjoint.getLoc());

        // First, copy the classical computations directly to the target insertion point.
        IRMapping oldToCloned;
        AugmentedCircuitGenerator generator{oldToCloned, rewriter, cache};
        generator.generate(adjoint.getRegion());

        // Initialize the backward pass with the operand of the quantum.yield
        auto yieldOp = cast<quantum::YieldOp>(adjoint.getRegion().front().getTerminator());
        assert(yieldOp.getNumOperands() == 1 && "Expected quantum.yield to have one operand");
        oldToCloned.map(yieldOp.getOperands().front(), adjoint.getQreg());

        // Emit the adjoint quantum operations and reversed control flow, using cached values.
        generateReversedQuantum(oldToCloned, adjoint.getRegion(), rewriter, cache);

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
