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

/// A class that generates the quantum "backwards pass" of the adjoint operation using the stored
/// gate parameters and cached control flow.
class AdjointGenerator {
  public:
    AdjointGenerator(IRMapping &remappedValues, OpBuilder &builder, QuantumCache &cache)
        : remappedValues(remappedValues), builder(builder), cache(cache)
    {
    }

    /// Recursively generate the adjoint version of `region` with reversed control flow and adjoint
    /// quantum gates.
    void generate(Region &region)
    {
        assert(region.hasOneBlock() &&
               "Expected only structured control flow (each region should have a single block)");

        for (Operation &op : llvm::reverse(region.front().without_terminator())) {
            LLVM_DEBUG(dbgs() << "generating adjoint for: " << op << "\n");
            if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                visitOperation(forOp);
            }
            else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                visitOperation(ifOp);
            }
            else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
                visitOperation(whileOp);
            }
            else if (auto insertOp = dyn_cast<quantum::InsertOp>(op)) {
                Value dynamicWire = getDynamicWire(insertOp);
                auto extractOp = builder.create<quantum::ExtractOp>(
                    insertOp.getLoc(), insertOp.getQubit().getType(),
                    remappedValues.lookup(insertOp.getOutQreg()), dynamicWire,
                    insertOp.getIdxAttrAttr());
                remappedValues.map(insertOp.getQubit(), extractOp.getResult());
                remappedValues.map(insertOp.getInQreg(),
                                   remappedValues.lookup(insertOp.getOutQreg()));
            }
            else if (auto extractOp = dyn_cast<quantum::ExtractOp>(op)) {
                Value dynamicWire = getDynamicWire(extractOp);
                auto insertOp = builder.create<quantum::InsertOp>(
                    extractOp.getLoc(), extractOp.getQreg().getType(),
                    remappedValues.lookup(extractOp.getQreg()), dynamicWire,
                    extractOp.getIdxAttrAttr(), remappedValues.lookup(extractOp.getQubit()));
                remappedValues.map(extractOp.getQreg(), insertOp.getResult());
            }
            else if (auto gate = dyn_cast<quantum::QuantumGate>(op)) {
                for (const auto &[qubitResult, qubitOperand] :
                     llvm::zip(gate.getQubitResults(), gate.getQubitOperands())) {
                    remappedValues.map(qubitOperand, remappedValues.lookup(qubitResult));
                }

                auto clone = cast<QuantumGate>(builder.clone(*gate, remappedValues));
                clone.setAdjointFlag(!gate.getAdjointFlag());

                // Read cached differentiable parameters from the recorded parameter vector.
                if (auto differentiableGate = dyn_cast<quantum::DifferentiableGate>(op)) {
                    OpBuilder::InsertionGuard insertionGuard(builder);
                    builder.setInsertionPoint(clone);
                    SmallVector<Value> cachedParams;
                    ValueRange diffParams = differentiableGate.getDiffParams();
                    for (unsigned i = 0; i < diffParams.size(); i++) {
                        cachedParams.push_back(builder.create<ListPopOp>(
                            differentiableGate.getLoc(), cache.paramVector));
                    }
                    MutableOperandRange(clone, differentiableGate.getDiffOperandIdx(),
                                        diffParams.size())
                        .assign(cachedParams);
                }

                for (const auto &[qubitResult, qubitOperand] :
                     llvm::zip(clone.getQubitResults(), gate.getQubitOperands())) {
                    remappedValues.map(qubitOperand, qubitResult);
                }
            }
            else if (auto adjointOp = dyn_cast<quantum::AdjointOp>(&op)) {
                BlockArgument regionArg = adjointOp.getRegion().getArgument(0);
                Value result = adjointOp.getResult();
                remappedValues.map(regionArg, remappedValues.lookup(result));
                Value reversedResult = cloneAdjointRegion(adjointOp, builder, remappedValues);
                remappedValues.map(adjointOp.getQreg(), reversedResult);
            }
        }
    }

    template <typename IndexingOp> Value getDynamicWire(IndexingOp op)
    {
        Value dynamicWire;
        if (!op.getIdxAttr().has_value()) {
            dynamicWire = builder.create<ListPopOp>(op.getLoc(), cache.wireVector);
        }
        return dynamicWire;
    }

    std::optional<Value> getQuantumReg(ValueRange values)
    {
        for (Value value : values) {
            if (isa<quantum::QuregType>(value.getType())) {
                return value;
            }
        }
        return std::nullopt;
    }

    void visitOperation(scf::ForOp forOp)
    {
        std::optional<Value> yieldedQureg =
            getQuantumReg(forOp.getBody()->getTerminator()->getOperands());
        if (!yieldedQureg.has_value()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(forOp);
        // Popping the start, stop, and step implies that these are backwards relative to
        // the order they were pushed.
        Value step = builder.create<ListPopOp>(forOp.getLoc(), tape);
        Value stop = builder.create<ListPopOp>(forOp.getLoc(), tape);
        Value start = builder.create<ListPopOp>(forOp.getLoc(), tape);

        Value reversedResult = remappedValues.lookup(getQuantumReg(forOp.getResults()).value());
        auto replacedFor = builder.create<scf::ForOp>(
            forOp.getLoc(), start, stop, step, /*iterArgsInit=*/reversedResult,
            [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange iterArgs) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                remappedValues.map(yieldedQureg.value(), iterArgs[0]);
                generate(forOp.getBodyRegion());
                builder.create<scf::YieldOp>(
                    loc, remappedValues.lookup(getQuantumReg(forOp.getRegionIterArgs()).value()));
            });
        remappedValues.map(getQuantumReg(forOp.getInitArgs()).value(), replacedFor.getResult(0));
    }

    void visitOperation(scf::IfOp ifOp)
    {
        std::optional<Value> qureg = getQuantumReg(ifOp.getResults());
        if (!qureg.has_value()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(ifOp);
        Value condition = builder.create<ListPopOp>(ifOp.getLoc(), tape);
        condition = builder.create<index::CastSOp>(ifOp.getLoc(), builder.getI1Type(), condition);
        Value reversedResult = remappedValues.lookup(getQuantumReg(ifOp.getResults()).value());

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
            return [&](OpBuilder &bodyBuilder, Location loc) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                std::optional<Value> yieldedQureg =
                    getQuantumReg(oldRegion.front().getTerminator()->getOperands());
                remappedValues.map(yieldedQureg.value(), reversedResult);
                generate(oldRegion);
                builder.create<scf::YieldOp>(
                    loc, remappedValues.lookup(findOldestQuregInRegion(oldRegion)));
            };
        };
        auto reversedIf = builder.create<scf::IfOp>(ifOp.getLoc(), condition,
                                                    getRegionBuilder(ifOp.getThenRegion()),
                                                    getRegionBuilder(ifOp.getElseRegion()));
        Value startingThenQureg = findOldestQuregInRegion(ifOp.getThenRegion());
        Value startingElseQureg = findOldestQuregInRegion(ifOp.getElseRegion());
        assert(startingThenQureg == startingElseQureg &&
               "Expected the same input register for both scf.if branches");
        remappedValues.map(startingThenQureg, reversedIf.getResult(0));
    }

    void visitOperation(scf::WhileOp whileOp)
    {
        std::optional<Value> yieldedQureg =
            getQuantumReg(whileOp.getAfter().front().getTerminator()->getOperands());
        if (!yieldedQureg.has_value()) {
            // This operation is purely classical
            return;
        }

        Value tape = cache.controlFlowTapes.at(whileOp);
        Value numIterations = builder.create<ListPopOp>(whileOp.getLoc(), tape);
        Value c0 = builder.create<index::ConstantOp>(whileOp.getLoc(), 0);
        Value c1 = builder.create<index::ConstantOp>(whileOp.getLoc(), 1);

        Value iterArgInit = remappedValues.lookup(getQuantumReg(whileOp.getResults()).value());
        auto replacedWhile = builder.create<scf::ForOp>(
            whileOp.getLoc(), /*start=*/c0, /*stop=*/numIterations, /*step=*/c1, iterArgInit,
            /*bodyBuilder=*/
            [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange iterArgs) {
                OpBuilder::InsertionGuard insertionGuard(builder);
                builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());

                remappedValues.map(yieldedQureg.value(), iterArgs[0]);
                generate(whileOp.getAfter());
                builder.create<scf::YieldOp>(
                    loc, remappedValues.lookup(
                             getQuantumReg(whileOp.getAfter().front().getArguments()).value()));
            });
        remappedValues.map(getQuantumReg(whileOp.getInits()).value(), replacedWhile.getResult(0));
    }

  private:
    IRMapping &remappedValues;
    OpBuilder &builder;
    QuantumCache &cache;
};

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
        AugmentedCircuitGenerator augmentedGenerator{oldToCloned, rewriter, cache};
        augmentedGenerator.generate(adjoint.getRegion());

        // Initialize the backward pass with the operand of the quantum.yield
        auto yieldOp = cast<quantum::YieldOp>(adjoint.getRegion().front().getTerminator());
        assert(yieldOp.getNumOperands() == 1 && "Expected quantum.yield to have one operand");
        oldToCloned.map(yieldOp.getOperands().front(), adjoint.getQreg());

        // Emit the adjoint quantum operations and reversed control flow, using cached values.
        AdjointGenerator adjointGenerator{oldToCloned, rewriter, cache};
        adjointGenerator.generate(adjoint.getRegion());

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
