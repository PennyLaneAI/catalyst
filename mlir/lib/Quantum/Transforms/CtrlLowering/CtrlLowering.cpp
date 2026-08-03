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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

/// Read a segment-sizes attribute from an operation and return it as a SmallVector<int32_t>. The
/// attribute is expected to be a DenseI32ArrayAttr, and the returned vector contains the sizes of
/// the operand/result segments in order.
static SmallVector<int32_t> readSegmentSizes(Operation *op, StringRef name) {
    auto seg = op->getAttrOfType<DenseI32ArrayAttr>(name);
    return SmallVector<int32_t>(seg.asArrayRef().begin(), seg.asArrayRef().end());
}

/// Rebuild a quantum gate with additional control qubits/values appended to whatever controls it
/// already carries. The new op is inserted at the rewriter's insertion point.
static Operation *createControlledGate(PatternRewriter &rewriter, QuantumGate gate, IRMapping &map,
                                       ValueRange addCtrlQubits, ValueRange addCtrlValues) {
    Operation *op = gate.getOperation();
    ValueRange nonCtrlQubits = gate.getNonCtrlQubitOperands();
    ValueRange oldCtrlQubits = gate.getCtrlQubitOperands();
    ValueRange oldCtrlValues = gate.getCtrlValueOperands();

    // Everything before the (non-ctrl) qubit operands is classical data (params, matrices, angles).
    unsigned numLeading =
        op->getNumOperands() - nonCtrlQubits.size() - oldCtrlQubits.size() - oldCtrlValues.size();

    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands() + addCtrlQubits.size() + addCtrlValues.size());
    for (unsigned i = 0; i < numLeading; ++i) {
        operands.push_back(map.lookupOrDefault(op->getOperand(i)));
    }
    for (Value q : nonCtrlQubits) {
        operands.push_back(map.lookupOrDefault(q));
    }
    for (Value q : oldCtrlQubits) {
        operands.push_back(map.lookupOrDefault(q));
    }
    operands.append(addCtrlQubits.begin(), addCtrlQubits.end());
    for (Value v : oldCtrlValues) {
        operands.push_back(map.lookupOrDefault(v));
    }
    operands.append(addCtrlValues.begin(), addCtrlValues.end());

    // The added controls grow the (last) out_ctrl_qubits result group.
    Type qubitType = QubitType::get(rewriter.getContext());
    SmallVector<Type> resultTypes(op->getResultTypes().begin(), op->getResultTypes().end());
    resultTypes.append(addCtrlQubits.size(), qubitType);

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    for (NamedAttribute attr : op->getAttrs()) {
        StringRef attrName = attr.getName().strref();
        if (attrName == "operandSegmentSizes" || attrName == "resultSegmentSizes") {
            continue;
        }
        state.addAttribute(attr.getName(), attr.getValue());
    }

    SmallVector<int32_t> operandSegments = readSegmentSizes(op, "operandSegmentSizes");
    operandSegments[operandSegments.size() - 2] += static_cast<int32_t>(addCtrlQubits.size());
    operandSegments[operandSegments.size() - 1] += static_cast<int32_t>(addCtrlValues.size());
    state.addAttribute("operandSegmentSizes", rewriter.getDenseI32ArrayAttr(operandSegments));

    if (op->getAttrOfType<DenseI32ArrayAttr>("resultSegmentSizes")) {
        SmallVector<int32_t> resultSegments = readSegmentSizes(op, "resultSegmentSizes");
        resultSegments[resultSegments.size() - 1] += static_cast<int32_t>(addCtrlQubits.size());
        state.addAttribute("resultSegmentSizes", rewriter.getDenseI32ArrayAttr(resultSegments));
    }

    return rewriter.create(state);
}

/// Rebuild a nested `quantum.ctrl` op with the enclosing controls merged in.
static CtrlOp mergeNestedCtrl(PatternRewriter &rewriter, CtrlOp inner, IRMapping &map,
                              ValueRange addCtrlQubits, ValueRange addCtrlValues) {
    Location loc = inner.getLoc();
    Type qubitType = QubitType::get(rewriter.getContext());

    SmallVector<Value> mergedCtrlQubits;
    for (Value q : inner.getInCtrlQubits()) {
        mergedCtrlQubits.push_back(map.lookupOrDefault(q));
    }
    mergedCtrlQubits.append(addCtrlQubits.begin(), addCtrlQubits.end());

    SmallVector<Value> mergedCtrlValues;
    for (Value v : inner.getInCtrlValues()) {
        mergedCtrlValues.push_back(map.lookupOrDefault(v));
    }
    mergedCtrlValues.append(addCtrlValues.begin(), addCtrlValues.end());

    SmallVector<Value> innerArgs;
    for (Value a : inner.getArgs()) {
        innerArgs.push_back(map.lookupOrDefault(a));
    }

    SmallVector<Value> operands;
    operands.append(mergedCtrlQubits.begin(), mergedCtrlQubits.end());
    operands.append(mergedCtrlValues.begin(), mergedCtrlValues.end());
    operands.append(innerArgs.begin(), innerArgs.end());

    // The target-results group is everything after the (leading) out_ctrl_qubits results.
    ResultRange innerResults = inner->getResults();
    unsigned numInnerCtrl = inner.getInCtrlQubits().size();
    unsigned numInnerTargets = innerResults.size() - numInnerCtrl;

    SmallVector<Type> resultTypes(mergedCtrlQubits.size(), qubitType);
    for (unsigned i = 0; i < numInnerTargets; ++i) {
        resultTypes.push_back(innerResults[numInnerCtrl + i].getType());
    }

    OperationState state(loc, CtrlOp::getOperationName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttribute("operandSegmentSizes",
                       rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(mergedCtrlQubits.size()),
                                                      static_cast<int32_t>(mergedCtrlValues.size()),
                                                      static_cast<int32_t>(innerArgs.size())}));
    state.addAttribute("resultSegmentSizes",
                       rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(mergedCtrlQubits.size()),
                                                      static_cast<int32_t>(numInnerTargets)}));
    state.addRegion();

    Operation *merged = rewriter.create(state);

    // Move the nested region body into the freshly created op (its block arguments, the target
    // qubits, are unaffected by adding controls).
    rewriter.inlineRegionBefore(inner.getRegion(), merged->getRegion(0),
                                merged->getRegion(0).end());
    return cast<CtrlOp>(merged);
}

/// Lower a single `quantum.ctrl` op by distributing its controls over the enclosed operations.
struct CtrlLoweringRewritePattern : public OpRewritePattern<CtrlOp> {
    using OpRewritePattern<CtrlOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CtrlOp ctrl, PatternRewriter &rewriter) const override
    {
        // Defer (not an error) if the region still contains a nested quantum.adjoint region.
        // Distributing controls needs an op-level body, so the inner region must be reduced first.
        // The pipeline runs (ctrl-lowering, adjoint-lowering) to a fixpoint: adjoint-lowering
        // reduces the inner region to op-level gates, then this ctrl op lowers on a later iteration.
        // A pre-scan avoids a partial rewrite (creating ops, then bailing out mid-region).
        if (ctrl.getRegion().walk([](AdjointOp) { return WalkResult::interrupt(); }).wasInterrupted()) {
            return failure();
        }

        Block &block = ctrl.getRegion().front();

        // Map the region's block arguments (the target qubits/registers) to the ctrl op operands.
        IRMapping map;
        for (auto [blockArg, operand] : llvm::zip_equal(block.getArguments(), ctrl.getArgs())) {
            map.map(blockArg, operand);
        }

        // The control qubits are threaded through every enclosed gate; the control values are
        // constant for the whole region.
        SmallVector<Value> currentCtrlQubits(ctrl.getInCtrlQubits().begin(),
                                             ctrl.getInCtrlQubits().end());
        ValueRange ctrlValues = ctrl.getInCtrlValues();

        rewriter.setInsertionPoint(ctrl);

        for (Operation &op : block.without_terminator()) {
            // Measurements (quantum.measure and MeasurementProcess ops) are already
            // rejected by the CtrlOp verifier, so they never reach here in a verified
            // pipeline.
            if (auto gate = dyn_cast<QuantumGate>(op)) {
                unsigned numOldControls = gate.getCtrlQubitOperands().size();
                Operation *newOp =
                    createControlledGate(rewriter, gate, map, currentCtrlQubits, ctrlValues);
                auto newGate = cast<QuantumGate>(newOp);

                for (auto [oldResult, newResult] : llvm::zip_equal(
                         gate.getNonCtrlQubitResults(), newGate.getNonCtrlQubitResults())) {
                    map.map(oldResult, newResult);
                }
                // The new control results are [old controls ..., threaded region controls ...].
                ResultRange newCtrlResults = newGate.getCtrlQubitResults();
                for (unsigned i = 0; i < numOldControls; ++i) {
                    map.map(gate.getCtrlQubitResults()[i], newCtrlResults[i]);
                }
                currentCtrlQubits.assign(newCtrlResults.begin() + numOldControls,
                                         newCtrlResults.end());
                continue;
            }
            if (auto inner = dyn_cast<CtrlOp>(op)) {
                unsigned numInnerControls = inner.getInCtrlQubits().size();
                CtrlOp merged =
                    mergeNestedCtrl(rewriter, inner, map, currentCtrlQubits, ctrlValues);

                ResultRange mergedCtrlResults = merged.getOutCtrlQubits();
                for (unsigned i = 0; i < numInnerControls; ++i) {
                    map.map(inner.getOutCtrlQubits()[i], mergedCtrlResults[i]);
                }
                // Map the target results (everything after the out_ctrl_qubits group) one-to-one.
                ResultRange innerAll = inner->getResults();
                ResultRange mergedAll = merged->getResults();
                unsigned numMergedControls = merged.getInCtrlQubits().size();
                unsigned numTargets = innerAll.size() - numInnerControls;
                for (unsigned i = 0; i < numTargets; ++i) {
                    map.map(innerAll[numInnerControls + i], mergedAll[numMergedControls + i]);
                }
                currentCtrlQubits.assign(mergedCtrlResults.begin() + numInnerControls,
                                         mergedCtrlResults.end());
                continue;
            }
            if (isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp>(op)) {
                op.emitError(
                    "control flow inside a quantum.ctrl region is not supported by ctrl-lowering");
                return failure();
            }
            if (isa<InsertOp, ExtractOp, AllocOp, DeallocOp, AllocQubitOp, DeallocQubitOp>(op)) {
                // Structural ops carry no controls; thread their operands/results through the map.
                rewriter.clone(op, map);
                continue;
            }
            if (isa<QuantumDialect>(op.getDialect())) {
                op.emitError("unsupported quantum operation inside a quantum.ctrl region");
                return failure();
            }
            // Classical op: clone it, threading operands and recording result mappings.
            rewriter.clone(op, map);
        }

        // Assemble the ctrl op results: out_ctrl_qubits followed by the target results.
        auto yield = cast<YieldOp>(block.getTerminator());
        SmallVector<Value> results;
        results.append(currentCtrlQubits.begin(), currentCtrlQubits.end());
        for (Value retval : yield.getRetvals()) {
            results.push_back(map.lookupOrDefault(retval));
        }
        rewriter.replaceOp(ctrl, results);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_CTRLLOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct CtrlLoweringPass : impl::CtrlLoweringPassBase<CtrlLoweringPass> {
    using CtrlLoweringPassBase::CtrlLoweringPassBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        patterns.add<CtrlLoweringRewritePattern>(patterns.getContext(), 1);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
