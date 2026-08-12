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

/// Control every op in `block` (excluding its terminator) on `currentCtrlQubits`, threading the
/// control qubits through and recording result mappings in `map`. On return, `currentCtrlQubits`
/// holds the control-qubit values after the last op. New ops are created at the rewriter's current
/// insertion point.
static LogicalResult distributeControls(PatternRewriter &rewriter, Block &block, IRMapping &map,
                                        SmallVector<Value> &currentCtrlQubits,
                                        ValueRange ctrlValues);

/// Control an `scf.if` by turning the control qubits into extra results: each branch tracks the
/// controls through its body and yields them alongside the original results.
static LogicalResult controlScfIf(PatternRewriter &rewriter, scf::IfOp ifOp, IRMapping &map,
                                  SmallVector<Value> &currentCtrlQubits, ValueRange ctrlValues) {
    Type qubitType = QubitType::get(rewriter.getContext());
    unsigned numCtrl = currentCtrlQubits.size();

    // New results = the if's original results, plus one qubit per threaded control.
    SmallVector<Type> resultTypes(ifOp.getResultTypes().begin(), ifOp.getResultTypes().end());
    resultTypes.append(numCtrl, qubitType);

    Value cond = map.lookupOrDefault(ifOp.getCondition());
    auto newIf = scf::IfOp::create(rewriter, ifOp.getLoc(), resultTypes, cond,
                                   /*withElseRegion=*/true);

    // Control one branch: `oldBlock` may be null (a missing else), in which case the branch just
    // threads the incoming controls through unchanged.
    auto controlBranch = [&](Block *oldBlock, Block *newBlock) -> LogicalResult {
        IRMapping branchMap = map; // copy: branch-local mappings must not leak across branches
        SmallVector<Value> branchCtrl(currentCtrlQubits.begin(), currentCtrlQubits.end());
        rewriter.setInsertionPointToStart(newBlock);

        SmallVector<Value> yielded;
        Location yieldLoc = ifOp.getLoc();
        if (oldBlock) {
            if (failed(
                    distributeControls(rewriter, *oldBlock, branchMap, branchCtrl, ctrlValues))) {
                return failure();
            }
            auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
            yieldLoc = oldYield.getLoc();
            for (Value v : oldYield.getOperands()) {
                yielded.push_back(branchMap.lookupOrDefault(v));
            }
        }
        yielded.append(branchCtrl.begin(), branchCtrl.end());
        rewriter.setInsertionPointToEnd(newBlock);
        scf::YieldOp::create(rewriter, yieldLoc, yielded);
        return success();
    };

    if (failed(controlBranch(&ifOp.getThenRegion().front(), newIf.thenBlock()))) {
        return failure();
    }
    if (failed(controlBranch(ifOp.elseBlock(), newIf.elseBlock()))) {
        return failure();
    }

    // Map the original results one-to-one; the trailing results are the threaded control qubits.
    unsigned numOrig = ifOp.getNumResults();
    for (unsigned i = 0; i < numOrig; ++i) {
        map.map(ifOp.getResult(i), newIf.getResult(i));
    }
    currentCtrlQubits.assign(newIf.getResults().begin() + numOrig, newIf.getResults().end());

    // Restore the insertion point to after the new op so the enclosing walk keeps appending there
    // (controlBranch left it inside the else block).
    rewriter.setInsertionPointAfter(newIf);
    return success();
}

/// Control an `scf.for` by adding the control qubits as extra loop-carried `iter_args`: the loop
/// structure (bounds, iteration) stays classical, the control qubits are tracked through the body
/// each iteration, and the final controls come out as the loop's trailing results.
static LogicalResult controlScfFor(PatternRewriter &rewriter, scf::ForOp forOp, IRMapping &map,
                                   SmallVector<Value> &currentCtrlQubits, ValueRange ctrlValues) {
    unsigned numOrig = forOp.getInitArgs().size();

    // New init args = the loop's original init args, plus the incoming control qubits.
    SmallVector<Value> newInits;
    for (Value init : forOp.getInitArgs()) {
        newInits.push_back(map.lookupOrDefault(init));
    }
    newInits.append(currentCtrlQubits.begin(), currentCtrlQubits.end());

    Value lb = map.lookupOrDefault(forOp.getLowerBound());
    Value ub = map.lookupOrDefault(forOp.getUpperBound());
    Value step = map.lookupOrDefault(forOp.getStep());

    // With non-empty iter args and no body-builder, scf.for creates the body block (induction var +
    // iter-arg block args) without a terminator, which we fill in below.
    auto newFor = scf::ForOp::create(rewriter, forOp.getLoc(), lb, ub, step, newInits);
    Block *newBody = newFor.getBody();
    ValueRange newIterArgs = newFor.getRegionIterArgs();

    // Seed a body-local mapping; the trailing iter args are this iteration's threaded controls.
    IRMapping bodyMap = map;
    bodyMap.map(forOp.getInductionVar(), newFor.getInductionVar());
    for (unsigned i = 0; i < numOrig; ++i) {
        bodyMap.map(forOp.getRegionIterArg(i), newIterArgs[i]);
    }
    SmallVector<Value> bodyCtrl(newIterArgs.begin() + numOrig, newIterArgs.end());

    rewriter.setInsertionPointToStart(newBody);
    Block &oldBody = forOp.getRegion().front();
    if (failed(distributeControls(rewriter, oldBody, bodyMap, bodyCtrl, ctrlValues))) {
        return failure();
    }

    auto oldYield = cast<scf::YieldOp>(oldBody.getTerminator());
    SmallVector<Value> yielded;
    for (Value v : oldYield.getOperands()) {
        yielded.push_back(bodyMap.lookupOrDefault(v));
    }
    yielded.append(bodyCtrl.begin(), bodyCtrl.end());
    rewriter.setInsertionPointToEnd(newBody);
    scf::YieldOp::create(rewriter, oldYield.getLoc(), yielded);

    // Map the original results one-to-one; the trailing results are the threaded control qubits.
    for (unsigned i = 0; i < numOrig; ++i) {
        map.map(forOp.getResult(i), newFor.getResult(i));
    }
    currentCtrlQubits.assign(newFor.getResults().begin() + numOrig, newFor.getResults().end());

    // Restore the insertion point to after the new op so the enclosing walk keeps appending there.
    rewriter.setInsertionPointAfter(newFor);
    return success();
}

/// Control an `scf.while` by adding the control qubits as extra loop-carried values.
static LogicalResult controlScfWhile(PatternRewriter &rewriter, scf::WhileOp whileOp,
                                     IRMapping &map, SmallVector<Value> &currentCtrlQubits,
                                     ValueRange ctrlValues) {
    Type qubitType = QubitType::get(rewriter.getContext());
    unsigned numCtrl = currentCtrlQubits.size();
    unsigned numInit = whileOp.getInits().size();

    // New inits = original inits, plus the incoming control qubits.
    SmallVector<Value> newInits;
    for (Value init : whileOp.getInits()) {
        newInits.push_back(map.lookupOrDefault(init));
    }
    newInits.append(currentCtrlQubits.begin(), currentCtrlQubits.end());

    SmallVector<Type> newResultTypes(whileOp.getResultTypes().begin(),
                                     whileOp.getResultTypes().end());
    newResultTypes.append(numCtrl, qubitType);

    Block &oldBefore = whileOp.getBefore().front();
    Block &oldAfter = whileOp.getAfter().front();
    scf::ConditionOp oldCond = whileOp.getConditionOp();
    auto oldYield = cast<scf::YieldOp>(oldAfter.getTerminator());
    // For the results: the `before` block forwards them and the `after` block
    // receives them, and they all surface as the while results when the condition
    // is false.
    unsigned numFwd = oldCond.getArgs().size();

    LogicalResult status = success();
    // The before/after builder overload creates the before block args (init types + controls)
    // and after block args (result types + controls);
    // the callbacks fill each region.
    auto newWhile = scf::WhileOp::create(
        rewriter, whileOp.getLoc(), newResultTypes, newInits,
        [&](OpBuilder & /*builder*/, Location /*loc*/, ValueRange beforeArgs) {
            IRMapping beforeMap = map;
            for (unsigned i = 0; i < numInit; ++i) {
                beforeMap.map(oldBefore.getArgument(i), beforeArgs[i]);
            }
            SmallVector<Value> beforeCtrl(beforeArgs.begin() + numInit, beforeArgs.end());
            if (failed(
                    distributeControls(rewriter, oldBefore, beforeMap, beforeCtrl, ctrlValues))) {
                status = failure();
                return;
            }
            Value cond = beforeMap.lookupOrDefault(oldCond.getCondition());
            SmallVector<Value> fwd;
            for (Value v : oldCond.getArgs()) {
                fwd.push_back(beforeMap.lookupOrDefault(v));
            }
            fwd.append(beforeCtrl.begin(), beforeCtrl.end());
            scf::ConditionOp::create(rewriter, oldCond.getLoc(), cond, fwd);
        },
        [&](OpBuilder & /*builder*/, Location /*loc*/, ValueRange afterArgs) {
            IRMapping afterMap = map;
            for (unsigned i = 0; i < numFwd; ++i) {
                afterMap.map(oldAfter.getArgument(i), afterArgs[i]);
            }
            SmallVector<Value> afterCtrl(afterArgs.begin() + numFwd, afterArgs.end());
            if (failed(distributeControls(rewriter, oldAfter, afterMap, afterCtrl, ctrlValues))) {
                status = failure();
                return;
            }
            SmallVector<Value> yielded;
            for (Value v : oldYield.getOperands()) {
                yielded.push_back(afterMap.lookupOrDefault(v));
            }
            yielded.append(afterCtrl.begin(), afterCtrl.end());
            scf::YieldOp::create(rewriter, oldYield.getLoc(), yielded);
        });

    if (failed(status)) {
        return failure();
    }

    // Map the original results one-to-one; the trailing results are the threaded control qubits.
    unsigned numOrig = whileOp.getNumResults();
    for (unsigned i = 0; i < numOrig; ++i) {
        map.map(whileOp.getResult(i), newWhile.getResult(i));
    }
    currentCtrlQubits.assign(newWhile.getResults().begin() + numOrig, newWhile.getResults().end());
    rewriter.setInsertionPointAfter(newWhile);
    return success();
}

/// Control an `scf.index_switch` by turning the control qubits into extra results
static LogicalResult controlScfIndexSwitch(PatternRewriter &rewriter, scf::IndexSwitchOp switchOp,
                                           IRMapping &map, SmallVector<Value> &currentCtrlQubits,
                                           ValueRange ctrlValues) {
    Type qubitType = QubitType::get(rewriter.getContext());
    unsigned numCtrl = currentCtrlQubits.size();

    SmallVector<Type> resultTypes(switchOp.getResultTypes().begin(),
                                  switchOp.getResultTypes().end());
    resultTypes.append(numCtrl, qubitType);

    Value arg = map.lookupOrDefault(switchOp.getArg());
    SmallVector<int64_t> cases(switchOp.getCases().begin(), switchOp.getCases().end());
    auto newSwitch = scf::IndexSwitchOp::create(rewriter, switchOp.getLoc(), resultTypes, arg,
                                                cases, cases.size());

    // Control one region.
    // Regions have no block arguments, so bodies reference outer
    // values via `map`, as with scf.if branches.
    auto controlCase = [&](Block &oldBlock, Region &newRegion) -> LogicalResult {
        IRMapping caseMap = map;
        SmallVector<Value> caseCtrl(currentCtrlQubits.begin(), currentCtrlQubits.end());
        Block *newBlock = rewriter.createBlock(&newRegion);
        if (failed(distributeControls(rewriter, oldBlock, caseMap, caseCtrl, ctrlValues))) {
            return failure();
        }
        auto oldYield = cast<scf::YieldOp>(oldBlock.getTerminator());
        SmallVector<Value> yielded;
        for (Value v : oldYield.getOperands()) {
            yielded.push_back(caseMap.lookupOrDefault(v));
        }
        yielded.append(caseCtrl.begin(), caseCtrl.end());
        rewriter.setInsertionPointToEnd(newBlock);
        scf::YieldOp::create(rewriter, oldYield.getLoc(), yielded);
        return success();
    };

    for (unsigned i = 0; i < cases.size(); ++i) {
        if (failed(controlCase(switchOp.getCaseBlock(i), newSwitch.getCaseRegions()[i]))) {
            return failure();
        }
    }
    if (failed(controlCase(switchOp.getDefaultBlock(), newSwitch.getDefaultRegion()))) {
        return failure();
    }

    unsigned numOrig = switchOp.getNumResults();
    for (unsigned i = 0; i < numOrig; ++i) {
        map.map(switchOp.getResult(i), newSwitch.getResult(i));
    }
    currentCtrlQubits.assign(newSwitch.getResults().begin() + numOrig,
                             newSwitch.getResults().end());
    rewriter.setInsertionPointAfter(newSwitch);
    return success();
}

static LogicalResult distributeControls(PatternRewriter &rewriter, Block &block, IRMapping &map,
                                        SmallVector<Value> &currentCtrlQubits,
                                        ValueRange ctrlValues) {
    for (Operation &op : block.without_terminator()) {
        if (isa<MeasurementProcess, MeasureOp>(op)) {
            op.emitError("cannot control a measurement inside a quantum.ctrl region");
            return failure();
        }
        if (auto gate = dyn_cast<QuantumGate>(op)) {
            unsigned numOldControls = gate.getCtrlQubitOperands().size();
            Operation *newOp =
                createControlledGate(rewriter, gate, map, currentCtrlQubits, ctrlValues);
            auto newGate = cast<QuantumGate>(newOp);

            for (auto [oldResult, newResult] :
                 llvm::zip_equal(gate.getNonCtrlQubitResults(), newGate.getNonCtrlQubitResults())) {
                map.map(oldResult, newResult);
            }
            // The new control results are [old controls ..., threaded region controls ...].
            ResultRange newCtrlResults = newGate.getCtrlQubitResults();
            for (unsigned i = 0; i < numOldControls; ++i) {
                map.map(gate.getCtrlQubitResults()[i], newCtrlResults[i]);
            }
            currentCtrlQubits.assign(newCtrlResults.begin() + numOldControls, newCtrlResults.end());
            continue;
        }
        if (auto inner = dyn_cast<CtrlOp>(op)) {
            unsigned numInnerControls = inner.getInCtrlQubits().size();
            CtrlOp merged = mergeNestedCtrl(rewriter, inner, map, currentCtrlQubits, ctrlValues);

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
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            if (failed(controlScfIf(rewriter, ifOp, map, currentCtrlQubits, ctrlValues))) {
                return failure();
            }
            continue;
        }
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            if (failed(controlScfFor(rewriter, forOp, map, currentCtrlQubits, ctrlValues))) {
                return failure();
            }
            continue;
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
            if (failed(controlScfWhile(rewriter, whileOp, map, currentCtrlQubits, ctrlValues))) {
                return failure();
            }
            continue;
        }
        if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
            if (failed(controlScfIndexSwitch(rewriter, switchOp, map, currentCtrlQubits,
                                             ctrlValues))) {
                return failure();
            }
            continue;
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
        // Any other scf ops would need their body controlled too,
        // which is not supported:
        if (op.getNumRegions() > 0) {
            op.emitError("unsupported scf operation inside a quantum.ctrl region");
            return failure();
        }
        // Classical op: clone it:
        rewriter.clone(op, map);
    }
    return success();
}

/// Lower a single `quantum.ctrl` op by distributing its controls over the enclosed operations.
struct CtrlLoweringRewritePattern : public OpRewritePattern<CtrlOp> {
    using OpRewritePattern<CtrlOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CtrlOp ctrl, PatternRewriter &rewriter) const override {
        // Defer (not an error) if the region still contains a nested quantum.adjoint region.
        // Distributing controls needs an op-level body, so the inner region must be reduced first.
        // The pipeline runs (ctrl-lowering, adjoint-lowering) to a fixpoint: adjoint-lowering
        // reduces the inner region to op-level gates, then this ctrl op lowers on a later
        // iteration. A pre-scan avoids a partial rewrite (creating ops, then bailing out
        // mid-region).
        if (ctrl.getRegion()
                .walk([](AdjointOp) { return WalkResult::interrupt(); })
                .wasInterrupted()) {
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

        if (failed(distributeControls(rewriter, block, map, currentCtrlQubits, ctrlValues))) {
            return failure();
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
