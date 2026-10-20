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

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"
#include "Quantum/Transforms/Patterns.h"

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

static StringRef mergeWorkWireTypes(StringRef lhs, StringRef rhs) {
    // borrowed takes precedence
    return lhs == "borrowed" || rhs == "borrowed" ? "borrowed" : "zeroed";
}

static DictionaryAttr appendStaticData(OpBuilder &builder, DictionaryAttr staticData,
                                       StringRef workWireType) {
    NamedAttrList entries(staticData);
    entries.set("ctrl_work_wire_type", builder.getStringAttr(workWireType));
    return entries.getDictionary(builder.getContext());
}

static DictionaryAttr appendCtrlWorkQubitMap(OpBuilder &builder, DictionaryAttr qubitMap,
                                             unsigned oldNumQubits, unsigned numWorkQubits) {
    NamedAttrList entries(qubitMap);
    SmallVector<int64_t> workIndices;
    if (auto existing = dyn_cast_or_null<DenseI64ArrayAttr>(entries.get("ctrl_work_wires"))) {
        workIndices.append(existing.asArrayRef().begin(), existing.asArrayRef().end());
    }
    for (unsigned i = 0; i < numWorkQubits; ++i) {
        workIndices.push_back(oldNumQubits + i);
    }
    entries.set("ctrl_work_wires", builder.getDenseI64ArrayAttr(workIndices));
    return entries.getDictionary(builder.getContext());
}

static DictionaryAttr buildParamMap(OpBuilder &builder, DecomposableGate gate) {
    SmallVector<NamedAttribute> entries;
    unsigned offset = 0;
    for (const auto &[name, types] : gate.getDynamicShape()) {
        SmallVector<int64_t> indices;
        for (unsigned i = 0; i < types.size(); ++i) {
            indices.push_back(offset++);
        }
        entries.emplace_back(builder.getStringAttr(name), builder.getDenseI64ArrayAttr(indices));
    }
    return builder.getDictionaryAttr(entries);
}

static DictionaryAttr buildQubitMap(OpBuilder &builder, DecomposableGate gate) {
    SmallVector<NamedAttribute> entries;
    unsigned offset = 0;
    for (const auto &[name, numWires] : gate.getWireLens()) {
        SmallVector<int64_t> indices;
        for (unsigned i = 0; i < numWires; ++i) {
            indices.push_back(offset++);
        }
        entries.emplace_back(builder.getStringAttr(name), builder.getDenseI64ArrayAttr(indices));
    }
    assert(offset == gate.getNonCtrlQubitOperands().size() &&
           "wire lengths must cover every non-control qubit");
    return builder.getDictionaryAttr(entries);
}

/// Rebuild a gate with additional controls. Work wires require quantum.operator's named wire
/// groups; without them, preserve the original gate operation.
static Operation *createControlledGate(PatternRewriter &rewriter, QuantumGate gate, IRMapping &map,
                                       ValueRange addCtrlQubits, ValueRange addCtrlValues,
                                       ValueRange addWorkQubits, StringRef workWireType) {
    Operation *op = gate.getOperation();
    auto operatorOp = dyn_cast<OperatorOp>(op);

    // Register addressing cannot absorb the individual control/work qubits threaded here.
    if (operatorOp && operatorOp.getInQreg()) {
        op->emitError("cannot control a register-mode quantum operation inside a quantum.ctrl "
                      "region");
        return nullptr;
    }

    ValueRange nonCtrlQubits = gate.getNonCtrlQubitOperands();
    ValueRange oldCtrlQubits = gate.getCtrlQubitOperands();
    ValueRange oldCtrlValues = gate.getCtrlValueOperands();

    SmallVector<Value> qubits;
    for (Value value : nonCtrlQubits) {
        qubits.push_back(map.lookupOrDefault(value));
    }

    SmallVector<Value> ctrlQubits;
    for (Value value : oldCtrlQubits) {
        ctrlQubits.push_back(map.lookupOrDefault(value));
    }
    ctrlQubits.append(addCtrlQubits.begin(), addCtrlQubits.end());

    SmallVector<Value> ctrlValues;
    for (Value value : oldCtrlValues) {
        ctrlValues.push_back(map.lookupOrDefault(value));
    }
    ctrlValues.append(addCtrlValues.begin(), addCtrlValues.end());

    if (!addWorkQubits.empty()) {
        auto decomposable = dyn_cast<DecomposableGate>(op);
        if (!decomposable) {
            op->emitError(
                "a gate controlled with work wires must implement the DecomposableGate interface.");
            return nullptr;
        }

        SmallVector<Value> params;
        if (operatorOp) {
            for (Value value : operatorOp.getParams()) {
                params.push_back(map.lookupOrDefault(value));
            }
        } else if (auto parametrized = dyn_cast<ParametrizedGate>(op)) {
            for (Value value : parametrized.getAllParams()) {
                params.push_back(map.lookupOrDefault(value));
            }
        }

        unsigned oldNumQubits = qubits.size();
        qubits.append(addWorkQubits.begin(), addWorkQubits.end());

        SmallVector<Value> forwardArgs;
        DictionaryAttr paramMap;
        DictionaryAttr qubitMap;
        std::optional<int64_t> uid;
        if (operatorOp) {
            for (Value value : operatorOp.getForwardArgs()) {
                forwardArgs.push_back(map.lookupOrDefault(value));
            }
            paramMap = operatorOp.getParamMapAttr();
            qubitMap = operatorOp.getQubitMap();
            if (!operatorOp.getQubitMapAttr()) {
                SmallVector<int64_t> indices;
                for (unsigned i = 0; i < oldNumQubits; ++i) {
                    indices.push_back(i);
                }
                qubitMap = rewriter.getDictionaryAttr(
                    {rewriter.getNamedAttr("wires", rewriter.getDenseI64ArrayAttr(indices))});
            }
            uid = operatorOp.getUID();
            if (auto existing = dyn_cast_or_null<DenseI64ArrayAttr>(
                    operatorOp.getQubitMap().get("ctrl_work_wires"));
                existing && !existing.empty()) {
                auto oldWorkWireType =
                    cast<StringAttr>(operatorOp.getStaticData().get("ctrl_work_wire_type"));
                workWireType = mergeWorkWireTypes(oldWorkWireType.getValue(), workWireType);
            }
        } else {
            paramMap = buildParamMap(rewriter, decomposable);
            qubitMap = buildQubitMap(rewriter, decomposable);
        }

        // Generic work-wire groups stay separate; extend only the control work-wire group.
        DictionaryAttr staticData =
            appendStaticData(rewriter, decomposable.getStaticData(), workWireType);
        qubitMap = appendCtrlWorkQubitMap(rewriter, qubitMap, oldNumQubits, addWorkQubits.size());

        return OperatorOp::create(rewriter, op->getLoc(), decomposable.getOperatorName(), params,
                                  qubits, ctrlQubits, ctrlValues, forwardArgs,
                                  gate.getAdjointFlag(), uid, staticData, paramMap, qubitMap);
    }

    // Everything before the (non-ctrl) qubit operands is classical data (params, matrices, angles).
    unsigned numLeading =
        op->getNumOperands() - nonCtrlQubits.size() - oldCtrlQubits.size() - oldCtrlValues.size();

    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands() + addCtrlQubits.size() + addCtrlValues.size());
    for (unsigned i = 0; i < numLeading; ++i) {
        operands.push_back(map.lookupOrDefault(op->getOperand(i)));
    }
    operands.append(qubits.begin(), qubits.end());
    operands.append(ctrlQubits.begin(), ctrlQubits.end());
    operands.append(ctrlValues.begin(), ctrlValues.end());

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

    // By ODS convention a QuantumGate lays its operands out as [<classical>, (in_qubits,)
    // in_ctrl_qubits, in_ctrl_values, <trailing>], with in_ctrl_values immediately following
    // in_ctrl_qubits (some gates, e.g. `quantum.gphase`, have no in_qubits segment at all).
    // Locate the in_ctrl_qubits segment by its operand start offset -- the classical operands
    // (`numLeading`) plus the non-ctrl qubits -- rather than assuming it is a trailing segment.
    SmallVector<int32_t> operandSegments = readSegmentSizes(op, "operandSegmentSizes");
    int32_t ctrlQubitsStart = static_cast<int32_t>(numLeading + nonCtrlQubits.size());
    unsigned ctrlQubitsSeg = 0;
    for (int32_t acc = 0; ctrlQubitsSeg < operandSegments.size(); ++ctrlQubitsSeg) {
        if (acc == ctrlQubitsStart) {
            break;
        }
        acc += operandSegments[ctrlQubitsSeg];
    }
    unsigned ctrlValuesSeg = ctrlQubitsSeg + 1;
    assert(ctrlValuesSeg < operandSegments.size() &&
           operandSegments[ctrlQubitsSeg] == static_cast<int32_t>(oldCtrlQubits.size()) &&
           operandSegments[ctrlValuesSeg] == static_cast<int32_t>(oldCtrlValues.size()) &&
           "unexpected control-operand segment layout");
    operandSegments[ctrlQubitsSeg] += static_cast<int32_t>(addCtrlQubits.size());
    operandSegments[ctrlValuesSeg] += static_cast<int32_t>(addCtrlValues.size());
    state.addAttribute("operandSegmentSizes", rewriter.getDenseI32ArrayAttr(operandSegments));

    if (op->getAttrOfType<DenseI32ArrayAttr>("resultSegmentSizes")) {
        // Results follow the pattern out_qubits, out_ctrl_qubits, [out_qreg]; out_ctrl_qubits is
        // always the second result segment (index 1), which is not necessarily the last one.
        SmallVector<int32_t> resultSegments = readSegmentSizes(op, "resultSegmentSizes");
        constexpr unsigned outCtrlQubitsSeg = 1;
        assert(outCtrlQubitsSeg < resultSegments.size() &&
               resultSegments[outCtrlQubitsSeg] == static_cast<int32_t>(oldCtrlQubits.size()) &&
               "unexpected control-result segment layout");
        resultSegments[outCtrlQubitsSeg] += static_cast<int32_t>(addCtrlQubits.size());
        state.addAttribute("resultSegmentSizes", rewriter.getDenseI32ArrayAttr(resultSegments));
    }

    return rewriter.create(state);
}

/// Rebuild a nested `quantum.ctrl` op with the enclosing controls merged in.
static CtrlOp mergeNestedCtrl(PatternRewriter &rewriter, CtrlOp inner, IRMapping &map,
                              ValueRange addCtrlQubits, ValueRange addCtrlValues,
                              ValueRange addWorkQubits, StringRef addWorkWireType) {
    Location loc = inner.getLoc();

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

    SmallVector<Value> mergedWorkQubits;
    for (Value q : inner.getInWorkQubits()) {
        mergedWorkQubits.push_back(map.lookupOrDefault(q));
    }
    mergedWorkQubits.append(addWorkQubits.begin(), addWorkQubits.end());

    SmallVector<Value> innerArgs;
    for (Value a : inner.getArgs()) {
        innerArgs.push_back(map.lookupOrDefault(a));
    }

    StringRef mergedWorkWireType =
        inner.getInWorkQubits().empty()
            ? addWorkWireType
            : (addWorkQubits.empty()
                   ? inner.getWorkWireType()
                   : mergeWorkWireTypes(inner.getWorkWireType(), addWorkWireType));
    auto merged =
        CtrlOp::create(rewriter, loc, TypeRange(mergedCtrlQubits), TypeRange(mergedWorkQubits),
                       inner.getResults().getTypes(), mergedCtrlQubits, mergedCtrlValues,
                       mergedWorkQubits, mergedWorkWireType, innerArgs);

    // Move the nested region body into the freshly created op (its block arguments, the target
    // qubits, are unaffected by adding controls).
    rewriter.inlineRegionBefore(inner.getRegion(), merged.getRegion(), merged.getRegion().end());
    return merged;
}

/// Control every op in `block` (excluding its terminator), threading controls and work qubits
/// through `currentQubits` and recording result mappings in `map`. The first ctrlValues.size()
/// entries are controls; the remaining entries are work qubits. New ops are created at the
/// rewriter's insertion point.
static LogicalResult distributeControls(PatternRewriter &rewriter, Block &block, IRMapping &map,
                                        SmallVector<Value> &currentQubits, ValueRange ctrlValues,
                                        StringRef workWireType);

/// Thread controls and work qubits through each `scf.if` branch as extra results.
static LogicalResult controlScfIf(PatternRewriter &rewriter, scf::IfOp ifOp, IRMapping &map,
                                  SmallVector<Value> &currentQubits, ValueRange ctrlValues,
                                  StringRef workWireType) {
    Type qubitType = QubitType::get(rewriter.getContext());
    unsigned numThreadedQubits = currentQubits.size();

    // New results = the if's original results, plus the threaded controls and work qubits.
    SmallVector<Type> resultTypes(ifOp.getResultTypes().begin(), ifOp.getResultTypes().end());
    resultTypes.append(numThreadedQubits, qubitType);

    Value cond = map.lookupOrDefault(ifOp.getCondition());
    auto newIf = scf::IfOp::create(rewriter, ifOp.getLoc(), resultTypes, cond,
                                   /*withElseRegion=*/true);
    // Carry over hints such as `catalyst.estimated_probability`.
    newIf->setDiscardableAttrs(ifOp->getDiscardableAttrDictionary());

    // Control one branch: `oldBlock` may be null (a missing else), in which case the branch just
    // threads the incoming controls and work qubits through unchanged.
    auto controlBranch = [&](Block *oldBlock, Block *newBlock) -> LogicalResult {
        IRMapping branchMap = map; // copy: branch-local mappings must not leak across branches
        SmallVector<Value> branchQubits(currentQubits.begin(), currentQubits.end());
        rewriter.setInsertionPointToStart(newBlock);

        SmallVector<Value> yielded;
        Location yieldLoc = ifOp.getLoc();
        if (oldBlock) {
            if (failed(distributeControls(rewriter, *oldBlock, branchMap, branchQubits, ctrlValues,
                                          workWireType))) {
                return failure();
            }
            auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
            yieldLoc = oldYield.getLoc();
            for (Value v : oldYield.getOperands()) {
                yielded.push_back(branchMap.lookupOrDefault(v));
            }
        }
        yielded.append(branchQubits.begin(), branchQubits.end());
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

    // Map the original results one-to-one; the trailing results are the threaded controls and work
    // qubits.
    unsigned numOrig = ifOp.getNumResults();
    for (unsigned i = 0; i < numOrig; ++i) {
        map.map(ifOp.getResult(i), newIf.getResult(i));
    }
    currentQubits.assign(newIf.getResults().begin() + numOrig, newIf.getResults().end());

    // Restore the insertion point to after the new op so the enclosing walk keeps appending there
    // (controlBranch left it inside the else block).
    rewriter.setInsertionPointAfter(newIf);
    return success();
}

/// Thread controls and work qubits through `scf.for` as extra loop-carried `iter_args`.
static LogicalResult controlScfFor(PatternRewriter &rewriter, scf::ForOp forOp, IRMapping &map,
                                   SmallVector<Value> &currentQubits, ValueRange ctrlValues,
                                   StringRef workWireType) {
    unsigned numOrig = forOp.getInitArgs().size();

    // New init args = the loop's original init args, plus the incoming controls and work qubits.
    SmallVector<Value> newInits;
    for (Value init : forOp.getInitArgs()) {
        newInits.push_back(map.lookupOrDefault(init));
    }
    newInits.append(currentQubits.begin(), currentQubits.end());

    Value lb = map.lookupOrDefault(forOp.getLowerBound());
    Value ub = map.lookupOrDefault(forOp.getUpperBound());
    Value step = map.lookupOrDefault(forOp.getStep());

    // With non-empty iter args and no body-builder, scf.for creates the body block (induction var +
    // iter-arg block args) without a terminator, which we fill in below.
    auto newFor = scf::ForOp::create(rewriter, forOp.getLoc(), lb, ub, step, newInits);
    // Carry over hints such as `catalyst.estimated_iterations`.
    newFor->setDiscardableAttrs(forOp->getDiscardableAttrDictionary());
    Block *newBody = newFor.getBody();
    ValueRange newIterArgs = newFor.getRegionIterArgs();

    // Seed a body-local mapping; the trailing iter args are this iteration's controls and work
    // qubits.
    IRMapping bodyMap = map;
    bodyMap.map(forOp.getInductionVar(), newFor.getInductionVar());
    for (unsigned i = 0; i < numOrig; ++i) {
        bodyMap.map(forOp.getRegionIterArg(i), newIterArgs[i]);
    }
    SmallVector<Value> bodyQubits(newIterArgs.begin() + numOrig, newIterArgs.end());

    rewriter.setInsertionPointToStart(newBody);
    Block &oldBody = forOp.getRegion().front();
    if (failed(
            distributeControls(rewriter, oldBody, bodyMap, bodyQubits, ctrlValues, workWireType))) {
        return failure();
    }

    auto oldYield = cast<scf::YieldOp>(oldBody.getTerminator());
    SmallVector<Value> yielded;
    for (Value v : oldYield.getOperands()) {
        yielded.push_back(bodyMap.lookupOrDefault(v));
    }
    yielded.append(bodyQubits.begin(), bodyQubits.end());
    rewriter.setInsertionPointToEnd(newBody);
    scf::YieldOp::create(rewriter, oldYield.getLoc(), yielded);

    // Map the original results one-to-one; the trailing results are the threaded controls and work
    // qubits.
    for (unsigned i = 0; i < numOrig; ++i) {
        map.map(forOp.getResult(i), newFor.getResult(i));
    }
    currentQubits.assign(newFor.getResults().begin() + numOrig, newFor.getResults().end());

    // Restore the insertion point to after the new op so the enclosing walk keeps appending there.
    rewriter.setInsertionPointAfter(newFor);
    return success();
}

/// Thread controls and work qubits through `scf.while` as extra loop-carried values.
static LogicalResult controlScfWhile(PatternRewriter &rewriter, scf::WhileOp whileOp,
                                     IRMapping &map, SmallVector<Value> &currentQubits,
                                     ValueRange ctrlValues, StringRef workWireType) {
    Type qubitType = QubitType::get(rewriter.getContext());
    unsigned numThreadedQubits = currentQubits.size();
    unsigned numInit = whileOp.getInits().size();

    // New inits = original inits, plus the incoming controls and work qubits.
    SmallVector<Value> newInits;
    for (Value init : whileOp.getInits()) {
        newInits.push_back(map.lookupOrDefault(init));
    }
    newInits.append(currentQubits.begin(), currentQubits.end());

    SmallVector<Type> newResultTypes(whileOp.getResultTypes().begin(),
                                     whileOp.getResultTypes().end());
    newResultTypes.append(numThreadedQubits, qubitType);

    Block &oldBefore = whileOp.getBefore().front();
    Block &oldAfter = whileOp.getAfter().front();
    scf::ConditionOp oldCond = whileOp.getConditionOp();
    auto oldYield = cast<scf::YieldOp>(oldAfter.getTerminator());
    // For the results: the `before` block forwards them and the `after` block
    // receives them, and they all surface as the while results when the condition
    // is false.
    unsigned numFwd = oldCond.getArgs().size();

    LogicalResult status = success();
    // The before/after builder overload creates the before block args (init types + threaded
    // qubits) and after block args (result types + threaded qubits); the callbacks fill each
    // region.
    auto newWhile = scf::WhileOp::create(
        rewriter, whileOp.getLoc(), newResultTypes, newInits,
        [&](OpBuilder & /*builder*/, Location /*loc*/, ValueRange beforeArgs) {
            IRMapping beforeMap = map;
            for (unsigned i = 0; i < numInit; ++i) {
                beforeMap.map(oldBefore.getArgument(i), beforeArgs[i]);
            }
            SmallVector<Value> beforeQubits(beforeArgs.begin() + numInit, beforeArgs.end());
            if (failed(distributeControls(rewriter, oldBefore, beforeMap, beforeQubits, ctrlValues,
                                          workWireType))) {
                status = failure();
                return;
            }
            Value cond = beforeMap.lookupOrDefault(oldCond.getCondition());
            SmallVector<Value> fwd;
            for (Value v : oldCond.getArgs()) {
                fwd.push_back(beforeMap.lookupOrDefault(v));
            }
            fwd.append(beforeQubits.begin(), beforeQubits.end());
            scf::ConditionOp::create(rewriter, oldCond.getLoc(), cond, fwd);
        },
        [&](OpBuilder & /*builder*/, Location /*loc*/, ValueRange afterArgs) {
            IRMapping afterMap = map;
            for (unsigned i = 0; i < numFwd; ++i) {
                afterMap.map(oldAfter.getArgument(i), afterArgs[i]);
            }
            SmallVector<Value> afterQubits(afterArgs.begin() + numFwd, afterArgs.end());
            if (failed(distributeControls(rewriter, oldAfter, afterMap, afterQubits, ctrlValues,
                                          workWireType))) {
                status = failure();
                return;
            }
            SmallVector<Value> yielded;
            for (Value v : oldYield.getOperands()) {
                yielded.push_back(afterMap.lookupOrDefault(v));
            }
            yielded.append(afterQubits.begin(), afterQubits.end());
            scf::YieldOp::create(rewriter, oldYield.getLoc(), yielded);
        });
    // Carry over hints such as `catalyst.estimated_iterations`.
    newWhile->setDiscardableAttrs(whileOp->getDiscardableAttrDictionary());

    if (failed(status)) {
        return failure();
    }

    // Map the original results one-to-one; the trailing results are the threaded controls and work
    // qubits.
    unsigned numOrig = whileOp.getNumResults();
    for (unsigned i = 0; i < numOrig; ++i) {
        map.map(whileOp.getResult(i), newWhile.getResult(i));
    }
    currentQubits.assign(newWhile.getResults().begin() + numOrig, newWhile.getResults().end());
    rewriter.setInsertionPointAfter(newWhile);
    return success();
}

/// Thread controls and work qubits through `scf.index_switch` as extra results.
static LogicalResult controlScfIndexSwitch(PatternRewriter &rewriter, scf::IndexSwitchOp switchOp,
                                           IRMapping &map, SmallVector<Value> &currentQubits,
                                           ValueRange ctrlValues, StringRef workWireType) {
    Type qubitType = QubitType::get(rewriter.getContext());
    unsigned numThreadedQubits = currentQubits.size();

    SmallVector<Type> resultTypes(switchOp.getResultTypes().begin(),
                                  switchOp.getResultTypes().end());
    resultTypes.append(numThreadedQubits, qubitType);

    Value arg = map.lookupOrDefault(switchOp.getArg());
    SmallVector<int64_t> cases(switchOp.getCases().begin(), switchOp.getCases().end());
    auto newSwitch = scf::IndexSwitchOp::create(rewriter, switchOp.getLoc(), resultTypes, arg,
                                                cases, cases.size());
    // Carry over hints such as `catalyst.estimated_probabilities`.
    newSwitch->setDiscardableAttrs(switchOp->getDiscardableAttrDictionary());

    // Control one region.
    // Regions have no block arguments, so bodies reference outer
    // values via `map`, as with scf.if branches.
    auto controlCase = [&](Block &oldBlock, Region &newRegion) -> LogicalResult {
        IRMapping caseMap = map;
        SmallVector<Value> caseQubits(currentQubits.begin(), currentQubits.end());
        Block *newBlock = rewriter.createBlock(&newRegion);
        if (failed(distributeControls(rewriter, oldBlock, caseMap, caseQubits, ctrlValues,
                                      workWireType))) {
            return failure();
        }
        auto oldYield = cast<scf::YieldOp>(oldBlock.getTerminator());
        SmallVector<Value> yielded;
        for (Value v : oldYield.getOperands()) {
            yielded.push_back(caseMap.lookupOrDefault(v));
        }
        yielded.append(caseQubits.begin(), caseQubits.end());
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
    currentQubits.assign(newSwitch.getResults().begin() + numOrig, newSwitch.getResults().end());
    rewriter.setInsertionPointAfter(newSwitch);
    return success();
}

static LogicalResult distributeControls(PatternRewriter &rewriter, Block &block, IRMapping &map,
                                        SmallVector<Value> &currentQubits, ValueRange ctrlValues,
                                        StringRef workWireType) {
    unsigned numCtrlQubits = ctrlValues.size();
    for (Operation &op : block.without_terminator()) {
        if (isa<MeasurementProcess, MeasureOp>(op)) {
            op.emitError("cannot control a measurement inside a quantum.ctrl region");
            return failure();
        }
        if (auto gate = dyn_cast<QuantumGate>(op)) {
            unsigned numOldControls = gate.getCtrlQubitOperands().size();
            unsigned numOldQubits = gate.getNonCtrlQubitOperands().size();
            ValueRange addCtrlQubits = ValueRange(currentQubits).take_front(numCtrlQubits);
            ValueRange addWorkQubits = ValueRange(currentQubits).drop_front(numCtrlQubits);
            Operation *newOp = createControlledGate(rewriter, gate, map, addCtrlQubits, ctrlValues,
                                                    addWorkQubits, workWireType);
            if (!newOp) {
                return failure();
            }
            auto newGate = cast<QuantumGate>(newOp);

            ResultRange newNonCtrlResults = newGate.getNonCtrlQubitResults();
            ResultRange oldNonCtrlResults = gate.getNonCtrlQubitResults();
            assert(oldNonCtrlResults.size() == numOldQubits &&
                   newNonCtrlResults.size() >= numOldQubits &&
                   "unexpected non-control result count");
            for (unsigned i = 0; i < numOldQubits; ++i) {
                map.map(oldNonCtrlResults[i], newNonCtrlResults[i]);
            }
            // The new control results are [old controls ..., threaded region controls ...].
            ResultRange newCtrlResults = newGate.getCtrlQubitResults();
            for (unsigned i = 0; i < numOldControls; ++i) {
                map.map(gate.getCtrlQubitResults()[i], newCtrlResults[i]);
            }
            currentQubits.assign(newCtrlResults.begin() + numOldControls, newCtrlResults.end());
            currentQubits.append(newNonCtrlResults.begin() + numOldQubits, newNonCtrlResults.end());
            continue;
        }
        if (auto inner = dyn_cast<CtrlOp>(op)) {
            unsigned numInnerControls = inner.getInCtrlQubits().size();
            unsigned numInnerWork = inner.getInWorkQubits().size();
            ValueRange addCtrlQubits = ValueRange(currentQubits).take_front(numCtrlQubits);
            ValueRange addWorkQubits = ValueRange(currentQubits).drop_front(numCtrlQubits);
            CtrlOp merged = mergeNestedCtrl(rewriter, inner, map, addCtrlQubits, ctrlValues,
                                            addWorkQubits, workWireType);

            ResultRange mergedCtrlResults = merged.getOutCtrlQubits();
            for (unsigned i = 0; i < numInnerControls; ++i) {
                map.map(inner.getOutCtrlQubits()[i], mergedCtrlResults[i]);
            }
            ResultRange mergedWorkResults = merged.getOutWorkQubits();
            for (unsigned i = 0; i < numInnerWork; ++i) {
                map.map(inner.getOutWorkQubits()[i], mergedWorkResults[i]);
            }
            for (auto [oldResult, newResult] :
                 llvm::zip_equal(inner.getResults(), merged.getResults())) {
                map.map(oldResult, newResult);
            }
            currentQubits.assign(mergedCtrlResults.begin() + numInnerControls,
                                 mergedCtrlResults.end());
            currentQubits.append(mergedWorkResults.begin() + numInnerWork, mergedWorkResults.end());
            continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            if (failed(
                    controlScfIf(rewriter, ifOp, map, currentQubits, ctrlValues, workWireType))) {
                return failure();
            }
            continue;
        }
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            if (failed(
                    controlScfFor(rewriter, forOp, map, currentQubits, ctrlValues, workWireType))) {
                return failure();
            }
            continue;
        }
        if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
            if (failed(controlScfWhile(rewriter, whileOp, map, currentQubits, ctrlValues,
                                       workWireType))) {
                return failure();
            }
            continue;
        }
        if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
            if (failed(controlScfIndexSwitch(rewriter, switchOp, map, currentQubits, ctrlValues,
                                             workWireType))) {
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
        if (isa<scf::SCFDialect>(op.getDialect()) && op.getNumRegions() > 0) {
            op.emitError("unsupported scf operation inside a quantum.ctrl region");
            return failure();
        }
        // Cloning a call that acts on quantum values would drop the controls for its body.
        auto isQuantum = [](Type type) { return isa<QubitType, QuregType>(type); };
        if (isa<CallOpInterface>(op) && (llvm::any_of(op.getOperandTypes(), isQuantum) ||
                                         llvm::any_of(op.getResultTypes(), isQuantum))) {
            op.emitError("cannot control a function call inside a quantum.ctrl region; decompose "
                         "the controlled operations before passes that introduce function calls, "
                         "such as gridsynth");
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
        // Returning failure() drives the greedy fixpoint: adjoint-lowering reduces the inner region
        // to op-level gates, then the greedy driver re-tries this ctrl op and it lowers. This works
        // whether the two patterns run in separate alternating passes or together in the combined
        // modifiers-lowering pass. A pre-scan avoids a partial rewrite (creating ops, then bailing
        // out mid-region).
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

        // Controls and control work qubits are threaded through every enclosed gate. They share
        // one vector for structural control flow, with controls first so they can still be split
        // from work qubits when rebuilding a gate.
        SmallVector<Value> currentQubits(ctrl.getInCtrlQubits().begin(),
                                         ctrl.getInCtrlQubits().end());
        currentQubits.append(ctrl.getInWorkQubits().begin(), ctrl.getInWorkQubits().end());
        ValueRange ctrlValues = ctrl.getInCtrlValues();

        rewriter.setInsertionPoint(ctrl);

        if (failed(distributeControls(rewriter, block, map, currentQubits, ctrlValues,
                                      ctrl.getWorkWireType()))) {
            return failure();
        }

        // Assemble the ctrl op results: controls, work qubits, then target results.
        auto yield = cast<YieldOp>(block.getTerminator());
        SmallVector<Value> results;
        results.append(currentQubits.begin(), currentQubits.end());
        for (Value retval : yield.getRetvals()) {
            results.push_back(map.lookupOrDefault(retval));
        }
        rewriter.replaceOp(ctrl, results);
        return success();
    }
};

} // namespace

void catalyst::quantum::populateCtrlLoweringPatterns(RewritePatternSet &patterns) {
    patterns.add<CtrlLoweringRewritePattern>(patterns.getContext(), 1);
}

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_CTRLLOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct CtrlLoweringPass : impl::CtrlLoweringPassBase<CtrlLoweringPass> {
    using CtrlLoweringPassBase::CtrlLoweringPassBase;

    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        populateCtrlLoweringPatterns(patterns);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
