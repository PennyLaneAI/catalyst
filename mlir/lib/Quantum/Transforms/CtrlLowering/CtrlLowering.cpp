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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefOps.h"
#include "Quantum/Transforms/Passes.h"

using namespace mlir;

namespace catalyst::qref {

static LogicalResult distributeControls(PatternRewriter &rewriter, Block &block,
                                        SmallVector<Value> &currentCtrlQubits,
                                        ValueRange ctrlValues,
                                        SmallVector<Operation *> &opsToErase);

/// Read a segment-sizes attribute from an operation and return it as a SmallVector<int32_t>. The
/// attribute is expected to be a DenseI32ArrayAttr, and the returned vector contains the sizes of
/// the operand/result segments in order.
static SmallVector<int32_t> readSegmentSizes(Operation *op, StringRef name) {
    auto seg = op->getAttrOfType<DenseI32ArrayAttr>(name);
    return SmallVector<int32_t>(seg.asArrayRef().begin(), seg.asArrayRef().end());
}

/// Rebuild a qref.quantum.gate with additional control qubits/values appended to whatever controls
/// it already carries. The new op is inserted at the rewriter's insertion point.
void createControlledGate(PatternRewriter &rewriter, QuantumGate gate, ValueRange addCtrlQubits,
                          ValueRange addCtrlValues) {
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
        operands.push_back(op->getOperand(i));
    }
    for (Value q : nonCtrlQubits) {
        operands.push_back(q);
    }
    for (Value q : oldCtrlQubits) {
        operands.push_back(q);
    }
    operands.append(addCtrlQubits.begin(), addCtrlQubits.end());
    for (Value v : oldCtrlValues) {
        operands.push_back(v);
    }
    operands.append(addCtrlValues.begin(), addCtrlValues.end());

    // The control operands/results are NOT necessarily the trailing segments: `quantum.operator`
    // declares `in_qreg`/`arr_*` operand segments and an `out_qreg` result segment *after* the
    // control segments. Assembling the operand vector above (leading, non-ctrl qubits, controls)
    // and appending the new control results at the end is only correct when those trailing
    // segments are empty (i.e. qubit-mode gates). Bail out cleanly on register-mode gates rather
    // than silently miscompiling.
    unsigned numTrailing = op->getNumOperands() - numLeading - nonCtrlQubits.size() -
                           oldCtrlQubits.size() - oldCtrlValues.size();
    if (numTrailing != 0) {
        op->emitError("cannot control a register-mode quantum operation inside a qref.ctrl "
                      "region");
        return;
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(op->getResultTypes());
    for (NamedAttribute attr : op->getAttrs()) {
        StringRef attrName = attr.getName().strref();
        if (attrName == "operandSegmentSizes") {
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

    rewriter.create(state);
}

static LogicalResult distributeControls(PatternRewriter &rewriter, Block &block,
                                        SmallVector<Value> &currentCtrlQubits,
                                        ValueRange ctrlValues,
                                        SmallVector<Operation *> &opsToErase) {
    for (Operation &op : block.without_terminator()) {
        if (auto gate = dyn_cast<QuantumGate>(op)) {
            rewriter.setInsertionPoint(&op);
            createControlledGate(rewriter, gate, currentCtrlQubits, ctrlValues);
            opsToErase.push_back(&op);
            continue;
        }
        if (auto inner = dyn_cast<CtrlOp>(op)) {
            rewriter.modifyOpInPlace(inner, [&] {
                inner.getCtrlQubitsMutable().append(currentCtrlQubits);
                inner.getCtrlValuesMutable().append(ctrlValues);
            });
            continue;
        }
        if (isa<scf::IfOp, scf::ForOp, scf::WhileOp, scf::IndexSwitchOp>(op)) {
            for (Region &region : op.getRegions()) {
                if (!region.empty()) {
                    if (failed(distributeControls(rewriter, region.front(), currentCtrlQubits,
                                                  ctrlValues, opsToErase))) {
                        return failure();
                    }
                }
            }
            continue;
        }

        if (isa<GetOp, AllocOp, DeallocOp, AllocQubitOp, DeallocQubitOp>(op)) {
            // Structural ops carry no controls; thread their operands/results through the map.
            continue;
        }
        if (isa<QRefDialect>(op.getDialect())) {
            op.emitError("unsupported qref operation inside a qref.ctrl region");
            return failure();
        }
        // Any other scf ops would need their body controlled too,
        // which is not supported:
        if (isa<scf::SCFDialect>(op.getDialect()) && op.getNumRegions() > 0) {
            op.emitError("unsupported scf operation inside a qref.ctrl region");
            return failure();
        }
    }
    return success();
}

// match and rewrite a qref.ctrl op with reference semantics
// this needs to take in a qref.ctrl op and output qref.custum op
struct CtrlLoweringRewritePattern : public OpRewritePattern<CtrlOp> {
    using OpRewritePattern<CtrlOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CtrlOp ctrl, PatternRewriter &rewriter) const override {
        Block &block = ctrl.getRegion().front();

        // Defer (not an error) if the region still contains a nested quantum.adjoint region.
        // Distributing controls needs an op-level body, so the inner region must be reduced first.
        // The pipeline runs (ctrl-lowering, adjoint-lowering) to a fixpoint: adjoint-lowering
        // reduces the inner region to op-level gates, then this ctrl op lowers on a later
        // iteration. A pre-scan avoids a partial rewrite (creating ops, then bailing out
        // mid-region).
        for (Operation &op : block.without_terminator()) {
            if (isa<MeasureOp>(op)) {
                op.emitError("cannot control a measurement inside a qref.ctrl region");
                return failure();
            }
            if (isa<AdjointOp>(op)) {
                return failure();
            }
        }

        // The control qubits are threaded through every enclosed gate; the control values are
        // constant for the whole region.
        SmallVector<Value> currentCtrlQubits(ctrl.getCtrlQubits().begin(),
                                             ctrl.getCtrlQubits().end());
        ValueRange ctrlValues = ctrl.getCtrlValues();
        SmallVector<Operation *> opsToErase;

        // MUTATION: Now that we know the block is safe, perform the lowering.
        if (failed(
                distributeControls(rewriter, block, currentCtrlQubits, ctrlValues, opsToErase))) {
            return failure();
        }

        for (Operation *op : opsToErase) {
            rewriter.eraseOp(op);
        }
        rewriter.inlineBlockBefore(&block, ctrl);
        // Assemble the ctrl op results: out_ctrl_qubits followed by the target results.
        rewriter.eraseOp(ctrl);
        return success();
    }
};

} // namespace catalyst::qref

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_CTRLLOWERINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct CtrlLoweringPass : impl::CtrlLoweringPassBase<CtrlLoweringPass> {
    using CtrlLoweringPassBase::CtrlLoweringPassBase;

    void runOnOperation() final {
        Operation *op = getOperation();
        // Convert to reference-semantics
        {
            OpPassManager ReferenceSemanticsPm(op->getName());
            ReferenceSemanticsPm.addPass(createReferenceSemanticsConversionPass());
            if (failed(runPipeline(ReferenceSemanticsPm, op))) {
                return signalPassFailure();
            }
        }

        RewritePatternSet patterns(&getContext());
        patterns.add<qref::CtrlLoweringRewritePattern>(patterns.getContext(), 1);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace quantum
} // namespace catalyst
