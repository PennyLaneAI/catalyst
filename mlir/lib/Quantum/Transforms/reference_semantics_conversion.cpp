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

#define DEBUG_TYPE "reference-semantics-conversion"

#include "reference_semantics_conversion.h"

#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "MBQC/IR/MBQCOps.h"
#include "PBC/IR/PBCOps.h"
#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/IR/QRefTypes.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"

using namespace mlir;
using namespace catalyst;

// In this file, variable names like "vQubit" stand for "qubits in value semantics",
// and variable names like "rQubit" stand for "qubits in reference semantics".

namespace {

void eraseSCFYieldQuantumOperands(scf::YieldOp yieldOp)
{
    // scf.yield can yield both classical and quantum values
    // We need to only keep the classical yields from scf regions

    // Somewhat unfortunately, the operand erasure API does not have a lambda version
    // So we need to be a bit manual here
    BitVector eraseIndices(yieldOp->getNumOperands());
    for (auto [i, argType] : llvm::enumerate(yieldOp.getOperandTypes())) {
        eraseIndices[i] = isa<quantum::QuregType, quantum::QubitType>(argType);
    }
    yieldOp->eraseOperands(eraseIndices);
}

struct QubitValueTracker {
  public:
    QubitValueTracker() = default;

    void setRQreg(Value vQreg, Value rQreg)
    {
        assert(isa<quantum::QuregType>(vQreg.getType()) && "Expected quantum.reg type");
        assert(isa<qref::QuregType>(rQreg.getType()) && "Expected qref.reg type");
        if (failed(this->checkReferenceIsVisible(vQreg, rQreg))) {
            vQreg.getDefiningOp()->emitError(
                "The value semantics quantum value is referring to a quantum reference that is not "
                "visible to its scope. The reference must exist in the same scope, or a parent "
                "scope as the value.");
        }

        this->qreg_map[vQreg] = rQreg;
    }

    Value getRQreg(Value vQreg)
    {
        assert(isa<quantum::QuregType>(vQreg.getType()) && "Expected quantum.reg type");

        Value rQreg = this->qreg_map.at(vQreg);
        assert(isa<qref::QuregType>(rQreg.getType()) && "Expected qref.reg type");
        return rQreg;
    }

    void setRQubit(Value vQubit, Value rQubit)
    {
        assert(isa<quantum::QubitType>(vQubit.getType()) && "Expected quantum.bit type");
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");

        if (failed(this->checkReferenceIsVisible(vQubit, rQubit))) {
            vQubit.getDefiningOp()->emitError(
                "The value semantics quantum value is referring to a quantum reference that is not "
                "visible to its scope. The reference must exist in the same scope, or a parent "
                "scope as the value.");
        }

        this->qubit_map[vQubit] = rQubit;
    }

    Value getRQubit(Value vQubit)
    {
        assert(isa<quantum::QubitType>(vQubit.getType()) && "Expected quantum.bit type");

        Value rQubit = this->qubit_map.at(vQubit);
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        return rQubit;
    }

  private:
    llvm::DenseMap<Value, Value> qreg_map;
    llvm::DenseMap<Value, Value> qubit_map;

    // We need to make sure that any value semantics quantum Value is only referring to a reference
    // that is actually scope-visible.
    LogicalResult checkReferenceIsVisible(Value vVal, Value rVal)
    {
        Region *vRegion = vVal.getParentRegion();
        Region *rRegion = rVal.getParentRegion();
        if (!rRegion->isAncestor(vRegion)) {
            return failure();
        }
        return success();
    }
}; // struct QubitValueTracker

void cascadeMapAhead(Operation *vOp, QubitValueTracker &tracker)
{
    // Cascade the tracker for gate-like ops
    SmallVector<Value> vQuantumOperands;
    SmallVector<Value> vQuantumResults;
    for (Value v : vOp->getOperands()) {
        if (isa<quantum::QubitType, quantum::QuregType>(v.getType())) {
            vQuantumOperands.push_back(v);
        }
    }

    for (Value v : vOp->getResults()) {
        if (isa<quantum::QubitType, quantum::QuregType>(v.getType())) {
            vQuantumResults.push_back(v);
        }
    }

    for (auto [vqo, vqr] : llvm::zip_equal(vQuantumOperands, vQuantumResults)) {
        if (isa<quantum::QubitType>(vqo.getType())) {
            tracker.setRQubit(vqr, tracker.getRQubit(vqo));
        }
        else if (isa<quantum::QuregType>(vqo.getType())) {
            tracker.setRQreg(vqr, tracker.getRQreg(vqo));
        }
    }
}

template <typename OpTy>
OpTy migrateOpToReferenceSemantics(IRRewriter &builder, Operation *vOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vOp);
    Location loc = vOp->getLoc();

    // Create the new op using the generic state-based approach
    // We cannot just clone, since we are changing the op type
    OperationState state(loc, OpTy::getOperationName());

    SmallVector<Value> rOperands;
    for (Value v : vOp->getOperands()) {
        if (isa<quantum::QubitType>(v.getType())) {
            rOperands.push_back(tracker.getRQubit(v));
        }
        else if (isa<quantum::QuregType>(v.getType())) {
            rOperands.push_back(tracker.getRQreg(v));
        }
        else {
            // This branch includes classical operands
            rOperands.push_back(v);
        }
    }
    state.addOperands(rOperands);
    state.addAttributes(vOp->getAttrs());

    // Remove all quantum result values from the value semantics op
    // Need to keep classical results, e.g. measurement's i1 results
    SmallVector<Type> outTypes;
    for (Type t : vOp->getResultTypes()) {
        if (!isa<quantum::QubitType, quantum::QuregType>(t)) {
            outTypes.push_back(t);
        }
    }
    state.addTypes(outTypes);

    Operation *newOp = builder.create(state);

    if (isa<quantum::QuantumOperation, quantum::MeasureOp, pbc::PPMeasurementOp,
            mbqc::MeasureInBasisOp>(vOp)) {
        cascadeMapAhead(vOp, tracker);
    }

    return cast<OpTy>(newOp);
}

void handleAlloc(IRRewriter &builder, quantum::AllocOp vAllocOp, QubitValueTracker &tracker,
                 SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vAllocOp);

    Location loc = vAllocOp.getLoc();
    MLIRContext *ctx = vAllocOp.getContext();

    qref::AllocOp rAllocOp;
    Type qregType;
    if (vAllocOp.getNqubitsAttr().has_value()) {
        qregType = qref::QuregType::get(ctx, vAllocOp.getNqubitsAttrAttr());
        rAllocOp = qref::AllocOp::create(builder, loc, qregType, {}, vAllocOp.getNqubitsAttrAttr());
    }
    else {
        qregType = qref::QuregType::get(ctx, builder.getI64IntegerAttr(ShapedType::kDynamic));
        rAllocOp = qref::AllocOp::create(builder, loc, qregType, vAllocOp.getNqubits(), nullptr);
    }

    tracker.setRQreg(vAllocOp.getQreg(), rAllocOp.getQreg());
    erasureWorklist.push_back(vAllocOp);
}

void handleDealloc(IRRewriter &builder, quantum::DeallocOp vDeallocOp, QubitValueTracker &tracker,
                   SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vDeallocOp);
    Location loc = vDeallocOp.getLoc();

    qref::DeallocOp::create(builder, loc, tracker.getRQreg(vDeallocOp.getQreg()));

    erasureWorklist.push_back(vDeallocOp);
}

void handleAllocQubit(IRRewriter &builder, quantum::AllocQubitOp vAllocQbOp,
                      QubitValueTracker &tracker, SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vAllocQbOp);
    Location loc = vAllocQbOp.getLoc();

    auto rAllocQbOp = qref::AllocQubitOp::create(builder, loc);
    tracker.setRQubit(vAllocQbOp.getQubit(), rAllocQbOp.getQubit());
    erasureWorklist.push_back(vAllocQbOp);
}

void handleDeallocQubit(IRRewriter &builder, quantum::DeallocQubitOp vDeallocQbOp,
                        QubitValueTracker &tracker, SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vDeallocQbOp);
    Location loc = vDeallocQbOp.getLoc();

    qref::DeallocQubitOp::create(builder, loc, tracker.getRQubit(vDeallocQbOp.getQubit()));

    erasureWorklist.push_back(vDeallocQbOp);
}

// Although multiple qref.get ops can alias, multiple quantum.extract ops will definitely not alias
void handleExtract(IRRewriter &builder, quantum::ExtractOp vExtractOp, QubitValueTracker &tracker,
                   SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vExtractOp);
    Location loc = vExtractOp.getLoc();

    std::optional<uint64_t> idxAttr = vExtractOp.getIdxAttr();
    Value idxValue = vExtractOp.getIdx();
    assert((idxAttr.has_value() ^ (idxValue != nullptr)) &&
           "expected exactly one index for extract op");

    Value rQreg = tracker.getRQreg(vExtractOp.getQreg());

    Type qubitType = qref::QubitType::get(vExtractOp->getContext());
    Type i64Type = builder.getI64Type();
    qref::GetOp getOp;
    if (idxAttr.has_value()) {
        getOp = qref::GetOp::create(builder, loc, qubitType, rQreg, {},
                                    IntegerAttr::get(i64Type, idxAttr.value()));
    }
    else {
        getOp = qref::GetOp::create(builder, loc, qubitType, rQreg, idxValue, nullptr);
    }

    tracker.setRQubit(vExtractOp.getQubit(), getOp.getQubit());
    erasureWorklist.push_back(vExtractOp);
}

// Although insert ops do not materialize to anything in reference semantics, they need to be
// visited to update the tracker.
void handleInsert(quantum::InsertOp vInsertOp, QubitValueTracker &tracker,
                  SmallVector<Operation *> &erasureWorklist)
{
    tracker.setRQreg(vInsertOp.getOutQreg(), tracker.getRQreg(vInsertOp.getInQreg()));
    erasureWorklist.push_back(vInsertOp);
}

void handleGate(IRRewriter &builder, quantum::QuantumOperation vGateOp, QubitValueTracker &tracker,
                SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);

    qref::QuantumOperation rGateOp;
    Operation *_vGateOp = vGateOp.getOperation();
    if (auto vCustomOp = dyn_cast<quantum::CustomOp>(_vGateOp)) {
        rGateOp = migrateOpToReferenceSemantics<qref::CustomOp>(builder, vCustomOp, tracker);
        rGateOp->removeAttr("resultSegmentSizes");
    }
    else if (auto vPauliRotOP = dyn_cast<quantum::PauliRotOp>(_vGateOp)) {
        rGateOp = migrateOpToReferenceSemantics<qref::PauliRotOp>(builder, vPauliRotOP, tracker);
        rGateOp->removeAttr("resultSegmentSizes");
    }
    else if (auto vGPhaseOp = dyn_cast<quantum::GlobalPhaseOp>(_vGateOp)) {
        rGateOp = migrateOpToReferenceSemantics<qref::GlobalPhaseOp>(builder, vGPhaseOp, tracker);
    }
    else if (auto vMultiRZOp = dyn_cast<quantum::MultiRZOp>(_vGateOp)) {
        rGateOp = migrateOpToReferenceSemantics<qref::MultiRZOp>(builder, vMultiRZOp, tracker);
        rGateOp->removeAttr("resultSegmentSizes");
    }
    else if (auto vPCPhaseOp = dyn_cast<quantum::PCPhaseOp>(_vGateOp)) {
        rGateOp = migrateOpToReferenceSemantics<qref::PCPhaseOp>(builder, vPCPhaseOp, tracker);
        rGateOp->removeAttr("resultSegmentSizes");
    }
    else if (auto vQubitUnitaryOp = dyn_cast<quantum::QubitUnitaryOp>(_vGateOp)) {
        rGateOp =
            migrateOpToReferenceSemantics<qref::QubitUnitaryOp>(builder, vQubitUnitaryOp, tracker);
        rGateOp->removeAttr("resultSegmentSizes");
    }
    else if (auto vSetStateOp = dyn_cast<quantum::SetStateOp>(_vGateOp)) {
        rGateOp = migrateOpToReferenceSemantics<qref::SetStateOp>(builder, vSetStateOp, tracker);
    }
    else if (auto vSetBasisStateOp = dyn_cast<quantum::SetBasisStateOp>(_vGateOp)) {
        rGateOp = migrateOpToReferenceSemantics<qref::SetBasisStateOp>(builder, vSetBasisStateOp,
                                                                       tracker);
    }
    else {
        vGateOp->emitOpError("unknown gate op in quantum dialect");
    }

    erasureWorklist.push_back(vGateOp);
}

void handleMeasure(IRRewriter &builder, quantum::MeasureOp vMeasureOp, QubitValueTracker &tracker,
                   SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);

    auto rMeasureOp = migrateOpToReferenceSemantics<qref::MeasureOp>(builder, vMeasureOp, tracker);
    builder.replaceAllUsesWith(vMeasureOp.getMres(), rMeasureOp.getMres());
    erasureWorklist.push_back(vMeasureOp);
}

void handleMeasureInBasis(IRRewriter &builder, mbqc::MeasureInBasisOp vMeasureInBasisOp,
                          QubitValueTracker &tracker, SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);

    auto rMeasureInBasisOp = migrateOpToReferenceSemantics<mbqc::RefMeasureInBasisOp>(
        builder, vMeasureInBasisOp, tracker);
    builder.replaceAllUsesWith(vMeasureInBasisOp.getMres(), rMeasureInBasisOp.getMres());
    erasureWorklist.push_back(vMeasureInBasisOp);
}

void handlePPM(IRRewriter &builder, pbc::PPMeasurementOp vPPMOp, QubitValueTracker &tracker,
               SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);

    auto rPPMOp = migrateOpToReferenceSemantics<pbc::RefPPMeasurementOp>(builder, vPPMOp, tracker);

    builder.replaceAllUsesWith(vPPMOp.getMres(), rPPMOp.getMres());
    erasureWorklist.push_back(vPPMOp);
}

void handleCompbasis(IRRewriter &builder, quantum::ComputationalBasisOp vCompbasisOp,
                     QubitValueTracker &tracker)
{
    auto rCompbasisOp =
        migrateOpToReferenceSemantics<qref::ComputationalBasisOp>(builder, vCompbasisOp, tracker);
    builder.replaceOp(vCompbasisOp, rCompbasisOp);
}

void handleNamedObs(IRRewriter &builder, quantum::NamedObsOp vNamedObsOp,
                    QubitValueTracker &tracker)
{
    auto rNamedObsOp =
        migrateOpToReferenceSemantics<qref::NamedObsOp>(builder, vNamedObsOp, tracker);
    builder.replaceOp(vNamedObsOp, rNamedObsOp);
}

void handleHermitian(IRRewriter &builder, quantum::HermitianOp vHermitianOp,
                     QubitValueTracker &tracker)
{
    auto rHermitianOp =
        migrateOpToReferenceSemantics<qref::HermitianOp>(builder, vHermitianOp, tracker);
    builder.replaceOp(vHermitianOp, rHermitianOp);
}

void handleGraphStatePrep(IRRewriter &builder, mbqc::GraphStatePrepOp vGraphStatePrepOp,
                          QubitValueTracker &tracker, SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vGraphStatePrepOp);

    Location loc = vGraphStatePrepOp.getLoc();
    MLIRContext *ctx = vGraphStatePrepOp.getContext();

    size_t numQubits = vGraphStatePrepOp.getNumQubitsFromAdjMatrixSize();
    Type qregType = qref::QuregType::get(ctx, builder.getI64IntegerAttr(numQubits));

    auto rGraphStatePrepOp = mbqc::RefGraphStatePrepOp::create(
        builder, loc, qregType, vGraphStatePrepOp.getAdjMatrix(), vGraphStatePrepOp.getInitOp(),
        vGraphStatePrepOp.getEntangleOp());

    tracker.setRQreg(vGraphStatePrepOp.getQreg(), rGraphStatePrepOp.getQreg());
    erasureWorklist.push_back(vGraphStatePrepOp);
}

void handleAdjoint(IRRewriter &builder, quantum::AdjointOp vAdjointOp, QubitValueTracker &tracker,
                   SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vAdjointOp);
    Location loc = vAdjointOp->getLoc();
    QubitValueTracker regionTracker = tracker;

    // Add block args to the map
    for (auto [blockArg, operand] : llvm::zip_equal(vAdjointOp.getRegion().front().getArguments(),
                                                    vAdjointOp->getOperands())) {
        if (isa<quantum::QubitType>(blockArg.getType())) {
            regionTracker.setRQubit(blockArg, tracker.getRQubit(operand));
        }
        else if (isa<quantum::QuregType>(blockArg.getType())) {
            regionTracker.setRQreg(blockArg, tracker.getRQreg(operand));
        }
    }

    // Create the rAdjoint op and handle
    auto rAdjointOp = qref::AdjointOp::create(builder, loc);
    builder.inlineRegionBefore(vAdjointOp.getRegion(), rAdjointOp.getRegion(),
                               rAdjointOp.getRegion().end());
    handleRegion(builder, rAdjointOp.getRegion(), regionTracker);
    cascadeMapAhead(vAdjointOp, tracker);
    erasureWorklist.push_back(vAdjointOp);

    // Remove the moved over value semantics block args
    rAdjointOp.getRegion().front().eraseArguments([](BlockArgument arg) {
        return isa<quantum::QubitType, quantum::QuregType>(arg.getType());
    });
}

void handleFor(IRRewriter &builder, scf::ForOp forOp, QubitValueTracker &tracker,
               SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(forOp);
    Location loc = forOp->getLoc();
    QubitValueTracker regionTracker = tracker;

    // Add quantum block args to the map
    // Classical args must remain on the new for loop op
    SmallVector<Value> newIterArgs;
    SmallVector<unsigned> classicalIndices;
    for (auto [i, operand] : llvm::enumerate(forOp.getInitArgs())) {
        BlockArgument blockArg = forOp.getRegionIterArg(i);
        if (isa<quantum::QubitType>(blockArg.getType())) {
            regionTracker.setRQubit(blockArg, tracker.getRQubit(operand));
        }
        else if (isa<quantum::QuregType>(blockArg.getType())) {
            regionTracker.setRQreg(blockArg, tracker.getRQreg(operand));
        }
        else {
            newIterArgs.push_back(operand);
            classicalIndices.push_back(i);
        }
    }

    // Create the new for op and handle
    auto newLoop = scf::ForOp::create(builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                                      forOp.getStep(), newIterArgs);
    builder.eraseBlock(newLoop.getBody());

    builder.inlineRegionBefore(forOp.getRegion(), newLoop.getRegion(), newLoop.getRegion().end());
    eraseSCFYieldQuantumOperands(cast<scf::YieldOp>(newLoop.getRegion().front().getTerminator()));
    handleRegion(builder, newLoop.getRegion(), regionTracker);
    cascadeMapAhead(forOp, tracker);
    erasureWorklist.push_back(forOp);

    // Remove the moved over value semantics block args
    newLoop.getRegion().front().eraseArguments([](BlockArgument arg) {
        return isa<quantum::QubitType, quantum::QuregType>(arg.getType());
    });

    // Update use of classical results
    for (auto [oldResultIdx, newResult] :
         llvm::zip_equal(classicalIndices, newLoop->getResults())) {
        builder.replaceAllUsesWith(forOp->getResult(oldResultIdx), newResult);
    }
}

void handleWhile(IRRewriter &builder, scf::WhileOp whileOp, QubitValueTracker &tracker,
                 SmallVector<Operation *> &erasureWorklist)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(whileOp);
    Location loc = whileOp->getLoc();

    // Add quantum block args to the map
    // Classical args must remain on the new while loop op
    QubitValueTracker beforeRegionTracker = tracker;
    SmallVector<Value> newBeforeArgs;
    for (auto [blockArg, operand] :
         llvm::zip_equal(whileOp.getBeforeArguments(), whileOp.getInits())) {
        if (isa<quantum::QubitType>(operand.getType())) {
            beforeRegionTracker.setRQubit(blockArg, tracker.getRQubit(operand));
        }
        else if (isa<quantum::QuregType>(operand.getType())) {
            beforeRegionTracker.setRQreg(blockArg, tracker.getRQreg(operand));
        }
        else {
            newBeforeArgs.push_back(operand);
        }
    }

    // Collect new while op results. Only classical results remain.
    // Note that while loop op's result and operand types don't necessarily match
    SmallVector<Type> newResultTypes;
    SmallVector<unsigned> oldClassicalResultIndices;
    for (auto [i, res] : llvm::enumerate(whileOp->getResults())) {
        Type t = res.getType();
        if (!isa<quantum::QubitType, quantum::QuregType>(t)) {
            newResultTypes.push_back(t);
            oldClassicalResultIndices.push_back(i);
        }
    }

    // Create the new while op
    auto newLoop = scf::WhileOp::create(builder, loc, newResultTypes, newBeforeArgs);
    builder.inlineRegionBefore(whileOp.getBefore(), newLoop.getBefore(), newLoop.getBefore().end());
    builder.inlineRegionBefore(whileOp.getAfter(), newLoop.getAfter(), newLoop.getAfter().end());

    // Remove condition op's quantum terminating values
    scf::ConditionOp newConditionOp = newLoop.getConditionOp();
    SmallVector<Value> conditionOpArgSave(newConditionOp.getArgs());
    BitVector conditionOpEraseIndices(newConditionOp->getNumOperands());
    for (auto [i, arg] : llvm::enumerate(newConditionOp.getArgs())) {
        if (isa<quantum::QuregType>(arg.getType())) {
            conditionOpEraseIndices.set(i + 1);
        }
        else if (isa<quantum::QubitType>(arg.getType())) {
            conditionOpEraseIndices.set(i + 1);
        }
    }
    newConditionOp->eraseOperands(conditionOpEraseIndices);

    // Handle the Before region, and cascade the map into the After region
    SmallVector<Operation *> beforeRegionErasureWorklist =
        handleRegion(builder, newLoop.getBefore(), beforeRegionTracker, false).value();

    QubitValueTracker afterRegionTracker = tracker;
    for (auto [i, arg] : llvm::enumerate(conditionOpArgSave)) {
        if (isa<quantum::QuregType>(arg.getType())) {
            afterRegionTracker.setRQreg(newLoop.getAfterArguments()[i],
                                        beforeRegionTracker.getRQreg(arg));
        }
        else if (isa<quantum::QubitType>(arg.getType())) {
            afterRegionTracker.setRQubit(newLoop.getAfterArguments()[i],
                                         beforeRegionTracker.getRQubit(arg));
        }
    }

    // Handle the After region
    eraseSCFYieldQuantumOperands(cast<scf::YieldOp>(newLoop.getAfter().front().getTerminator()));
    handleRegion(builder, newLoop.getAfter(), afterRegionTracker);

    // Erase value semantics residue
    for (auto op : llvm::reverse(beforeRegionErasureWorklist)) {
        op->erase();
    }
    newLoop.getBeforeBody()->eraseArguments([](BlockArgument arg) {
        return isa<quantum::QubitType, quantum::QuregType>(arg.getType());
    });
    newLoop.getAfterBody()->eraseArguments([](BlockArgument arg) {
        return isa<quantum::QubitType, quantum::QuregType>(arg.getType());
    });

    // Deal with connection back to the outside
    cascadeMapAhead(whileOp, tracker);
    erasureWorklist.push_back(whileOp);
    for (auto [oldResultIdx, newResult] :
         llvm::zip_equal(oldClassicalResultIndices, newLoop->getResults())) {
        builder.replaceAllUsesWith(whileOp->getResult(oldResultIdx), newResult);
    }
}

std::optional<SmallVector<Operation *>> handleRegion(IRRewriter &builder, Region &r,
                                                     QubitValueTracker &tracker, bool erase)
{
    assert(r.hasOneBlock() && "Expected region to have a single block");
    assert(r.front().getTerminator() && "Expected region to have a terminator");

    SmallVector<Operation *> erasureWorklist;
    r.walk<WalkOrder::PreOrder>([&](Operation *op) {
        llvm::TypeSwitch<Operation *, void>(op)
            .Case<quantum::AllocOp>(
                [&](auto o) { handleAlloc(builder, o, tracker, erasureWorklist); })
            .Case<quantum::DeallocOp>(
                [&](auto o) { handleDealloc(builder, o, tracker, erasureWorklist); })
            .Case<quantum::ExtractOp>(
                [&](auto o) { handleExtract(builder, o, tracker, erasureWorklist); })
            .Case<quantum::InsertOp>([&](auto o) { handleInsert(o, tracker, erasureWorklist); })
            .Case<quantum::AllocQubitOp>(
                [&](auto o) { handleAllocQubit(builder, o, tracker, erasureWorklist); })
            .Case<quantum::DeallocQubitOp>(
                [&](auto o) { handleDeallocQubit(builder, o, tracker, erasureWorklist); })
            .Case<quantum::QuantumOperation>(
                [&](auto o) { handleGate(builder, o, tracker, erasureWorklist); })
            // .Case<func::CallOp>([&](auto o) { handleCall(builder, o, tracker); })
            .Case<quantum::ComputationalBasisOp>(
                [&](auto o) { handleCompbasis(builder, o, tracker); })
            .Case<quantum::NamedObsOp>([&](auto o) { handleNamedObs(builder, o, tracker); })
            .Case<quantum::HermitianOp>([&](auto o) { handleHermitian(builder, o, tracker); })
            .Case<quantum::MeasureOp>(
                [&](auto o) { handleMeasure(builder, o, tracker, erasureWorklist); })
            .Case<mbqc::MeasureInBasisOp>(
                [&](auto o) { handleMeasureInBasis(builder, o, tracker, erasureWorklist); })
            .Case<pbc::PPMeasurementOp>(
                [&](auto o) { handlePPM(builder, o, tracker, erasureWorklist); })
            .Case<quantum::AdjointOp>(
                [&](auto o) { handleAdjoint(builder, o, tracker, erasureWorklist); })
            // .Case<scf::IfOp>([&](auto o) { handleIf(builder, o, tracker); })
            // .Case<scf::IndexSwitchOp>([&](auto o) { handleSwitch(builder, o, tracker); })
            .Case<scf::ForOp>([&](auto o) { handleFor(builder, o, tracker, erasureWorklist); })
            .Case<scf::WhileOp>([&](auto o) { handleWhile(builder, o, tracker, erasureWorklist); })
            .Case<mbqc::GraphStatePrepOp>(
                [&](auto o) { handleGraphStatePrep(builder, o, tracker, erasureWorklist); })
            .Default([](Operation *) {});
    });

    if (erase) {
        if (isa<quantum::YieldOp>(r.front().getTerminator())) {
            erasureWorklist.push_back(r.front().getTerminator());
        }
        for (auto op : llvm::reverse(erasureWorklist)) {
            op->erase();
        }
        return std::nullopt;
    }
    else {
        return erasureWorklist;
    }
}

} // namespace

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_REFERENCESEMANTICSCONVERSIONPASS
#define GEN_PASS_DEF_REFERENCESEMANTICSCONVERSIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct ReferenceSemanticsConversionPass
    : impl::ReferenceSemanticsConversionPassBase<ReferenceSemanticsConversionPass> {
    using ReferenceSemanticsConversionPassBase::ReferenceSemanticsConversionPassBase;

    void runOnOperation() final
    {
        Operation *mod = getOperation();
        MLIRContext *ctx = mod->getContext();
        IRRewriter builder(ctx);

        // Collect all functions that need to be converted
        SetVector<func::FuncOp> targetFuncs;
        mod->walk([&](func::FuncOp f) {
            if (f->hasAttrOfType<UnitAttr>("quantum.node")) {
                targetFuncs.insert(f);
            }
        });

        // Convert the main quantum.mode functions
        for (auto targetFunc : targetFuncs) {
            QubitValueTracker tracker;
            handleRegion(builder, targetFunc.getBody(), tracker);
        }
    }
};

} // namespace quantum
} // namespace catalyst
