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

#define DEBUG_TYPE "value-semantics-conversion"

#include <cstdint>
#include <memory>
#include <optional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"  // for PassManager
#include "mlir/Transforms/Passes.h" // for createCSEPass

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

[[maybe_unused]] void dumpMap(const llvm::DenseMap<Value, Value> &map)
{
    for ([[maybe_unused]] auto _ : map) {
        llvm::errs() << "(" << _.first << ", " << _.second << ") \n";
    }
}

/**
This class tracks the current vQreg and vQubit Values for a given rQreg Value.

The managed entity is always with respect to the unique rQreg Value in reference semantics IR.
*/
struct QubitValueTracker {
  public:
    QubitValueTracker(Value _rQreg) : rQreg(_rQreg)
    {
        assert(isa<qref::QuregType>(_rQreg.getType()) && "Expected qref.reg type");
    }

    /**
    Return the current vQreg Value corresponding to the managed rQreg.
    */
    Value getCurrentVQreg()
    {
        assert(isa<quantum::QuregType>(this->vQreg.getType()) && "Expected quantum.reg type");
        return this->vQreg;
    }

    /**
    Set the current vQreg Value corresponding to this rQreg.
    */
    void setCurrentVQreg(Value _vQreg)
    {
        assert(isa<quantum::QuregType>(_vQreg.getType()) && "Expected quantum.reg type");
        this->vQreg = _vQreg;
    }

    /**
    Return the current vQubit Value for the given rQubit.

    This method will fail if the given rQubit does not belong to the managed rQreg on this tracker
    instance.

    If the value semantics qubit SSA Value does not exist in the IR yet, for example due to a
    previous call to insertAllDanglingQubits(), a quantum.extract op from the corresponding
    quantum.reg is created, and the newly extracted quantum.bit value is updated into the tracker
    map and then returned.
    */
    Value getCurrentVQubit(Value rQubit, IRRewriter &builder)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        assert(this->belongs(rQubit) && "The qubit value does not belong to the qreg");

        if (this->qubit_map.contains(rQubit)) {
            return this->qubit_map.at(rQubit);
        }
        else {
            auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
            assert(getOp && "Only qref.get ops can produce qref.bit SSA values");

            Value newlyExtracted =
                this->extract(builder, getOp.getIdxAttr(), getOp.getIdx()).getQubit();
            this->setCurrentVQubit(rQubit, newlyExtracted);
            return newlyExtracted;
        }
    }

    /**
    Set the current vQubit Value corresponding to the given rQubit.
    */
    void setCurrentVQubit(Value rQubit, Value vQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        assert(isa<quantum::QubitType>(vQubit.getType()) && "Expected quantum.bit type");

        if (this->qubit_map.contains(rQubit)) {
            this->qubit_map[rQubit] = vQubit;
        }
        else {
            this->qubit_map.insert({rQubit, vQubit});
        }
    }

    /**
    Create a quantum.extract op from the vQreg Value, at the given index.

    Note that this method only creates the extract op. It does not update the tracker maps.
    */
    quantum::ExtractOp extract(IRRewriter &builder, std::optional<uint64_t> idxAttr = std::nullopt,
                               Value idxValue = nullptr)
    {
        assert((idxAttr.has_value() ^ (idxValue != nullptr)) &&
               "expected exactly one index for extract op");

        OpBuilder::InsertionGuard guard(builder);

        Type qubitType = quantum::QubitType::get(rQreg.getContext());
        Type i64Type = builder.getI64Type();
        if (idxAttr.has_value()) {
            return quantum::ExtractOp::create(builder, rQreg.getLoc(), qubitType,
                                              this->getCurrentVQreg(), {},
                                              IntegerAttr::get(i64Type, idxAttr.value()));
        }
        else {
            return quantum::ExtractOp::create(builder, rQreg.getLoc(), qubitType, vQreg, idxValue,
                                              nullptr);
        }
    }

    /**
    Insert all vQubits currently in the map back into the current vQreg.
    The resultant vQreg Value from the final quantum.insert op is recorded as the current vQreg
    Value and then returned. The qubit tracker map is cleared.

    Insertion point is before the `insertionPoint` argument.
    */
    Value insertAllDanglingQubits(IRRewriter &builder, Operation *insertionPoint)
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(insertionPoint);

        Location loc = this->rQreg.getLoc();
        MLIRContext *ctx = this->rQreg.getContext();
        Type i64Type = builder.getI64Type();

        Value current = this->vQreg;
        for (auto pair : this->qubit_map) {
            Value rQubit = pair.first;
            assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
            Value vQubit = pair.second;
            assert(isa<quantum::QubitType>(vQubit.getType()) && "Expected quantum.bit type");

            auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
            assert(getOp && "Only qref.get ops can produce qref.bit SSA values");
            std::optional<uint64_t> idxAttr = getOp.getIdxAttr();
            if (idxAttr.has_value()) {
                current = quantum::InsertOp::create(
                    builder, loc, quantum::QuregType::get(ctx), current, {},
                    IntegerAttr::get(i64Type, idxAttr.value()), vQubit);
            }
            else {
                current = quantum::InsertOp::create(builder, loc, quantum::QuregType::get(ctx),
                                                    current, getOp.getIdx(), nullptr, vQubit);
            }
        }

        this->vQreg = current;
        this->qubit_map.clear();
        return current;
    }

  private:
    // The unique qref.qreg value.
    Value rQreg;

    // The current quantum.qreg value.
    Value vQreg;

    // The map of all qubits belonging to this qreg.
    // The map is from qref.bit values to the current quantum.bit values.
    //
    // Note that we need an ordered map, otherwise when inserting all dangling qubits the order of
    // the insert ops will not be deterministic!
    llvm::MapVector<Value, Value> qubit_map;

    /**
    Checks whether the given rQubit value belongs to the managed rQreg value on this tracker
    instance.
    */
    bool belongs(Value rQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");

        auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
        assert(getOp && "Only qref.get ops can produce qref.bit SSA values");

        return getOp.getQreg() == this->rQreg;
    }
};

void handleRegion(IRRewriter &builder, Region &r,
                  llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers);

//
// Misc helpers
//

/**
Given a rQubit Value, return the rQreg Value that it belongs to.
 */
Value getRSourceRegisterValue(Value rQubit)
{
    assert(isa<qref::QubitType>(rQubit.getType()) && "Can only query qref.bit types");
    auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
    assert(getOp && "Only qref.get ops can produce qref.bit SSA values");
    return getOp.getQreg();
}

/**
Given a reference semantics operation instance, migrate it to value semantics.

We create the corresponding value semantics operation, with exactly the same operands and
attributes, except we replace the rQubit Values with the corresponding vQubit Values.

If the vQubit Values do not exist in the IR yet, a quantum.extract op from the corresponding
quantum.reg is created, and the newly extracted quantum.bit Value is used.

The result types of the value semantics operation will be the same as the old one, unless
explicitly overriden in the `newResultTypes` argument.

This method just creates the new op, and it does NOT update the trackers, except the extract ops
created in the above scenario.
 */
template <typename OpTy>
OpTy migrateOpToValueSemantics(
    IRRewriter &builder, Operation *qrefOp,
    const llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers,
    std::optional<TypeRange> newResultTypes = std::nullopt)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(qrefOp);
    Location loc = qrefOp->getLoc();

    // Create the new op using the generic state-based approach
    // We cannot just clone, since we are changing the op type
    OperationState state(loc, OpTy::getOperationName());

    SmallVector<Value> vOperands;
    for (Value v : qrefOp->getOperands()) {
        if (isa<qref::QubitType>(v.getType())) {
            vOperands.push_back(
                qubitValueTrackers.at(getRSourceRegisterValue(v))->getCurrentVQubit(v, builder));
        }
        else {
            vOperands.push_back(v);
        }
    }
    state.addOperands(vOperands);
    state.addAttributes(qrefOp->getAttrs());

    TypeRange outTypes =
        newResultTypes.has_value() ? newResultTypes.value() : TypeRange(qrefOp->getResultTypes());
    state.addTypes(outTypes);

    return cast<OpTy>(builder.create(state));
}

/**
Given a qref gate operation, compute the result segment sizes for the corresponding value semantics
gate operation.

The reference semantics gates do not produce results.
Therefore, we need to manually set the result segment sizes for the corresponding
value semantics gate.
 */
DenseI32ArrayAttr getResultSegmentSizes(IRRewriter &builder, qref::QuantumGate rGateOp)
{
    int32_t non_ctrl_len = rGateOp.getNonCtrlQubitOperands().size();
    int32_t ctrl_len = rGateOp.getCtrlQubitOperands().size();
    return builder.getDenseI32ArrayAttr({non_ctrl_len, ctrl_len});
}

/**
Collect the rQreg Values that are captured into a region from above via closure.
This includes any rQregs that produce rQubits that the region captures from above via closure.

Reference semantics dialect operations do not take in or produce qreg Values, which means all
qreg Values are taken in via closure from above.

When converting to value semantics, the vQregs need to be taken in by the region-ed operations
explicitly.
 */
void getNecessaryRegionRegisters(Region &r, SetVector<Value> &necessaryRegionRegisters)
{
    auto *qrefDialect = r.getContext()->getLoadedDialect<qref::QRefDialect>();

    r.walk([&](Operation *op) {
        if (op->getDialect() != qrefDialect) {
            return;
        }
        for (Value v : op->getOperands()) {
            if (isa<qref::QuregType>(v.getType())) {
                if (v.getParentRegion()->isProperAncestor(&r)) {
                    necessaryRegionRegisters.insert(v);
                }
            }
            else if (isa<qref::QubitType>(v.getType())) {
                Value rQreg = getRSourceRegisterValue(v);
                if (rQreg.getParentRegion()->isProperAncestor(&r)) {
                    necessaryRegionRegisters.insert(rQreg);
                }
            }
        }
    });
}

//
// Handlers for each op
//

// Memory ops

void handleAlloc(IRRewriter &builder, qref::AllocOp rAllocOp,
                 llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rAllocOp);

    Location loc = rAllocOp.getLoc();
    MLIRContext *ctx = rAllocOp.getContext();
    Type qregType = quantum::QuregType::get(ctx);
    Type i64Type = builder.getI64Type();

    quantum::AllocOp vAllocOp;
    std::optional<uint64_t> nqubitsAttr = rAllocOp.getNqubitsAttr();
    if (nqubitsAttr.has_value()) {
        vAllocOp = quantum::AllocOp::create(builder, loc, qregType, {},
                                            IntegerAttr::get(i64Type, *nqubitsAttr));
    }
    else {
        vAllocOp = quantum::AllocOp::create(builder, loc, qregType, rAllocOp.getNqubits(), nullptr);
    }
    qubitValueTrackers.at(rAllocOp.getQreg())->setCurrentVQreg(vAllocOp.getQreg());
}

void handleGet(IRRewriter &builder, qref::GetOp getOp,
               llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(getOp);

    Value rQreg = getOp.getQreg();
    quantum::ExtractOp extractOp =
        qubitValueTrackers.at(rQreg)->extract(builder, getOp.getIdxAttr(), getOp.getIdx());
    qubitValueTrackers.at(rQreg)->setCurrentVQubit(getOp.getQubit(), extractOp.getQubit());
}

void handleDealloc(IRRewriter &builder, qref::DeallocOp rDeallocOp,
                   llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rDeallocOp);
    Location loc = rDeallocOp.getLoc();

    Value insertedQreg =
        qubitValueTrackers.at(rDeallocOp.getQreg())->insertAllDanglingQubits(builder, rDeallocOp);
    quantum::DeallocOp::create(builder, loc, insertedQreg);

    builder.eraseOp(rDeallocOp);
}

// Gate Ops

void handleGate(IRRewriter &builder, qref::QuantumOperation rGateOp,
                llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = rGateOp.getContext();

    // Need to insert all dangling qubits if any of the qubit indices are dynamic
    for (Value rQubit : rGateOp.getQubitOperands()) {
        auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
        assert(getOp && "Only qref.get ops can produce qref.bit SSA values");
        if (getOp.getIdx()) {
            qubitValueTrackers.at(getRSourceRegisterValue(rQubit))
                ->insertAllDanglingQubits(builder, rGateOp);
        }
    }

    SmallVector<Type> qubitResultsType;
    for (size_t i = 0; i < rGateOp.getQubitOperands().size(); i++) {
        qubitResultsType.push_back(quantum::QubitType::get(ctx));
    }

    quantum::QuantumOperation vGateOp;
    Operation *_rGateOp = rGateOp.getOperation();
    if (auto rCustomOp = dyn_cast<qref::CustomOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::CustomOp>(
            builder, rCustomOp, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rCustomOp));
    }
    else if (auto rPauliRotOP = dyn_cast<qref::PauliRotOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PauliRotOp>(
            builder, rPauliRotOP, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rPauliRotOP));
    }
    else if (auto rGPhaseOp = dyn_cast<qref::GlobalPhaseOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::GlobalPhaseOp>(
            builder, rGPhaseOp, qubitValueTrackers, qubitResultsType);
    }
    else if (auto rMultiRZOp = dyn_cast<qref::MultiRZOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::MultiRZOp>(
            builder, rMultiRZOp, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rMultiRZOp));
    }
    else if (auto rPCPhaseOp = dyn_cast<qref::PCPhaseOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PCPhaseOp>(
            builder, rPCPhaseOp, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rPCPhaseOp));
    }
    else if (auto qQubitUnitaryOp = dyn_cast<qref::QubitUnitaryOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::QubitUnitaryOp>(
            builder, qQubitUnitaryOp, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, qQubitUnitaryOp));
    }
    else if (auto rSetStateOp = dyn_cast<qref::SetStateOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::SetStateOp>(
            builder, rSetStateOp, qubitValueTrackers, qubitResultsType);
    }
    else if (auto rSetBasisStateOp = dyn_cast<qref::SetBasisStateOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::SetBasisStateOp>(
            builder, rSetBasisStateOp, qubitValueTrackers, qubitResultsType);
    }

    for (auto [i, rQubit] : llvm::enumerate(rGateOp.getQubitOperands())) {
        qubitValueTrackers.at(getRSourceRegisterValue(rQubit))
            ->setCurrentVQubit(rQubit, vGateOp.getQubitResults()[i]);
    }

    // Need to insert all dangling qubits if any of the qubit indices are dynamic
    for (Value rQubit : rGateOp.getQubitOperands()) {
        auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
        assert(getOp && "Only qref.get ops can produce qref.bit SSA values");
        if (getOp.getIdx()) {
            qubitValueTrackers.at(getRSourceRegisterValue(rQubit))
                ->insertAllDanglingQubits(builder, rGateOp);
        }
    }

    builder.eraseOp(rGateOp);
}

// Observable Ops

void handleNamedObs(IRRewriter &builder, qref::NamedObsOp rNamedObsOp,
                    llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);

    auto vNamedObsOp =
        migrateOpToValueSemantics<quantum::NamedObsOp>(builder, rNamedObsOp, qubitValueTrackers);
    builder.replaceOp(rNamedObsOp, vNamedObsOp);
}

// Control flow

void handleFor(IRRewriter &builder, scf::ForOp forOp,
               llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);
    Location loc = forOp->getLoc();
    MLIRContext *ctx = forOp.getContext();

    SetVector<Value> rQregsUsedByRegion;
    getNecessaryRegionRegisters(forOp->getRegion(0), rQregsUsedByRegion);

    // 1. Insert all extracted qubits before the loop, and append the vQregs to the for loop args
    SmallVector<Value> newInitArgs(forOp.getInitArgs());
    builder.setInsertionPoint(forOp);
    for (Value rQreg : rQregsUsedByRegion) {
        Value insertedQreg = qubitValueTrackers.at(rQreg)->insertAllDanglingQubits(builder, forOp);
        newInitArgs.push_back(insertedQreg);
    }

    // 2. The new forOp iteration args must take in the registers
    // Remove the default empty block, it doesn't have the new block arg signature
    auto newLoop = scf::ForOp::create(builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                                      forOp.getStep(), newInitArgs);
    builder.eraseBlock(newLoop.getBody());

    // 3. Move operations from old body to new body
    // The old loop body still refers to the old block arguments.
    // We must map them to the new ones.
    builder.inlineRegionBefore(forOp.getRegion(), newLoop.getRegion(), newLoop.getRegion().end());
    for (auto rQreg : rQregsUsedByRegion) {
        assert(isa<qref::QuregType>(rQreg.getType()) &&
               "Expected the only extra loop arguments in value semantics to be quantum registers");
        Value vQregNewArg =
            newLoop.getRegion().getBlocks().front().addArgument(quantum::QuregType::get(ctx), loc);
        qubitValueTrackers.at(rQreg)->setCurrentVQreg(vQregNewArg);
    }

    handleRegion(builder, newLoop.getRegion(), qubitValueTrackers);

    // 4. Insert loop region registers and Yield
    // builder.setInsertionPoint(newLoop.getRegion().getBlocks().back().getTerminator());
    auto yieldOp = cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
    size_t numOldYields = yieldOp->getNumResults();
    SmallVector<Value> yieldOperands(yieldOp.getOperands());
    for (Value rQreg : rQregsUsedByRegion) {
        Value insertedQreg =
            qubitValueTrackers.at(rQreg)->insertAllDanglingQubits(builder, yieldOp);
        yieldOperands.push_back(insertedQreg);
    }
    builder.setInsertionPoint(yieldOp);
    scf::YieldOp::create(builder, loc, yieldOperands);
    builder.eraseOp(yieldOp);

    // 5. New for loop's returned qreg is the new value semantics qreg value in the outer scope
    for (auto [i, rQreg] : llvm::enumerate(rQregsUsedByRegion)) {
        qubitValueTrackers.at(rQreg)->setCurrentVQreg(newLoop->getResult(numOldYields + i));
    }

    builder.eraseOp(forOp);
}

// Driver

void handleRegion(IRRewriter &builder, Region &r,
                  llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> &qubitValueTrackers)
{
    r.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (auto rAllocOp = dyn_cast<qref::AllocOp>(op)) {
            handleAlloc(builder, rAllocOp, qubitValueTrackers);
        }
        else if (auto getOp = dyn_cast<qref::GetOp>(op)) {
            handleGet(builder, getOp, qubitValueTrackers);
        }
        else if (auto rGateOp = dyn_cast<qref::QuantumOperation>(op)) {
            handleGate(builder, rGateOp, qubitValueTrackers);
        }
        else if (auto rNamedObsOp = dyn_cast<qref::NamedObsOp>(op)) {
            handleNamedObs(builder, rNamedObsOp, qubitValueTrackers);
        }
        else if (auto rDeallocOp = dyn_cast<qref::DeallocOp>(op)) {
            handleDealloc(builder, rDeallocOp, qubitValueTrackers);
        }
        else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            handleFor(builder, forOp, qubitValueTrackers);
        }
    });
}

} // anonymous namespace

namespace catalyst {
namespace qref {

#define GEN_PASS_DECL_VALUESEMANTICSCONVERSIONPASS
#define GEN_PASS_DEF_VALUESEMANTICSCONVERSIONPASS
#include "QRef/Transforms/Passes.h.inc"

struct ValueSemanticsConversionPass
    : impl::ValueSemanticsConversionPassBase<ValueSemanticsConversionPass> {
    using ValueSemanticsConversionPassBase::ValueSemanticsConversionPassBase;

    void runOnOperation() final
    {
        Operation *mod = getOperation();
        MLIRContext *ctx = mod->getContext();
        auto *qrefDialect = ctx->getLoadedDialect<qref::QRefDialect>();
        IRRewriter builder(ctx);

        // CSE potential duplicated getOps
        PassManager pm(ctx);
        pm.addPass(createCSEPass());

        // Collect all qnode functions.
        SetVector<func::FuncOp> qnodeFuncs;
        mod->walk([&](func::FuncOp f) {
            if (f->hasAttrOfType<UnitAttr>("quantum.node")) {
                qnodeFuncs.insert(f);
            }
        });

        for (auto qnodeFunc : qnodeFuncs) {
            // This map tracks: qref.reg Value -> QubitValueTracker for that qreg
            llvm::DenseMap<Value, std::unique_ptr<QubitValueTracker>> qubitValueTrackers;

            SmallVector<Value> rQregs;
            qnodeFunc.walk<WalkOrder::PreOrder>(
                [&](qref::AllocOp rAllocOp) { rQregs.push_back(rAllocOp.getQreg()); });
            for (Value rQreg : rQregs) {
                qubitValueTrackers.insert({rQreg, std::make_unique<QubitValueTracker>(rQreg)});
            }

            handleRegion(builder, qnodeFunc.getBody(), qubitValueTrackers);

            // Clean up: erase remaining qref dialect ops
            // Due to the nature of the reference semantics dialect, qref ops all have full side
            // effects, and will not delete themselves.
            SmallVector<Operation *> toErase;
            qnodeFunc->walk([&](Operation *op) {
                if (op->getDialect() == qrefDialect) {
                    toErase.push_back(op);
                }
            });
            for (Operation *op : llvm::reverse(toErase)) {
                op->erase();
            }
        }
    }
};

} // namespace qref
} // namespace catalyst
