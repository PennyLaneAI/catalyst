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

#include <optional>

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
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

struct QubitValueTracker {
  public:
    QubitValueTracker() : rQreg(nullptr), vQreg(nullptr) {}
    QubitValueTracker(Value _rQreg) : rQreg(_rQreg)
    {
        assert(isa<qref::QuregType>(_rQreg.getType()) && "Expected qref.reg type");
    }

    Value getCurrentVQreg() { return vQreg; }
    void setCurrentVQreg(Value v)
    {
        assert(isa<quantum::QuregType>(v.getType()) && "Expected quantum.reg type");
        vQreg = v;
    }

    /**
    Return the current value semantics qubit Value for the reference semantics qubit.

    If the value semantics qubit SSA Value does not exist in the IR yet, for example due to a
    previous call to insertAllDanglingQubits(), a quantum.extract op from the corresponding
    quantum.reg is created, and the newly extracted quantum.bit value is updated into the tracker
    map and then returned.
    */
    Value getCurrentVQubit(Value rQubit, IRRewriter &builder)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");

        if (qubit_map.contains(rQubit)) {
            return qubit_map.lookup(rQubit);
        }
        else {
            auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
            assert(getOp && "Only qref.get ops can produce qref.bit SSA values");
            Value newlyExtracted = extract(builder, getOp.getIdxAttr(), getOp.getIdx()).getQubit();
            qubit_map.insert({rQubit, newlyExtracted});
            return newlyExtracted;
        }
    }

    void setCurrentVQubit(Value rQubit, Value v)
    {
        assert(isa<quantum::QubitType>(v.getType()) && "Expected quantum.bit type");
        if (qubit_map.contains(rQubit)) {
            qubit_map[rQubit] = v;
        }
        else {
            qubit_map.insert({rQubit, v});
        }
    }

    /**
    Create a quantum.extract op from the current value semantics qreg Value, at the given index.

    Note that this method only creates the ops, it does not update the tracker maps.
    */
    quantum::ExtractOp extract(IRRewriter &builder, std::optional<uint64_t> idxAttr = std::nullopt,
                               Value idxValue = nullptr)
    {
        assert((idxAttr.has_value() ^ (idxValue != nullptr)) &&
               "expected exactly one index for extract");

        OpBuilder::InsertionGuard guard(builder);

        Type qubitType = quantum::QubitType::get(rQreg.getContext());
        Type i64Type = builder.getI64Type();
        if (idxAttr.has_value()) {
            return quantum::ExtractOp::create(builder, rQreg.getLoc(), qubitType, getCurrentVQreg(),
                                              {}, IntegerAttr::get(i64Type, idxAttr.value()));
        }
        else {
            return quantum::ExtractOp::create(builder, rQreg.getLoc(), qubitType, vQreg, idxValue,
                                              nullptr);
        }
    }

    /**
    Insert all value semantics qubits currently in the map back into the value semantics register.
    Returns the resultant value semantics qreg Value from the final quantum.insert op.
    */
    Value insertAllDanglingQubits(IRRewriter &builder, Operation *insertionPoint)
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(insertionPoint);

        Location loc = rQreg.getLoc();
        MLIRContext *ctx = rQreg.getContext();
        Type i64Type = builder.getI64Type();

        Value current = vQreg;
        for (auto pair : qubit_map) {
            Value rQubit = pair.first;
            assert(isa<qref::QubitType>(rQubit.getType()) &&
                   "Expected a reference semantics qubit SSA value");
            Value vQubit = pair.second;
            assert(isa<quantum::QubitType>(vQubit.getType()) &&
                   "Expected a value semantics qubit SSA value");

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

        vQreg = current;
        qubit_map.clear();
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
};

void handleRegion(IRRewriter &builder, Region &r,
                  llvm::DenseMap<Value, QubitValueTracker *> &qubitValueTrackers);

//
// Misc helpers
//

/**
Given a Value of type qref.bit, return the qref.reg value that it belongs to.
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
attributes, except we replace the reference qubit SSA Values with the value qubit SSA Values.

If the value semantics qubit SSA Values do not exist in the IR yet, a quantum.extract op from
the corresponding quantum.reg is created, and the newly extracted quantum.bit value is used.

The result types of the value semantics operation will be the same as the old one, unless
explicitly overriden.

This method just creates the new op, and it does NOT update the trackers, except the extract ops
created in the above scenario.
 */
template <typename OpTy>
OpTy migrateOpToValueSemantics(IRRewriter &builder, Operation *qrefOp,
                               const llvm::DenseMap<Value, QubitValueTracker *> &qubitValueTrackers,
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
            vOperands.push_back(qubitValueTrackers.lookup(getRSourceRegisterValue(v))
                                    ->getCurrentVQubit(v, builder));
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

DenseI32ArrayAttr getResultSegmentSizes(IRRewriter &builder, qref::QuantumOperation rGateOp)
{
    // The reference semantics gates do not produce results.
    // Therefore, we need to manually set the result segment sizes for the corresponding
    // value semantics gate.

    int32_t ctrl_len, non_ctrl_len = -1;
    if (auto rCtrlGate = dyn_cast<qref::QuantumGate>(rGateOp.getOperation())) {
        // Has controls
        ctrl_len = rCtrlGate.getCtrlQubitOperands().size();
        non_ctrl_len = rCtrlGate.getNonCtrlQubitOperands().size();
    }
    else {
        // No controls
        ctrl_len = 0;
        non_ctrl_len = rGateOp.getQubitOperands().size();
    }

    return builder.getDenseI32ArrayAttr({non_ctrl_len, ctrl_len});
}

// void getNecessaryRegionRegisters(Operation *regionedOp, SetVector<Value>
// &necessaryRegionRegisters)
// {
//     auto *qrefDialect = regionedOp->getContext()->getLoadedDialect<qref::QRefDialect>();

//     regionedOp->walk([&](Operation *op) {
//         if (op->getDialect() != qrefDialect) {
//             return;
//         }
//         for (Value v : op->getOperands()) {
//             if (isa<qref::QuregType>(v.getType())) {
//                 necessaryRegionRegisters.insert(v);
//             }
//         }
//     });
// }

//
// Handlers for each op
//

// Memory ops

void handleAlloc(IRRewriter &builder, qref::AllocOp rAllocOp,
                 llvm::DenseMap<Value, QubitValueTracker *> &qubitValueTrackers)
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
    qubitValueTrackers.lookup(rAllocOp.getQreg())->setCurrentVQreg(vAllocOp.getQreg());
}

void handleGet(IRRewriter &builder, qref::GetOp getOp,
               llvm::DenseMap<Value, QubitValueTracker *> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(getOp);

    Value rQreg = getOp.getQreg();
    quantum::ExtractOp extractOp =
        qubitValueTrackers.lookup(rQreg)->extract(builder, getOp.getIdxAttr(), getOp.getIdx());
    qubitValueTrackers.lookup(rQreg)->setCurrentVQubit(getOp.getQubit(), extractOp.getQubit());
}

void handleDealloc(IRRewriter &builder, qref::DeallocOp rDeallocOp,
                   llvm::DenseMap<Value, QubitValueTracker *> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rDeallocOp);
    Location loc = rDeallocOp.getLoc();

    Value insertedQreg = qubitValueTrackers.lookup(rDeallocOp.getQreg())
                             ->insertAllDanglingQubits(builder, rDeallocOp);
    quantum::DeallocOp::create(builder, loc, insertedQreg);

    builder.eraseOp(rDeallocOp);
}

// Gate Ops

void handleGate(IRRewriter &builder, qref::QuantumOperation rGateOp,
                llvm::DenseMap<Value, QubitValueTracker *> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = rGateOp.getContext();

    {
        // Need to insert all dangling qubits if any of the qubit indices are dynamic
        SetVector<QubitValueTracker *> sourceQregs;
        for (Value rQubit : rGateOp.getQubitOperands()) {
            auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
            assert(getOp && "Only qref.get ops can produce qref.bit SSA values");
            if (getOp.getIdx()) {
                sourceQregs.insert(qubitValueTrackers.lookup(getRSourceRegisterValue(rQubit)));
            }
        }
        for (auto tracker : sourceQregs) {
            tracker->insertAllDanglingQubits(builder, rGateOp);
        }
    }

    SmallVector<Type> qubitResultsType;
    for (size_t i = 0; i < rGateOp.getQubitOperands().size(); i++) {
        qubitResultsType.push_back(quantum::QubitType::get(ctx));
    }

    quantum::QuantumOperation vGateOp;
    if (isa<qref::CustomOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::CustomOp>(builder, rGateOp, qubitValueTrackers,
                                                               qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }
    else if (isa<qref::PauliRotOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PauliRotOp>(
            builder, rGateOp, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }
    else if (isa<qref::GlobalPhaseOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::GlobalPhaseOp>(
            builder, rGateOp, qubitValueTrackers, qubitResultsType);
    }
    else if (isa<qref::MultiRZOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::MultiRZOp>(
            builder, rGateOp, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }
    else if (isa<qref::PCPhaseOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PCPhaseOp>(
            builder, rGateOp, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }
    else if (isa<qref::QubitUnitaryOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::QubitUnitaryOp>(
            builder, rGateOp, qubitValueTrackers, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }

    for (auto [i, rQubit] : llvm::enumerate(rGateOp.getQubitOperands())) {
        qubitValueTrackers.lookup(getRSourceRegisterValue(rQubit))
            ->setCurrentVQubit(rQubit, vGateOp.getQubitResults()[i]);
    }

    {
        // Need to insert all dangling qubits if any of the qubit indices are dynamic
        SetVector<QubitValueTracker *> sourceQregs;
        for (Value rQubit : rGateOp.getQubitOperands()) {
            auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
            assert(getOp && "Only qref.get ops can produce qref.bit SSA values");
            if (getOp.getIdx()) {
                sourceQregs.insert(qubitValueTrackers.lookup(getRSourceRegisterValue(rQubit)));
            }
        }
        for (auto tracker : sourceQregs) {
            tracker->insertAllDanglingQubits(builder, rGateOp);
        }
    }

    builder.eraseOp(rGateOp);
}

// Observable Ops

void handleNamedObs(IRRewriter &builder, qref::NamedObsOp rNamedObsOp,
                    llvm::DenseMap<Value, QubitValueTracker *> &qubitValueTrackers)
{
    OpBuilder::InsertionGuard guard(builder);

    auto vNamedObsOp =
        migrateOpToValueSemantics<quantum::NamedObsOp>(builder, rNamedObsOp, qubitValueTrackers);
    builder.replaceOp(rNamedObsOp, vNamedObsOp);
}

// Control flow

// void handleFor(IRRewriter &builder, scf::ForOp forOp, llvm::DenseMap<Value, QubitValueTracker>
// &qubitValueTrackers)
// {
//     // Insert all extracted qubits from the necessary registers, and update the maps
//     OpBuilder::InsertionGuard guard(builder);
//     Location loc = forOp->getLoc();
//     MLIRContext *ctx = forOp.getContext();

//     SetVector<Value> necessaryRegionRQregs;
//     getNecessaryRegionRegisters(forOp, necessaryRegionRQregs);

//     llvm::DenseMap<Value, Value> newForRegionCurrentQuregs;

//     // 1. Insert all extracted qubits before the loop
//     SmallVector<Value> newInitArgs(forOp.getInitArgs());
//     builder.setInsertionPoint(forOp);
//     for (Value rQreg : necessaryRegionRQregs) {
//         Value insertedQreg = insertQubits(builder, rQreg, currentQubits, currentQuregs, forOp);
//         newInitArgs.push_back(insertedQreg);
//     }

//     // 2. The new forOp iteration args must take in the registers
//     // Remove the default empty block, it doesn't have the new block arg signature
//     auto newLoop = scf::ForOp::create(builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
//                                       forOp.getStep(), newInitArgs);
//     builder.eraseBlock(newLoop.getBody());

//     // 3. Move operations from old body to new body
//     // The old loop body still refers to the old block arguments.
//     // We must map them to the new ones.
//     builder.inlineRegionBefore(forOp.getRegion(), newLoop.getRegion(),
//     newLoop.getRegion().end()); for (auto rQreg : necessaryRegionRQregs) {
//         assert(isa<qref::QuregType>(rQreg.getType()) &&
//                "Expected the only extra loop arguments in value semantics to be quantum
//                registers");
//         Value vQreg =
//             newLoop.getRegion().getBlocks().front().addArgument(quantum::QuregType::get(ctx),
//             loc);
//         newForRegionCurrentQuregs[rQreg] = vQreg;
//     }

//     llvm::DenseMap<Value, Value> newForRegionCurrentQubits;
//     handleRegion(builder, newLoop.getRegion(), qubitValueTrackers);

//     // 5. Insert loop region registers and Yield
//     // builder.setInsertionPoint(newLoop.getRegion().getBlocks().back().getTerminator());
//     auto yieldOp = cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
//     SmallVector<Value> yieldOperands(yieldOp.getOperands());
//     for (Value rQreg : necessaryRegionRQregs) {
//         Value insertedQreg =
//             insertQubits(builder, rQreg, newForRegionCurrentQubits, newForRegionCurrentQuregs,
//                          newLoop.getBody()->getTerminator());
//         yieldOperands.push_back(insertedQreg);
//     }
//     builder.setInsertionPoint(yieldOp);
//     scf::YieldOp::create(builder, loc, yieldOperands);
//     builder.eraseOp(yieldOp);

//     // 6. New for loop's returned qreg is the new value semantics qreg value in the outer scope
//     size_t numOldYields = forOp.getNumResults();
//     for (auto [i, rQreg] : llvm::enumerate(necessaryRegionRQregs)) {
//         currentQuregs[rQreg] = newLoop->getResult(numOldYields + i);
//     }

//     builder.eraseOp(forOp);
// }

// Driver

void handleRegion(IRRewriter &builder, Region &r,
                  llvm::DenseMap<Value, QubitValueTracker *> &qubitValueTrackers)
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
        // else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        //     handleFor(builder, forOp, currentQubits, currentQuregs);
        // }
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
        // We find qnode functions by identifying the parent function ops of MPs
        SetVector<func::FuncOp> qnodeFuncs;
        mod->walk([&](quantum::MeasurementProcess _mp) {
            qnodeFuncs.insert(_mp->getParentOfType<func::FuncOp>());
        });

        for (auto qnodeFunc : qnodeFuncs) {
            // This map tracks: qref.reg Value -> QubitValueTracker for that qreg
            llvm::DenseMap<Value, QubitValueTracker *> qubitValueTrackers;

            qnodeFunc.walk<WalkOrder::PreOrder>([&](qref::AllocOp rAllocOp) {
                auto tracker = new QubitValueTracker(rAllocOp.getQreg());
                qubitValueTrackers.insert({rAllocOp.getQreg(), tracker});
            });

            handleRegion(builder, qnodeFunc.getBody(), qubitValueTrackers);

            for (auto pair : qubitValueTrackers) {
                delete pair.second;
            }

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
