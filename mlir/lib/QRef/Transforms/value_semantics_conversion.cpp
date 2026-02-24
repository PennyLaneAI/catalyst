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

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Region.h>
#define DEBUG_TYPE "value-semantics-conversion"

#include <optional>
#include <variant>

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

void handleRegion(IRRewriter &builder, Region &r, llvm::DenseMap<Value, Value> &currentQubits,
                  llvm::DenseMap<Value, Value> &currentQuregs);

//
// Misc helpers
//

template <typename OpTy>
OpTy migrateOpToValueSemantics(IRRewriter &builder, Operation *qrefOp,
                               llvm::DenseMap<Value, Value> &currentQubits,
                               const llvm::DenseMap<Value, Value> &currentQuregs,
                               std::optional<TypeRange> newResultTypes = std::nullopt)
{
    // Given a reference semantics operation instance, migrate it to value semantics.
    // We create the corresponding value semantics operation, with exactly the same operands and
    // attributes, except we replace the reference qubit SSA Values with the value qubit SSA Values.
    // The result types of the value semantics operation will be the same as the old one, unless
    // explicitly overriden.

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(qrefOp);
    Location loc = qrefOp->getLoc();
    MLIRContext *ctx = qrefOp->getContext();

    // Create the new op using the generic state-based approach
    // We cannot just clone, since we are changing the op type
    OperationState state(loc, OpTy::getOperationName());

    SmallVector<Value> vOperands;
    for (Value v : qrefOp->getOperands()) {
        if (isa<qref::QubitType>(v.getType())) {
            if (currentQubits.contains(v)) {
                vOperands.push_back(currentQubits.lookup(v));
            }
            else {
                assert(isa<qref::GetOp>(v.getDefiningOp()) &&
                       "Expected unrecorded qubit references to only come from qref.get ops");

                auto getOp = cast<qref::GetOp>(v.getDefiningOp());
                Type qubitType = quantum::QubitType::get(ctx);
                Type i64Type = builder.getI64Type();

                Value rQreg = getOp.getQreg();
                Value vQreg = currentQuregs.lookup(rQreg);

                quantum::ExtractOp extractOp;
                std::optional<uint64_t> idxAttr = getOp.getIdxAttr();
                if (idxAttr.has_value()) {
                    extractOp = quantum::ExtractOp::create(builder, loc, qubitType, vQreg, {},
                                                           IntegerAttr::get(i64Type, *idxAttr));
                }
                else {
                    extractOp = quantum::ExtractOp::create(builder, loc, qubitType, vQreg,
                                                           getOp.getIdx(), nullptr);
                }
                currentQubits.insert({getOp.getQubit(), extractOp.getQubit()});
                vOperands.push_back(extractOp.getQubit());
            }
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

Value insertQubits(IRRewriter &builder, Value rQreg, llvm::DenseMap<Value, Value> &currentQubits,
                   llvm::DenseMap<Value, Value> &currentQuregs, Operation *insertionPoint)
{
    // Create a chain of insert ops, inserting all qubits into the qreg.
    // All qubits extracted from the value semantics qreg corresponding to the input reference
    // semantics qreg will be inserted.
    // Note that inserting a value semantics qubit invalidates it from further uses: the
    // corresponding qubit reference needs to be extracted again.

    assert(isa<qref::QuregType>(rQreg.getType()) &&
           "Expected a reference semantics qreg SSA value");

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertionPoint);

    Location loc = rQreg.getLoc();
    MLIRContext *ctx = rQreg.getContext();
    Type i64Type = builder.getI64Type();

    Value vQreg = currentQuregs.lookup(rQreg);

    // Collect all qref qubits that come from the register
    // A register is either used by a qref.get, or a region op (adjoint, control flow, ...)
    // Only getOps produce qubits
    // Only care about getOps in the same scope, and before the insertion point of interest
    SmallVector<std::pair<Value, std::variant<Value, IntegerAttr>>> qubitIdxPairs;
    for (Operation *user : rQreg.getUsers()) {
        if ((user->getBlock() != insertionPoint->getBlock()) ||
            (!user->isBeforeInBlock(insertionPoint))) {
            continue;
        }
        if (auto getOp = dyn_cast<qref::GetOp>(user)) {
            Value currentVQubit = currentQubits.lookup(getOp.getQubit());
            std::optional<uint64_t> idxAttr = getOp.getIdxAttr();
            if (idxAttr.has_value()) {
                qubitIdxPairs.push_back(
                    {currentVQubit, IntegerAttr::get(i64Type, idxAttr.value())});
            }
            else {
                qubitIdxPairs.push_back({currentVQubit, getOp.getIdx()});
            }
            currentQubits.erase(getOp.getQubit());
        }
        else {
            continue;
        }
    }

    // Perform the inserts
    Value current = vQreg;
    for (auto pair : qubitIdxPairs) {
        Value qubit = pair.first;
        assert(isa<quantum::QubitType>(qubit.getType()) &&
               "Expected a value semantics qubit SSA value");
        std::variant<Value, IntegerAttr> idx = pair.second;
        if (std::holds_alternative<Value>(idx)) {
            current = quantum::InsertOp::create(builder, loc, quantum::QuregType::get(ctx), current,
                                                std::get<Value>(idx), nullptr, qubit);
        }
        else if (std::holds_alternative<IntegerAttr>(idx)) {
            current = quantum::InsertOp::create(builder, loc, quantum::QuregType::get(ctx), current,
                                                {}, std::get<IntegerAttr>(idx), qubit);
        }
    }

    currentQuregs[rQreg] = current;
    return current;
}

void getNecessaryRegionRegisters(Operation *regionedOp, SetVector<Value> &necessaryRegionRegisters)
{
    auto *qrefDialect = regionedOp->getContext()->getLoadedDialect<qref::QRefDialect>();

    regionedOp->walk([&](Operation *op) {
        if (op->getDialect() != qrefDialect) {
            return;
        }
        for (Value v : op->getOperands()) {
            if (isa<qref::QuregType>(v.getType())) {
                necessaryRegionRegisters.insert(v);
            }
        }
    });
}

//
// Handlers for each op
//

// Memory ops

void handleAlloc(IRRewriter &builder, qref::AllocOp rAllocOp,
                 llvm::DenseMap<Value, Value> &currentQuregs)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(rAllocOp);

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
    currentQuregs.insert({rAllocOp.getQreg(), vAllocOp.getQreg()});
}

void handleGet(IRRewriter &builder, qref::GetOp getOp, llvm::DenseMap<Value, Value> &currentQubits,
               llvm::DenseMap<Value, Value> &currentQuregs)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(getOp);

    Location loc = getOp.getLoc();
    MLIRContext *ctx = getOp.getContext();
    Type qubitType = quantum::QubitType::get(ctx);
    Type i64Type = builder.getI64Type();

    Value rQreg = getOp.getQreg();
    Value vQreg = currentQuregs.lookup(rQreg);

    quantum::ExtractOp extractOp;
    std::optional<uint64_t> idxAttr = getOp.getIdxAttr();
    if (idxAttr.has_value()) {
        extractOp = quantum::ExtractOp::create(builder, loc, qubitType, vQreg, {},
                                               IntegerAttr::get(i64Type, *idxAttr));
    }
    else {
        extractOp =
            quantum::ExtractOp::create(builder, loc, qubitType, vQreg, getOp.getIdx(), nullptr);
    }
    currentQubits.insert({getOp.getQubit(), extractOp.getQubit()});
}

void handleDealloc(IRRewriter &builder, qref::DeallocOp rDeallocOp,
                   llvm::DenseMap<Value, Value> &currentQubits,
                   llvm::DenseMap<Value, Value> &currentQuregs)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(rDeallocOp);
    Location loc = rDeallocOp.getLoc();

    Value insertedQreg =
        insertQubits(builder, rDeallocOp.getQreg(), currentQubits, currentQuregs, rDeallocOp);
    quantum::DeallocOp::create(builder, loc, insertedQreg);

    builder.eraseOp(rDeallocOp);
}

// Gate Ops

void handleGate(IRRewriter &builder, qref::QuantumOperation rGateOp,
                llvm::DenseMap<Value, Value> &currentQubits,
                llvm::DenseMap<Value, Value> &currentQuregs)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = rGateOp.getContext();

    SmallVector<Type> qubitResultsType;
    for (size_t i = 0; i < rGateOp.getQubitOperands().size(); i++) {
        qubitResultsType.push_back(quantum::QubitType::get(ctx));
    }

    quantum::QuantumOperation vGateOp;
    if (isa<qref::CustomOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::CustomOp>(builder, rGateOp, currentQubits,
                                                               currentQuregs, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }
    else if (isa<qref::PauliRotOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PauliRotOp>(builder, rGateOp, currentQubits,
                                                                 currentQuregs, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }
    else if (isa<qref::GlobalPhaseOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::GlobalPhaseOp>(
            builder, rGateOp, currentQubits, currentQuregs, qubitResultsType);
    }
    else if (isa<qref::MultiRZOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::MultiRZOp>(builder, rGateOp, currentQubits,
                                                                currentQuregs, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }
    else if (isa<qref::PCPhaseOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PCPhaseOp>(builder, rGateOp, currentQubits,
                                                                currentQuregs, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }
    else if (isa<qref::QubitUnitaryOp>(rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::QubitUnitaryOp>(
            builder, rGateOp, currentQubits, currentQuregs, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rGateOp));
    }

    for (auto [i, qubitReference] : llvm::enumerate(rGateOp.getQubitOperands())) {
        currentQubits[qubitReference] = vGateOp.getQubitResults()[i];
    }

    builder.eraseOp(rGateOp);
}

// Observable Ops

void handleNamedObs(IRRewriter &builder, qref::NamedObsOp rNamedObsOp,
                    llvm::DenseMap<Value, Value> &currentQubits,
                    llvm::DenseMap<Value, Value> &currentQuregs)
{
    OpBuilder::InsertionGuard guard(builder);

    auto vNamedObsOp = migrateOpToValueSemantics<quantum::NamedObsOp>(builder, rNamedObsOp,
                                                                      currentQubits, currentQuregs);
    builder.replaceOp(rNamedObsOp, vNamedObsOp);
}

// Control flow

void handleFor(IRRewriter &builder, scf::ForOp forOp, llvm::DenseMap<Value, Value> &currentQubits,
               llvm::DenseMap<Value, Value> &currentQuregs)
{
    // Insert all extracted qubits from the necessary registers, and update the maps
    OpBuilder::InsertionGuard guard(builder);
    Location loc = forOp->getLoc();
    MLIRContext *ctx = forOp.getContext();

    SetVector<Value> necessaryRegionRQregs;
    getNecessaryRegionRegisters(forOp, necessaryRegionRQregs);

    llvm::DenseMap<Value, Value> newForRegionCurrentQuregs;

    // 1. Insert all extracted qubits before the loop
    SmallVector<Value> newInitArgs(forOp.getInitArgs());
    builder.setInsertionPoint(forOp);
    for (Value rQreg : necessaryRegionRQregs) {
        Value insertedQreg = insertQubits(builder, rQreg, currentQubits, currentQuregs, forOp);
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
    for (auto rQreg : necessaryRegionRQregs) {
        assert(isa<qref::QuregType>(rQreg.getType()) &&
               "Expected the only extra loop arguments in value semantics to be quantum registers");
        Value vQreg =
            newLoop.getRegion().getBlocks().front().addArgument(quantum::QuregType::get(ctx), loc);
        newForRegionCurrentQuregs[rQreg] = vQreg;
    }

    llvm::DenseMap<Value, Value> newForRegionCurrentQubits;
    handleRegion(builder, newLoop.getRegion(), newForRegionCurrentQubits,
                 newForRegionCurrentQuregs);

    // 5. Insert loop region registers and Yield
    // builder.setInsertionPoint(newLoop.getRegion().getBlocks().back().getTerminator());
    auto yieldOp = cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
    SmallVector<Value> yieldOperands(yieldOp.getOperands());
    for (Value rQreg : necessaryRegionRQregs) {
        Value insertedQreg =
            insertQubits(builder, rQreg, newForRegionCurrentQubits, newForRegionCurrentQuregs,
                         newLoop.getBody()->getTerminator());
        yieldOperands.push_back(insertedQreg);
    }
    builder.setInsertionPoint(yieldOp);
    scf::YieldOp::create(builder, loc, yieldOperands);
    builder.eraseOp(yieldOp);

    // 6. New for loop's returned qreg is the new value semantics qreg value in the outer scope
    size_t numOldYields = forOp.getNumResults();
    for (auto [i, rQreg] : llvm::enumerate(necessaryRegionRQregs)) {
        currentQuregs[rQreg] = newLoop->getResult(numOldYields + i);
    }

    builder.eraseOp(forOp);
}

// Driver

void handleRegion(IRRewriter &builder, Region &r, llvm::DenseMap<Value, Value> &currentQubits,
                  llvm::DenseMap<Value, Value> &currentQuregs)
{

    r.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (auto rAllocOp = dyn_cast<qref::AllocOp>(op)) {
            handleAlloc(builder, rAllocOp, currentQuregs);
        }
        else if (auto getOp = dyn_cast<qref::GetOp>(op)) {
            handleGet(builder, getOp, currentQubits, currentQuregs);
        }
        else if (auto rGateOp = dyn_cast<qref::QuantumOperation>(op)) {
            handleGate(builder, rGateOp, currentQubits, currentQuregs);
        }
        else if (auto rNamedObsOp = dyn_cast<qref::NamedObsOp>(op)) {
            handleNamedObs(builder, rNamedObsOp, currentQubits, currentQuregs);
        }
        else if (auto rDeallocOp = dyn_cast<qref::DeallocOp>(op)) {
            handleDealloc(builder, rDeallocOp, currentQubits, currentQuregs);
        }
        else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            handleFor(builder, forOp, currentQubits, currentQuregs);
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
        // Location loc = mod->getLoc();
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
            // This map tracks: qref.bit Value -> current quantum.bit Value
            llvm::DenseMap<Value, Value> currentQubits;

            // This map tracks: qref.reg Value -> current quantum.reg Value
            llvm::DenseMap<Value, Value> currentQuregs;

            // The top level qnode functions will not have qreg or qubit values in their arguments,
            // so the initial map is just empty
            handleRegion(builder, qnodeFunc.getBody(), currentQubits, currentQuregs);

            // dumpMap(currentQubits);
            // dumpMap(currentQuregs);

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
