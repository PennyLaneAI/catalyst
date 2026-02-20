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

//
// Misc helpers
//

template <typename OpTy>
OpTy migrateOpToValueSemantics(IRRewriter &builder, Operation *qrefOp,
                               const llvm::DenseMap<Value, Value> &currentQubits,
                               std::optional<TypeRange> newResultTypes = std::nullopt)
{
    // Given a reference semantics operation instance, migrate it to value semantics.
    // We create the corresponding value semantics operation, with exactly the same operands and
    // attributes, except we replace the reference qubit SSA Values with the value qubit SSA Values.
    // The result types of the value semantics operation will be the same as the old one, unless
    // explicitly overriden.

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(qrefOp);

    // Create the new op using the generic state-based approach
    // We cannot just clone, since we are changing the op type
    OperationState state(qrefOp->getLoc(), OpTy::getOperationName());

    SmallVector<Value> vOperands;
    for (Value v : qrefOp->getOperands()) {
        if (isa<qref::QubitType>(v.getType())) {
            assert(currentQubits.contains(v) && "The qubit reference is missing a qubit SSA value");
            vOperands.push_back(currentQubits.lookup(v));
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

Value insertQubits(
    IRRewriter &builder, Value qreg,
    const SmallVector<std::pair<Value, std::variant<Value, IntegerAttr>>> &qubitIdxPairs,
    Operation *insertionPoint)
{
    // Create a chain of insert ops, inserting all qubits into the qreg.
    assert(isa<quantum::QuregType>(qreg.getType()) && "Expected a value semantics qreg SSA value");

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertionPoint);

    Location loc = qreg.getLoc();
    MLIRContext *ctx = qreg.getContext();

    // Perform the inserts
    Value current = qreg;
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

    return current;
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

    Type i64Type = builder.getI64Type();

    Value rSourceQreg = rDeallocOp.getQreg();
    Value currentVQreg = currentQuregs.lookup(rSourceQreg);

    // Collect all qref qubits that come from the register
    // A register is either used by a qref.get, or a region op (adjoint, control flow, ...)
    // Only getOps produce qubits
    SmallVector<std::pair<Value, std::variant<Value, IntegerAttr>>> qubitIdxPairs;
    for (Operation *user : rSourceQreg.getUsers()) {
        if (auto getOp = dyn_cast<qref::GetOp>(user)) {
            Value currentVQubit = currentQubits.lookup(getOp.getQubit());
            std::optional<uint64_t> idxAttr = getOp.getIdxAttr();
            if (idxAttr.has_value()) {
                qubitIdxPairs.push_back({currentVQubit, IntegerAttr::get(i64Type, *idxAttr)});
            }
            else {
                qubitIdxPairs.push_back({currentVQubit, getOp.getIdx()});
            }
        }
        else {
            continue;
        }
    }

    Value insertedQreg = insertQubits(builder, currentVQreg, qubitIdxPairs, rDeallocOp);
    quantum::DeallocOp::create(builder, loc, insertedQreg);

    builder.eraseOp(rDeallocOp);
}

// Gate Ops

void handleGate(IRRewriter &builder, qref::QuantumOperation rGateOp,
                llvm::DenseMap<Value, Value> &currentQubits)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = rGateOp.getContext();

    SmallVector<Type> qubitResultsType;
    for (size_t i = 0; i < rGateOp.getQubitOperands().size(); i++) {
        qubitResultsType.push_back(quantum::QubitType::get(ctx));
    }

    quantum::QuantumOperation vGateOp;

    llvm::TypeSwitch<qref::QuantumOperation, void>(rGateOp)
        .Case([&](qref::CustomOp rCustomOp) {
            vGateOp = migrateOpToValueSemantics<quantum::CustomOp>(builder, rGateOp, currentQubits,
                                                                   qubitResultsType);
            vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rCustomOp));
        })
        .Default([&](Operation *op) {
            // other operations - do nothing
        });

    for (auto [i, qubitReference] : llvm::enumerate(rGateOp.getQubitOperands())) {
        currentQubits[qubitReference] = vGateOp.getQubitResults()[i];
    }

    builder.eraseOp(rGateOp);
}

// Observable Ops

void handleNamedObs(IRRewriter &builder, qref::NamedObsOp rNamedObsOp,
                    llvm::DenseMap<Value, Value> &currentQubits)
{
    OpBuilder::InsertionGuard guard(builder);

    auto vNamedObsOp =
        migrateOpToValueSemantics<quantum::NamedObsOp>(builder, rNamedObsOp, currentQubits);
    builder.replaceOp(rNamedObsOp, vNamedObsOp);
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
        // Location loc = mod->getLoc();
        IRRewriter builder(mod->getContext());

        // This map tracks: qref.bit Value -> current quantum.bit Value
        llvm::DenseMap<Value, Value> currentQubits;

        // This map tracks: qref.reg Value -> current quantum.reg Value
        llvm::DenseMap<Value, Value> currentQuregs;

        mod->walk<WalkOrder::PreOrder>([&](Operation *op) {
            if (auto rAllocOp = dyn_cast<qref::AllocOp>(op)) {
                handleAlloc(builder, rAllocOp, currentQuregs);
            }
            else if (auto getOp = dyn_cast<qref::GetOp>(op)) {
                handleGet(builder, getOp, currentQubits, currentQuregs);
            }
            else if (auto rGateOp = dyn_cast<qref::QuantumOperation>(op)) {
                handleGate(builder, rGateOp, currentQubits);
            }
            else if (auto rNamedObsOp = dyn_cast<qref::NamedObsOp>(op)) {
                handleNamedObs(builder, rNamedObsOp, currentQubits);
            }
            else if (auto rDeallocOp = dyn_cast<qref::DeallocOp>(op)) {
                handleDealloc(builder, rDeallocOp, currentQubits, currentQuregs);
            }
        });

        for (auto pair : currentQubits) {
            builder.eraseOp(pair.first.getDefiningOp());
        }
        for (auto pair : currentQuregs) {
            builder.eraseOp(pair.first.getDefiningOp());
        }
    }
};

} // namespace qref
} // namespace catalyst
