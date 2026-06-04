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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "PBC/IR/PBCOps.h"
#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/IR/QRefTypes.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Utils/QubitIndex.h"

using namespace mlir;
using namespace catalyst;

// In this file, variable names like "vQubit" stand for "qubits in value semantics",
// and variable names like "rQubit" stand for "qubits in reference semantics".

namespace {

void eraseAllRemainingAnchorVValues(func::FuncOp f)
{
    // f.walk([&](qref::GetOp getOp) {
    //     assert(getOp.use_empty() &&
    //            "qref.bit Values must have no uses after the semantic conversion");
    //     getOp->erase();
    // });
    f.walk([&](quantum::AllocOp allocOp) {
        assert(allocOp.use_empty() &&
               "quantum.reg Values must have no uses after the semantic conversion");
        allocOp->erase();
    });
    // f.walk([&](mbqc::RefGraphStatePrepOp graphStatePrepOp) {
    //     assert(graphStatePrepOp.use_empty() &&
    //            "qref.reg Values must have no uses after the semantic conversion");
    //     graphStatePrepOp->erase();
    // });
}

struct QubitValueTracker {
  public:
    QubitValueTracker() = default;

    void setRQreg(Value vQreg, Value rQreg)
    {
        assert(isa<quantum::QuregType>(vQreg.getType()) && "Expected quantum.reg type");
        assert(isa<qref::QuregType>(rQreg.getType()) && "Expected qref.reg type");
        this->qreg_map[vQreg] = rQreg;
    }

    Value getRQreg(Value vQreg)
    {
        assert(isa<quantum::QuregType>(vQreg.getType()) && "Expected quantum.reg type");

        Value rQreg = this->qreg_map.at(vQreg);
        assert(isa<qref::QuregType>(rQreg.getType()) && "Expected qref.reg type");
        return rQreg;
    }

  private:
    llvm::DenseMap<Value, Value> qreg_map;
    llvm::DenseMap<Value, quantum::QubitIndex> qubit_map;
}; // struct QubitValueTracker

void handleAlloc(IRRewriter &builder, quantum::AllocOp vAllocOp, QubitValueTracker &tracker)
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
}

void handleDealloc(IRRewriter &builder, quantum::DeallocOp vDeallocOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(vDeallocOp);
    Location loc = vDeallocOp.getLoc();

    qref::DeallocOp::create(builder, loc, tracker.getRQreg(vDeallocOp.getQreg()));

    builder.eraseOp(vDeallocOp);
}

void handleRegion(IRRewriter &builder, Region &r, QubitValueTracker &tracker)
{
    r.walk<WalkOrder::PreOrder>([&](Operation *op) {
        llvm::TypeSwitch<Operation *, void>(op)
            .Case<quantum::AllocOp>([&](auto o) { handleAlloc(builder, o, tracker); })
            .Case<quantum::DeallocOp>([&](auto o) { handleDealloc(builder, o, tracker); })
            // .Case<qref::AllocQubitOp>([&](auto o) { handleAllocQubit(builder, o, tracker); })
            // .Case<qref::DeallocQubitOp>([&](auto o) { handleDeallocQubit(builder, o, tracker); })
            // .Case<qref::QuantumOperation>([&](auto o) { handleGate(builder, o, tracker); })
            // .Case<func::CallOp>([&](auto o) { handleCall(builder, o, tracker); })
            // .Case<qref::ComputationalBasisOp>([&](auto o) { handleCompbasis(builder, o, tracker);
            // }) .Case<qref::NamedObsOp>([&](auto o) { handleNamedObs(builder, o, tracker); })
            // .Case<qref::HermitianOp>([&](auto o) { handleHermitian(builder, o, tracker); })
            // .Case<qref::MeasureOp>([&](auto o) { handleMeasure(builder, o, tracker); })
            // .Case<mbqc::RefMeasureInBasisOp>(
            //     [&](auto o) { handleMeasureInBasis(builder, o, tracker); })
            // .Case<pbc::RefPPMeasurementOp>([&](auto o) { handlePPM(builder, o, tracker); })
            // .Case<qref::AdjointOp>([&](auto o) { handleAdjoint(builder, o, tracker); })
            // .Case<scf::IfOp>([&](auto o) { handleIf(builder, o, tracker); })
            // .Case<scf::IndexSwitchOp>([&](auto o) { handleSwitch(builder, o, tracker); })
            // .Case<scf::ForOp>([&](auto o) { handleFor(builder, o, tracker); })
            // .Case<scf::WhileOp>([&](auto o) { handleWhile(builder, o, tracker); })
            // .Case<mbqc::RefGraphStatePrepOp>(
            //     [&](auto o) { handleGraphStatePrep(builder, o, tracker); })
            .Default([](Operation *) {});
    });
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
            eraseAllRemainingAnchorVValues(targetFunc);
        }
    }
};

} // namespace quantum
} // namespace catalyst
