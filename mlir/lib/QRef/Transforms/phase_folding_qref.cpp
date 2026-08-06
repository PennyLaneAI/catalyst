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

// This algorithm is taken from https://arxiv.org/pdf/1303.2042

#define DEBUG_TYPE "phase_folding_qref"

#include <cassert>
#include <cmath> // std::abs
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // arith::ConstantOp
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h" // mlir::matchPattern, mlir::m_Constant
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinOps.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "QRef/IR/QRefOps.h"

#include "SymbolicAnalysisQRef/ProgramAbstraction.hpp"
#include "SymbolicAnalysisQRef/RegionSummary.hpp"
#include "SymbolicAnalysisQRef/Gate.hpp"


using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace {

class PhaseFoldingAnalyzer {
  public:
    PhaseFoldingAnalyzer() = default;

    void analyzeFunction(mlir::func::FuncOp funcOp);
    void dumpSummaries();

    ProgramAbstraction &mainProgramAbstraction() { return mainProgramAbst; }
    std::vector<qref::CustomOp> &phaseOperations() { return phaseOps; }
    
  private:
    ProgramAbstraction mainProgramAbst;

    std::vector<qref::CustomOp> phaseOps;
    GateID gateID = -1;

    // llvm::StringMap<RegionSummary> procedureSummaries; // keyed by procedure name
    llvm::DenseMap<mlir::StringRef, RegionSummary> procedureSummaries; // keyed by procedure name

    llvm::DenseMap<mlir::Value, size_t> ssaToWireMap;
    llvm::DenseMap<mlir::Value, size_t> qregToBaseMap;

    static constexpr std::string_view mainFuncIndicator = "quantum.node";

    void analyzeOperation(mlir::Operation *op, ProgramAbstraction &currentAbst);
    void analyzeBlock(mlir::Block *block, ProgramAbstraction &currentAbst);

    void handleIfOp(mlir::scf::IfOp ifOp, ProgramAbstraction &parentAbst);
    void handleForOp(mlir::scf::ForOp forOp, ProgramAbstraction &parentAbst);
    void handleCallOp(mlir::func::CallOp callOp, ProgramAbstraction &parentAbst);

    bool isTrackedDialectOp(mlir::Operation *op);
    void handleCustomOp(qref::CustomOp customOp, ProgramAbstraction &progAbst, GateID &gateID);    // quantum ops
    void allocateRegister(mlir::Value qreg, auto regSize, ProgramAbstraction &progAbst);
    void allocateQubit(mlir::Value qubit, ProgramAbstraction &progAbst);
    void deallocateRegister(mlir::Value qreg);
    void deallocateQubit(mlir::Value qubit);
    void getFromQreg(qref::GetOp getOp);
    void initQubitsState(qref::SetBasisStateOp basisOp, ProgramAbstraction &progAbst);
    // void applyUndefinedOp(Operation *op, ProgramAbstraction &progAbst);
    
    llvm::SmallVector<size_t, 4> getWireIndices(mlir::ValueRange qubitValues);
    mlir::DenseElementsAttr extractBasisState(qref::SetBasisStateOp basisOp);
    Gate extractCliffTGate(qref::CustomOp &op);
};

void PhaseFoldingAnalyzer::analyzeFunction(mlir::func::FuncOp funcOp) {
    mlir::StringRef funcName = funcOp.getName();
    if (procedureSummaries.count(funcName)) return;

    ProgramAbstraction funcAbst; //(computeNumQubits(funcOp));
    analyzeBlock(&funcOp.getBody().front(), funcAbst);

    // TODO: what if we had multiple qnodes? better way of specifying the main function?
    if (funcOp->hasAttrOfType<mlir::UnitAttr>(mainFuncIndicator)) {
        mainProgramAbst = funcAbst; // is it a deep copy?
    }    
    procedureSummaries[funcName] = RegionSummary(RegionType::Procedure, funcAbst);

    // Functions typically have a single block in their body region
    // if (!funcOp.getBody().empty()) {
    // }
}

void PhaseFoldingAnalyzer::analyzeBlock(mlir::Block *block, ProgramAbstraction &currentAbst) {
    for (mlir::Operation &op : *block) {
        analyzeOperation(&op, currentAbst);
    }
    // llvm::outs() << "\nblock: \n" << currentAbst << "\n";
}

inline bool PhaseFoldingAnalyzer::isTrackedDialectOp(mlir::Operation *op) {
    mlir::Dialect *dialect = op->getDialect();
    if (!dialect) {
        return false;
    }
    return isa<qref::QRefDialect, scf::SCFDialect>(dialect);
}

void PhaseFoldingAnalyzer::analyzeOperation(mlir::Operation *op, ProgramAbstraction &currentAbst) {
    if (!isTrackedDialectOp(op)) {
        op->emitError("Operation is not a tracked dialect operation: " + op->getName().getStringRef());
        return;
    }
    llvm::TypeSwitch<mlir::Operation *, void>(op)
        .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp ifOp) {
            llvm::outs() << "IfOp:  " << "\n";
            handleIfOp(ifOp, currentAbst);
        })
        .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp forOp) {
            llvm::outs() << "ForOp: " << "\n";
            handleForOp(forOp, currentAbst);
        })
        .Case<mlir::func::CallOp>([&](mlir::func::CallOp callOp) {
            llvm::outs() << "CallOp:    " << callOp << "\n";
            handleCallOp(callOp, currentAbst);
        })
        .Case<qref::CustomOp>([&](qref::CustomOp customOp) {
            llvm::outs() << "CustomOp:  " << customOp << "\n";
            handleCustomOp(customOp, currentAbst, gateID);
        })
        .Case<qref::AllocOp>([&](qref::AllocOp allocOp) {
            llvm::outs() << "AllocOp:   " << allocOp << "\n";
            allocateRegister(allocOp.getResult(), allocOp.getNqubitsAttr(), currentAbst);
        })
        .Case<qref::AllocQubitOp>([&](qref::AllocQubitOp allocQbOp) {
            llvm::outs() << "AllocQubitOp:  " << allocQbOp << "\n";
            allocateQubit(allocQbOp.getResult(), currentAbst);
        })
        .Case<qref::DeallocOp>([&](qref::DeallocOp deallocOp) {
            llvm::outs() << "DeallocOp:   " << deallocOp << "\n";
            deallocateRegister(deallocOp.getQreg());
        })
        .Case<qref::DeallocQubitOp>([&](qref::DeallocQubitOp deallocQbOp) {
            llvm::outs() << "DeallocQubitOp:   " << deallocQbOp << "\n";
            deallocateQubit(deallocQbOp.getQubit());
        })
        .Case<qref::GetOp>([&](qref::GetOp getOp) {
            llvm::outs() << "GetOp: " << getOp << "\n";
            getFromQreg(getOp);
        })
        .Case<qref::SetBasisStateOp>([&](qref::SetBasisStateOp basisOp) {
            llvm::outs() << "SetBasisStateOp:   " << basisOp << "\n";
            initQubitsState(basisOp, currentAbst);
        })
        .Case<qref::MeasureOp>([&](qref::MeasureOp measureOp) {
            llvm::outs() << "MeasureOp:  " << measureOp << "\n";
        })
        .Case<qref::GlobalPhaseOp>([&](qref::GlobalPhaseOp gpOp) {
            llvm::outs() << "GlobalPhaseOp: " << gpOp << "\n";
        })
        .Case<mlir::scf::YieldOp>([&](mlir::scf::YieldOp yieldOp) {
            llvm::outs() << "YieldOp:  " << yieldOp << "\n";
            
        })
        .Default([&](mlir::Operation *unknownOp) {
            // Handle or ignore operations that don't affect phases (e.g., standard arithmetic)
            llvm::outs() << "UnknownOp: " << *unknownOp << "\n";
            // applyUndefinedOp(unknownOp, currentAbst);
        });
}

// --- Control Flow Handlers ---

void PhaseFoldingAnalyzer::handleIfOp(mlir::scf::IfOp ifOp, ProgramAbstraction &parentAbst) {
    // 1. Create fresh abstractions for branches
    ProgramAbstraction thenAbst(parentAbst.numQubits());
    ProgramAbstraction elseAbst(parentAbst.numQubits());

    // 2. Recursively analyze regions
    analyzeBlock(&ifOp.getThenRegion().front(), thenAbst);
    
    if (!ifOp.getElseRegion().empty()) {
        analyzeBlock(&ifOp.getElseRegion().front(), elseAbst);
    }

    // llvm::outs() << "thenAbst: " << thenAbst << "\n";
    // llvm::outs() << "elseAbst: " << elseAbst << "\n";
    // llvm::outs() << "parentAbst: " << parentAbst << "\n";
    // 3. Compute summary using your API
    RegionSummary branchSummary(RegionType::Conditional, thenAbst, &elseAbst);
    // llvm::outs() << "branchSummary: " << branchSummary << "\n";
    // 4. Apply to parent
    parentAbst.applySummary(std::move(branchSummary));
    // llvm::outs() << "parentAbst after applySummary: " << parentAbst << "\n";
}

void PhaseFoldingAnalyzer::handleForOp(mlir::scf::ForOp forOp, ProgramAbstraction &parentAbst) {
    // 1. Create abstraction for the loop body
    ProgramAbstraction loopAbst(parentAbst.numQubits());

    // 2. Analyze the loop body region
    analyzeBlock(&forOp.getBodyRegion().front(), loopAbst);

    // 3. Compute summary 
    RegionSummary loopSummary(RegionType::Loop, loopAbst);

    // 4. Apply to parent
    parentAbst.applySummary(std::move(loopSummary));
}

void PhaseFoldingAnalyzer::handleCallOp(mlir::func::CallOp callOp, ProgramAbstraction &parentAbst) {
    // // For procedures, you either do inter-procedural analysis by jumping 
    // // to the callee, or you look up a pre-computed RegionSummary for the callee.
    // // Assuming you compute inter-procedurally on the fly:
    // ProgramAbstraction procAbst(parentAbst.numQubits());
    
    // mlir::func::FuncOp callee = getCallee(callOp); // pseudo-code
    // analyzeBlock(&callee.getBody().front(), procAbst);
    
    // RegionSummary summary(RegionType::Procedure, procAbst);
    // parentAbst.applySummary(std::move(summary));

    // if getting from the stored functions, should copy the summary, since applySummary consumes it!
}

// --- Qubit Management ---

inline void PhaseFoldingAnalyzer::allocateRegister(mlir::Value qreg, auto regSize, ProgramAbstraction &progAbst)
{   // TODO: size 0 register! -- auto_qubit_management mode
    qregToBaseMap[qreg] = progAbst.numQubits();
    progAbst.extendQubitsBy(static_cast<size_t>(regSize.value_or(0)));
}

inline void PhaseFoldingAnalyzer::allocateQubit(mlir::Value qubit, ProgramAbstraction &progAbst)
{
    ssaToWireMap[qubit] = progAbst.numQubits();
    progAbst.extendQubitsBy(1);
}

inline void PhaseFoldingAnalyzer::deallocateRegister(mlir::Value qreg)
{
    qregToBaseMap.erase(qreg);
}

inline void PhaseFoldingAnalyzer::deallocateQubit(mlir::Value qubit)
{
    ssaToWireMap.erase(qubit);
}

void PhaseFoldingAnalyzer::getFromQreg(qref::GetOp getOp)
{
    mlir::Value qreg = getOp.getQreg();
    auto regIt = qregToBaseMap.find(qreg);
    if (regIt == qregToBaseMap.end()) {
        assert(false && "Error: GetOp references an untracked register.\n");
    }
    size_t baseIndex = regIt->second;

    auto staticIdx = getOp.getIdxAttr();
    if (!staticIdx.has_value()) {
        // auto dynamicIdx = getOp.getIdx();
        assert(false && "Error: Dynamic qubit extraction indices are not supported.\n");
    }
    ssaToWireMap[getOp.getQubit()] = baseIndex + static_cast<size_t>(staticIdx.value_or(0));
}

mlir::DenseElementsAttr PhaseFoldingAnalyzer::extractBasisState(qref::SetBasisStateOp basisOp)
{
    mlir::Value basisStateTensor = basisOp.getBasisState();
    mlir::Operation *defOp = basisStateTensor.getDefiningOp();

    if (auto constOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(defOp)) {
        if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
            return denseAttr;
        }
        else {
            assert(false && "Error: Basis state constant is not a DenseElementsAttr.\n");
        }
    }
    else {
        assert(false && "Error: Dynamic basis state initialization is not supported.\n");
    }
}

void PhaseFoldingAnalyzer::initQubitsState(qref::SetBasisStateOp basisOp, ProgramAbstraction &progAbst)
{
    llvm::SmallVector<size_t, 4> wires = getWireIndices(basisOp.getQubits());
    mlir::DenseElementsAttr basisState = extractBasisState(basisOp);

    assert(static_cast<size_t>(basisState.getNumElements()) == wires.size());

    for (const auto &[idx, val] : llvm::zip(wires, basisState.getValues<llvm::APInt>())) {
        progAbst.prepareQubit(idx, val.getBoolValue());
    }
}

// void PhaseFoldingAnalyzer::applyUndefinedOp(Operation *op, ProgramAbstraction &progAbst)
// {
//     llvm::SmallVector<size_t, 4> qubitIndices;
//     if (auto qGate = dyn_cast<QuantumGate>(op)) {
//         qubitIndices =
//             getWireIndices(qGate.getNonCtrlQubitOperands(), qGate.getNonCtrlQubitResults());
//     }
//     else if (auto stateOp = dyn_cast<SetStateOp>(op)) {
//         qubitIndices = getWireIndices(stateOp.getInQubits(), stateOp.getOutQubits());
//     }
//     else {
//         // op->emitError("Not supported");
//         return;
//     }

//     Gate gate = (isa<MultiRZOp>(op) || isa<PCPhaseOp>(op)) ? Gate::I : Gate::U;
//     progAbst.applyGate(gate, false, qubitIndices);
// }

llvm::SmallVector<size_t, 4> PhaseFoldingAnalyzer::getWireIndices(mlir::ValueRange qubitValues)
{
    size_t n = qubitValues.size();

    llvm::SmallVector<size_t, 4> indices;
    indices.reserve(n);

    for (size_t i = 0; i < n; i++) {
        mlir::Value inValue = qubitValues[i];

        size_t wire;
        auto it = ssaToWireMap.find(inValue);
        if (it == ssaToWireMap.end()) {
            assert(false && "Error: Operation references an untracked value.\n");
        }
        wire = it->second;
        indices.push_back(wire);
    }
    return indices;
}

Gate PhaseFoldingAnalyzer::extractCliffTGate(qref::CustomOp &op)
{
    Gate gate = gateWithName(op.getGateName());
    if (!op.getCtrlQubits().empty() || !op.getCtrlValues().empty()) {
        if (isPhaseGate(gate)) {
            return Gate::I; // C-Rz gates don't alter state space, but alter phase space
                            // non-linearly, which I'm not going to track for now, but should be
                            // trackable using xy = x + y - (x \oplus y).
        }
        // if (gate == Gate::X && op.getCtrlQubits().size() == 1 &&
        // op.getCtrlValues().empty()) {
        //     return Gate::CNOT;
        // }    should pass getCtrlQubit as qubitIn. will do it later.
        return Gate::U;
    }
    return gate;
}
    
void PhaseFoldingAnalyzer::handleCustomOp(qref::CustomOp customOp, ProgramAbstraction &progAbst, GateID &gateID)
{
    llvm::SmallVector<size_t, 4> wires = getWireIndices(customOp.getQubits());
    Gate gate = extractCliffTGate(customOp);

    if (isPhaseGate(gate)) {
        phaseOps.push_back(customOp);
        gateID++;
    }
    initialGateCount[static_cast<size_t>(gate)]++;

    progAbst.applyGate(gate, customOp.getAdjointFlag(), wires, gateID);
}

void PhaseFoldingAnalyzer::dumpSummaries()
{
    llvm::outs() << "\nAll Summaries:\n";
    for (auto &[funcName, summary] : procedureSummaries) {
        llvm::outs() << funcName << "\n";
        llvm::outs() << summary << "\n";
    }
    llvm::outs() << "\nMain Program Abstraction:\n" << mainProgramAbstraction() << "\n\n";
}

}   // namespace

namespace catalyst {
namespace qref {

#define GEN_PASS_DECL_PHASEFOLDINGQREFPASS
#define GEN_PASS_DEF_PHASEFOLDINGQREFPASS
#include "QRef/Transforms/Passes.h.inc"

struct PhaseFoldingQRefPass : public impl::PhaseFoldingQRefPassBase<PhaseFoldingQRefPass> {
    using impl::PhaseFoldingQRefPassBase<PhaseFoldingQRefPass>::PhaseFoldingQRefPassBase;

    // Gate Statistics:
    void updateStats(Gate gate, int incr) { insertedGateCount[static_cast<size_t>(gate)] += incr; }

    void reportStats()
    {
        llvm::outs() << "Stats:\n";
        for (size_t i = 0; i < PRIMITIV_GATES_COUNT; i++) {
            if (insertedGateCount[i] != 0) {
                llvm::outs() << GATE_NAME[i] << ": initial-> " << initialGateCount[i]
                             << ",  final-> " << (initialGateCount[i] + insertedGateCount[i])
                             << ". difference-> " << insertedGateCount[i] << "\n";
            }
        }
        llvm::outs() << "\n";
    }  

    // Rotation Angle Computations:
    double getPhase(qref::CustomOp &op)
    {
        double c = (op.getAdjointFlag() ? -1 : 1);
        double angle = rotAngle(gateWithName(op.getGateName()));

        return ((angle != UNKNOWN_ANGLE ? angle : extractRZAngle(op.getParams())) * c);
    }

    double extractRZAngle(mlir::ValueRange params)
    {
        if (params.empty()) {
            return 0.0;
        }

        mlir::FloatAttr floatAttr;
        if (mlir::matchPattern(params.front(), mlir::m_Constant(&floatAttr))) {
            return floatAttr.getValueAsDouble();
        }
        else { // dynamic param
            return 0.0;
        }
    }

    double sumAngles(const GateBundle &contributors, std::vector<qref::CustomOp> &phaseOps)
    {
        double sum = 0.0;
        for (GateID id : contributors.zeroAffineGates) {
            sum += getPhase(phaseOps[id]);
        }
        for (GateID id : contributors.oneAffineGates) {
            sum -= getPhase(phaseOps[id]);
        }
        return sum;
    }

    double normalizeAngle(double angle)
    { // returns equivalent angle between [-PI, PI]
        return std::remainder(angle, 2 * PI);
    }

    // IR Modifications:
    void updateTargetOp(qref::CustomOp &targetOp, double sumAngle)
    {
        double normAngle = normalizeAngle(sumAngle);
        bool isAdjoint = (normAngle < 0.0);
        normAngle = std::abs(normAngle);
        Gate gate = gateWithAngle(normAngle);

        if (gate == Gate::I) {
            killOp(targetOp);
        }
        else {
            replaceOpWith(targetOp, gate, isAdjoint, normAngle);
        }
    }

    void removePhaseOp(qref::CustomOp &op)
    {
        if (gateWithName(op.getGateName()) == Gate::Y) {
            replaceOpWith(op, Gate::X, false, 0.0);
        }
        else {
            killOp(op);
        }
    }

    void replaceOpWith(qref::CustomOp &preOp, Gate newGate, bool isAdjoint, double angle)
    {
        updateStats(gateWithName(preOp.getGateName()), -1);

        mlir::IRRewriter rewriter(preOp.getContext());
        rewriter.setInsertionPoint(preOp.getOperation());

        mlir::ValueRange params;
        if (newGate == Gate::RZ) {
            mlir::Value angleVal = arith::ConstantOp::create(rewriter, preOp.getLoc(),
                                                 rewriter.getF64FloatAttr(angle));
            params = mlir::ValueRange({angleVal});
        }

        rewriter.replaceOpWithNewOp<qref::CustomOp>(
            preOp, 
            /*params=*/ params,
            /*qubits=*/ preOp.getQubits(), 
            /*gate_name=*/GATE_NAME[static_cast<size_t>(newGate)], 
            /*adjoint=*/ isAdjoint,
            preOp.getCtrlQubits(),
            preOp.getCtrlValues());

        updateStats(newGate, +1);
    }

    void killOp(qref::CustomOp &op)
    {
        assert(op.getCtrlQubits().empty() &&
               op.getCtrlValues().empty()); // move to somewhere better
        updateStats(gateWithName(op.getGateName()), -1);
        op.erase();
    }

    // Phase-folding Algorithm:
    void phaseMerge(ProgramAbstraction &progAbst, std::vector<qref::CustomOp> &phaseOps)
    {
        auto removeGates = [&](auto gates,
                               std::optional<GateID> skipID = std::nullopt) {
            for (GateID id : gates) {
                if (id != skipID) {
                    removePhaseOp(phaseOps[id]);
                }
            }
        };

        auto tryMergeBundle = [&](GateBundle &bundle) {
            if (bundle.gateCount() > 1) {
                double angleSum = sumAngles(bundle, phaseOps);
                if (!bundle.isMergeTargetAffineZero()) {
                    angleSum = -angleSum;
                }

                GateID targetOpID = bundle.getMergeTarget();
                updateTargetOp(phaseOps[targetOpID], angleSum);
                removeGates(bundle.getAllGates(), targetOpID);
            }
        };

        for (auto &[parity, contributors] : progAbst.phases.activeBundles) {
            if (parity.isTrivial()) {
                removeGates(contributors.zeroAffineGates);
                contributors.zeroAffineGates.clear();
            }
            tryMergeBundle(contributors);
        }

        for (GateBundle &bundle : progAbst.phases.orphanBundles) {
            tryMergeBundle(bundle);
        }
    }

    void runOnOperation() override 
    {  
        llvm::outs() << "Hello phase-folding-QRef world!\n";

        PhaseFoldingAnalyzer analyzer;
        mlir::ModuleOp rootModule = dyn_cast<mlir::ModuleOp>(getOperation());

        for (auto func : rootModule.getOps<mlir::func::FuncOp>()) {
            // Skip external function declarations (functions without bodies)
            llvm::outs() << "FuncOp:\n" << func << "\n";
            if (!func.isExternal()) {   // only analyze functions with bodies
                analyzer.analyzeFunction(func);
            }
        }

        analyzer.dumpSummaries();

        phaseMerge(analyzer.mainProgramAbstraction(), analyzer.phaseOperations());

        reportStats();
    }
};

} // namespace qref
} // namespace catalyst


// test summary computation with nested control-flow
// test with different qubit number in different blocks
// dealloc, return, measure


// each module will have a single qnode, but a program can have multiple modules.