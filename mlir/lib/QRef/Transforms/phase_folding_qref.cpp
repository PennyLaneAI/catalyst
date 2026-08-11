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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // arith::ConstantOp
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h" // mlir::matchPattern, mlir::m_Constant
#include "mlir/Pass/Pass.h"
#include "Catalyst/IR/CatalystDialect.h"
#include "QRef/IR/QRefOps.h"

#include "GetOpInfo.hpp"
#include "PhaseFolding/ProgramAbstraction.hpp"
#include "PhaseFolding/RegionSummary.hpp"
#include "PhaseFolding/Gate.hpp"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace {

struct WireTable {
  public:
    size_t size() const { return nextWire; }
  
    void populate(mlir::func::FuncOp funcOp) {
        funcOp.walk([&](Operation *op) {
            if (auto getOp = dyn_cast<qref::GetOp>(op)) {
                wireForGet(getOp);
            }
            else if (auto allocQbOp = dyn_cast<qref::AllocQubitOp>(op)) {
                wireForAllocQubit(allocQbOp);
            }
        });
    }

    llvm::SmallVector<size_t, 4> getQubitWires(mlir::ValueRange qubitValues)
    {
        size_t n = qubitValues.size();

        llvm::SmallVector<size_t, 4> wires;
        wires.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            wires.push_back(wireForOperand(qubitValues[i])); 
        }
        return wires;
    }

  private:
    llvm::DenseMap<rQubitGetOpInfo, size_t> getOpToWireMap;
    llvm::DenseMap<mlir::Value, size_t> allocQubitToWireMap;
    
    size_t nextWire = 0;

    rQubitGetOpInfo mkGetOpInfo(qref::GetOp getOp)
    {
        auto staticIdx = getOp.getIdxAttr();
        if (staticIdx.has_value()) {
            return rQubitGetOpInfo(getOp.getQreg(), static_cast<int64_t>(staticIdx.value_or(0)));
        } else {
            return rQubitGetOpInfo(getOp.getQreg(), getOp.getIdx());
        }
    }

    size_t wireForGet(qref::GetOp getOp) 
    {
        auto [it, inserted] = getOpToWireMap.try_emplace(mkGetOpInfo(getOp), nextWire);
        if (inserted) {
            ++nextWire;
        }
        return it->second;
    }

    size_t wireForAllocQubit(qref::AllocQubitOp allocQbOp) 
    { 
        auto [it, inserted] = allocQubitToWireMap.try_emplace(allocQbOp.getResult(), nextWire);
        if (inserted) {
            ++nextWire;
        }
        return it->second;
    }

    size_t wireForOperand(mlir::Value qubit) 
    {
        auto definingOp = qubit.getDefiningOp();
        if (auto getOp = dyn_cast<qref::GetOp>(definingOp)) {
            return wireForGet(getOp);
        }
        if (auto allocQbOp = dyn_cast<qref::AllocQubitOp>(definingOp)) {
            return wireForAllocQubit(allocQbOp);
        }
        assert(false && "Untracked operand wire");  // function arguments?
        return 0;
    }
};

struct PhaseFoldingAnalyzer {
  public:
    PhaseFoldingAnalyzer() = default;

    ProgramAbstraction &mainProgramAbstraction() { return mainProgramAbst; }
    std::vector<qref::CustomOp> &phaseOperations() { return phaseOps; }
    
  private:
    ProgramAbstraction mainProgramAbst;

    std::vector<qref::CustomOp> phaseOps;
    GateID gateID = -1;

    llvm::StringMap<RegionSummary> procedureSummaries; // keyed by procedure name

    static constexpr std::string_view mainFuncIndicator = "quantum.node";
    
    // --- Helper Functions ---

    mlir::DenseElementsAttr extractBasisState(qref::SetBasisStateOp basisOp)
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

    Gate extractCliffTGate(qref::CustomOp &op)
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

    bool isTrackedDialectOp(mlir::Operation *op)
    {
        mlir::Dialect *dialect = op->getDialect();
        if (!dialect) {
            return false;
        }
        return isa<qref::QRefDialect, scf::SCFDialect>(dialect);
    }

    // --- Quantum Operations ---

    void handleCustomOp(qref::CustomOp customOp, ProgramAbstraction &progAbst, GateID &gateID, WireTable &wireTable)
    {
        llvm::SmallVector<size_t, 4> wires = wireTable.getQubitWires(customOp.getQubits());
        Gate gate = extractCliffTGate(customOp);
    
        if (isPhaseGate(gate)) {
            phaseOps.push_back(customOp);
            gateID++;
        }
        initialGateCount[static_cast<size_t>(gate)]++;
    
        progAbst.applyGate(gate, customOp.getAdjointFlag(), wires, gateID);
    }

    void initQubitsState(qref::SetBasisStateOp basisOp, ProgramAbstraction &progAbst, WireTable &wireTable)
    {
        llvm::SmallVector<size_t, 4> wires = wireTable.getQubitWires(basisOp.getQubits());
        mlir::DenseElementsAttr basisState = extractBasisState(basisOp);
    
        assert(static_cast<size_t>(basisState.getNumElements()) == wires.size());
    
        for (const auto &[idx, val] : llvm::zip(wires, basisState.getValues<llvm::APInt>())) {
            progAbst.prepareQubit(idx, val.getBoolValue());
        }
    }

    // void applyUndefinedOp(Operation *op, ProgramAbstraction &progAbst, WireTable &wireTable)
    // {
    //     llvm::SmallVector<size_t, 4> wires;
    //     if (auto qGate = dyn_cast<QuantumGate>(op)) {
    //         wires = wireTable.getQubitWires(qGate.getNonCtrlQubitOperands());
    //     }
    //     else if (auto stateOp = dyn_cast<SetStateOp>(op)) {
    //         wires = wireTable.getQubitWires(stateOp.getQubits());
    //     }
    //     else {
    //         // op->emitError("Not supported");
    //         return;
    //     }

    //     Gate gate = (isa<MultiRZOp>(op) || isa<PCPhaseOp>(op)) ? Gate::I : Gate::U;
    //     progAbst.applyGate(gate, false, wires);
    // }

    // --- Control Flow Handlers ---

    void handleIfOp(mlir::scf::IfOp ifOp, ProgramAbstraction &parentAbst, WireTable &wireTable) 
    {
        ProgramAbstraction thenAbst(parentAbst.numQubits());
        ProgramAbstraction elseAbst(parentAbst.numQubits());

        analyzeBlock(&ifOp.getThenRegion().front(), thenAbst, wireTable);
        
        if (!ifOp.getElseRegion().empty()) {
            analyzeBlock(&ifOp.getElseRegion().front(), elseAbst, wireTable);
        }

        // llvm::outs() << "thenAbst: " << thenAbst << "\n";
        // llvm::outs() << "elseAbst: " << elseAbst << "\n";
        // llvm::outs() << "parentAbst: " << parentAbst << "\n";
        RegionSummary branchSummary(RegionType::Conditional, thenAbst, &elseAbst);
        // llvm::outs() << "branchSummary: " << branchSummary << "\n";
        parentAbst.applySummary(std::move(branchSummary));
        // llvm::outs() << "parentAbst after applySummary: " << parentAbst << "\n";
    }

    void handleForOp(mlir::scf::ForOp forOp, ProgramAbstraction &parentAbst, WireTable &wireTable) 
    {
        ProgramAbstraction loopAbst(parentAbst.numQubits());

        analyzeBlock(&forOp.getBodyRegion().front(), loopAbst, wireTable);

        RegionSummary loopSummary(RegionType::Loop, loopAbst);

        parentAbst.applySummary(std::move(loopSummary));
    }

    void handleCallOp(mlir::func::CallOp callOp, ProgramAbstraction &parentAbst, WireTable &wireTable) 
    {
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

    // --- MLIR Analysis ---

    void analyzeOperation(mlir::Operation *op, ProgramAbstraction &currentAbst, WireTable &wireTable) 
    {
        if (!isTrackedDialectOp(op)) {
            op->emitError("Operation is not a tracked dialect operation: " + op->getName().getStringRef());
            return;
        }
        llvm::TypeSwitch<mlir::Operation *, void>(op)
            .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp ifOp) {
                llvm::outs() << "IfOp:  " << "\n";
                handleIfOp(ifOp, currentAbst, wireTable);
            })
            .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp forOp) {
                llvm::outs() << "ForOp: " << "\n";
                handleForOp(forOp, currentAbst, wireTable);
            })
            .Case<mlir::func::CallOp>([&](mlir::func::CallOp callOp) {
                llvm::outs() << "CallOp:    " << callOp << "\n";
                handleCallOp(callOp, currentAbst, wireTable);
            })
            .Case<qref::CustomOp>([&](qref::CustomOp customOp) {
                llvm::outs() << "CustomOp:  " << customOp << "\n";
                handleCustomOp(customOp, currentAbst, gateID, wireTable);
            })
            .Case<qref::SetBasisStateOp>([&](qref::SetBasisStateOp basisOp) {
                llvm::outs() << "SetBasisStateOp:   " << basisOp << "\n";
                initQubitsState(basisOp, currentAbst, wireTable);
            })
            .Case<qref::AllocOp>([&](qref::AllocOp allocOp) {
                llvm::outs() << "AllocOp:   " << allocOp << "\n";
            })
            .Case<qref::AllocQubitOp>([&](qref::AllocQubitOp allocQbOp) {
                llvm::outs() << "AllocQubitOp:  " << allocQbOp << "\n";
            })
            .Case<qref::DeallocOp>([&](qref::DeallocOp deallocOp) {
                llvm::outs() << "DeallocOp:   " << deallocOp << "\n";
                // deallocateRegister(deallocOp.getQreg());
            })
            .Case<qref::DeallocQubitOp>([&](qref::DeallocQubitOp deallocQbOp) {
                llvm::outs() << "DeallocQubitOp:   " << deallocQbOp << "\n";
                // deallocateQubit(deallocQbOp.getQubit());
            })
            .Case<qref::GetOp>([&](qref::GetOp getOp) {
                llvm::outs() << "GetOp: " << getOp << "\n";
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

    void analyzeBlock(mlir::Block *block, ProgramAbstraction &currentAbst, WireTable &wireTable) 
    {
        for (mlir::Operation &op : *block) {
            analyzeOperation(&op, currentAbst, wireTable);
        }
        // llvm::outs() << "\nblock: \n" << currentAbst << "\n";
    }

  public:
    void analyzeFuncOp(mlir::func::FuncOp funcOp)
    {
        mlir::StringRef funcName = funcOp.getName();
        if (procedureSummaries.contains(funcName)) return;

        WireTable wireTable;
        wireTable.populate(funcOp);

        ProgramAbstraction funcAbst(wireTable.size());
        analyzeBlock(&funcOp.getBody().front(), funcAbst, wireTable);

        // TODO: what if we had multiple qnodes? better way of specifying the main function?
        if (funcOp->hasAttrOfType<mlir::UnitAttr>(mainFuncIndicator)) {
            mainProgramAbst = funcAbst; // is it a deep copy?
        }
        procedureSummaries[funcName] = RegionSummary(RegionType::Procedure, funcAbst);

        // Functions typically have a single block in their body region
        // if (!funcOp.getBody().empty()) {
        // }
    }

    void dumpSummaries()
    {
        llvm::outs() << "\nAll Summaries:\n";
        for (auto &[funcName, summary] : procedureSummaries) {
            llvm::outs() << funcName << "\n";
            llvm::outs() << summary << "\n";
        }
        llvm::outs() << "\nMain Program Abstraction:\n" << mainProgramAbstraction() << "\n\n";
    }
};
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
                analyzer.analyzeFuncOp(func);
            }
        }

        analyzer.dumpSummaries();

        phaseMerge(analyzer.mainProgramAbstraction(), analyzer.phaseOperations());

        reportStats();
    }
};

} // namespace qref
} // namespace catalyst

// each module will have a single qnode, but a program can have multiple modules.
