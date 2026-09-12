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

#define DEBUG_TYPE "phase_folding"

#include <cassert>
#include <cmath> // std::abs
#include <memory>
#include <string>
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
using namespace catalyst::phase_folding;

namespace {

struct GateStatsHolder {
    int initialGateCount[PRIMITIV_GATES_COUNT] = {0};
    int insertedGateCount[PRIMITIV_GATES_COUNT] = {0};

    void updateInitialCount(Gate gate) { initialGateCount[static_cast<size_t>(gate)]++; }

    void updateModifications(Gate gate, int incr) { insertedGateCount[static_cast<size_t>(gate)] += incr; }

    void reportStats(llvm::raw_ostream &os = llvm::outs()) const
    {
        os << "Stats:\n";
        for (size_t i = 0; i < PRIMITIV_GATES_COUNT; i++) {
            if (initialGateCount[i] != 0 || insertedGateCount[i] != 0) {
                os << GATE_NAME[i] << ": initial-> " << initialGateCount[i]
                             << ",  final-> " << (initialGateCount[i] + insertedGateCount[i])
                             << ". difference-> " << insertedGateCount[i] << "\n";
            }
        }
        os << "\n";
    }
};

struct AbstractionTracer {
    struct ContextScope {
        AbstractionTracer *tracer;
        llvm::StringRef label;

        ContextScope(AbstractionTracer *tracer, llvm::StringRef label)
            : tracer(tracer), label(label)
        {
            if (tracer) {
                tracer->enter(label);
            }
        }

        ~ContextScope()
        {
            if (tracer) {
                tracer->exit(label);
            }
        }

        ContextScope(const ContextScope &) = delete;
        ContextScope &operator=(const ContextScope &) = delete;
    };

    struct StepContext {
        unsigned subStep = 0;
    };

    std::unique_ptr<llvm::raw_fd_ostream> os;
    unsigned rootStep = 0;
    llvm::SmallVector<unsigned, 8> stepPath;
    llvm::SmallVector<StepContext, 8> stepContexts;

    explicit AbstractionTracer(llvm::StringRef path)
    {
        std::error_code ec;
        auto stream = std::make_unique<llvm::raw_fd_ostream>(path, ec);
        if (!ec) {
            os = std::move(stream);
        }
    }

    explicit operator bool() const { return os != nullptr; }

    [[nodiscard]] ContextScope context(llvm::StringRef label)
    {
        return ContextScope(this, label);
    }

    void logAfterGate(qref::CustomOp &op, llvm::ArrayRef<size_t> wires,
                      const ProgramAbstraction &abst)
    {
        logStep(
            [&](llvm::raw_ostream &out) {
                out << "gate " << op.getGateName() << " on ";
                if (wires.size() == 1) {
                    out << "wire " << wires.front();
                } else {
                    out << "wires ";
                    for (size_t i = 0; i < wires.size(); ++i) {
                        if (i > 0) {
                            out << ", ";
                        }
                        out << wires[i];
                    }
                }
            },
            [&] {
                op.getLoc().print(*os);
                *os << '\n';
                writeValue(abst);
            });
    }

    void logBranchExit(llvm::StringRef branch, const ProgramAbstraction &abst)
    {
        logStep([&](llvm::raw_ostream &out) { out << branch << " branch-exit"; },
                [&] { writeValue(abst); });
    }

    void logRegionSummary(llvm::StringRef region, const RegionSummary &summary)
    {
        logStep([&](llvm::raw_ostream &out) { out << region << " region-summary"; },
                [&] { writeValue(summary); });
    }

    void logAfterSummaryApplied(llvm::StringRef region, const ProgramAbstraction &parent)
    {
        logStep([&](llvm::raw_ostream &out) { out << region << " parent-after-merge"; },
                [&] { writeValue(parent); });
    }

  private:
    void enter(llvm::StringRef label)
    {
        if (!os) {
            return;
        }
        unsigned anchor = stepContexts.empty() ? nextRootStep() : nextSubStep();
        stepPath.push_back(anchor);
        stepContexts.emplace_back();

        *os << "\n--- step ";
        writeStepNumber(*os);
        *os << " [enter " << label << "] ---\n";
    }

    void exit(llvm::StringRef label)
    {
        if (!os || stepContexts.empty()) {
            return;
        }
        *os << "\n--- step ";
        writeStepNumber(*os, nextSubStep());
        *os << " [exit " << label << "] ---\n";

        stepContexts.pop_back();
        stepPath.pop_back();
    }

    unsigned nextRootStep() { return ++rootStep; }

    unsigned nextSubStep()
    {
        assert(!stepContexts.empty());
        return ++stepContexts.back().subStep;
    }

    void writeStepNumber(llvm::raw_ostream &out, unsigned subStep = 0) const
    {
        if (stepPath.empty()) {
            out << subStep;
            return;
        }
        for (size_t i = 0; i < stepPath.size(); ++i) {
            if (i > 0) {
                out << '.';
            }
            out << stepPath[i];
        }
        if (subStep > 0) {
            out << '.' << subStep;
        }
    }

    void logStep(llvm::StringRef tag, llvm::function_ref<void()> body)
    {
        if (!os) {
            return;
        }
        *os << "\n--- step ";
        if (stepContexts.empty()) {
            *os << nextRootStep();
        } else {
            writeStepNumber(*os, nextSubStep());
        }
        *os << " [" << tag << "] ---\n";
        body();
    }

    void logStep(llvm::function_ref<void(llvm::raw_ostream &)> tagWriter,
                 llvm::function_ref<void()> body)
    {
        if (!os) {
            return;
        }
        *os << "\n--- step ";
        if (stepContexts.empty()) {
            *os << nextRootStep();
        } else {
            writeStepNumber(*os, nextSubStep());
        }
        *os << " [";
        tagWriter(*os);
        *os << "] ---\n";
        body();
    }

    template <typename T>
    void writeValue(const T &value)
    {
        *os << value << '\n';
    }
};

struct PhaseFoldingPlan {
    ProgramAbstraction mainProgramAbst;
    std::vector<qref::CustomOp> phaseOps;
    GateStatsHolder stats;

    void writeReport(StringRef path) {
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec);
        if (ec) return;
        
        stats.reportStats(os);
        os << "\nMain Program Abstraction:\n" << mainProgramAbst << "\n\n";
    }
};

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

struct PhaseAnalyzer {
  public:
    PhaseAnalyzer(PhaseFoldingPlan &plan, AbstractionTracer *tracer = nullptr) 
        : plan(plan), tracer(tracer) {}

  private:
    PhaseFoldingPlan &plan;
    AbstractionTracer *tracer;

    GateID gateID = -1;

    llvm::StringMap<RegionSummary> procedureSummaries; // keyed by procedure name
    
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
            plan.phaseOps.push_back(customOp);
            gateID++;
        }
        plan.stats.updateInitialCount(gate);
    
        progAbst.applyGate(gate, customOp.getAdjointFlag(), wires, gateID);

        if (tracer) {
            tracer->logAfterGate(customOp, wires, progAbst);
        }
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

    void mergeRegionIntoParent(llvm::StringRef regionLabel, RegionSummary summary,
                               ProgramAbstraction &parentAbst)
    {
        if (tracer) {
            tracer->logRegionSummary(regionLabel, summary);
        }
        parentAbst.applySummary(std::move(summary));
        if (tracer) {
            tracer->logAfterSummaryApplied(regionLabel, parentAbst);
        }
    }

    void handleIfOp(mlir::scf::IfOp ifOp, ProgramAbstraction &parentAbst, WireTable &wireTable) 
    {
        AbstractionTracer::ContextScope traceScope(tracer, "scf.if");

        ProgramAbstraction thenAbst(parentAbst.numQubits());
        ProgramAbstraction elseAbst(parentAbst.numQubits());

        analyzeBlock(&ifOp.getThenRegion().front(), thenAbst, wireTable);
        if (tracer) {
            tracer->logBranchExit("then", thenAbst);
        }

        if (!ifOp.getElseRegion().empty()) {
            analyzeBlock(&ifOp.getElseRegion().front(), elseAbst, wireTable);
            if (tracer) {
                tracer->logBranchExit("else", elseAbst);
            }
        }

        // llvm::outs() << "thenAbst: " << thenAbst << "\n";
        // llvm::outs() << "elseAbst: " << elseAbst << "\n";
        // llvm::outs() << "parentAbst: " << parentAbst << "\n";
        RegionSummary branchSummary(RegionType::Conditional, thenAbst, &elseAbst);
        mergeRegionIntoParent("scf.if", std::move(branchSummary), parentAbst);
    }

    void handleForOp(mlir::scf::ForOp forOp, ProgramAbstraction &parentAbst, WireTable &wireTable) 
    {
        AbstractionTracer::ContextScope traceScope(tracer, "scf.for");

        ProgramAbstraction loopAbst(parentAbst.numQubits());
        analyzeBlock(&forOp.getBodyRegion().front(), loopAbst, wireTable);
        if (tracer) {
            tracer->logBranchExit("body", loopAbst);
        }

        RegionSummary loopSummary(RegionType::Loop, loopAbst);
        mergeRegionIntoParent("scf.for", std::move(loopSummary), parentAbst);
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
            // op->emitError("Operation is not a tracked dialect operation: " + op->getName().getStringRef());
            return;
        }
        llvm::TypeSwitch<mlir::Operation *, void>(op)
            .Case<mlir::scf::IfOp>([&](mlir::scf::IfOp ifOp) {
                // llvm::outs() << "IfOp:  " << "\n";
                handleIfOp(ifOp, currentAbst, wireTable);
            })
            .Case<mlir::scf::ForOp>([&](mlir::scf::ForOp forOp) {
                // llvm::outs() << "ForOp: " << "\n";
                handleForOp(forOp, currentAbst, wireTable);
            })
            .Case<mlir::func::CallOp>([&](mlir::func::CallOp callOp) {
                // llvm::outs() << "CallOp:    " << callOp << "\n";
                // handleCallOp(callOp, currentAbst, wireTable);
            })
            .Case<qref::CustomOp>([&](qref::CustomOp customOp) {
                // llvm::outs() << "CustomOp:  " << customOp << "\n";
                handleCustomOp(customOp, currentAbst, gateID, wireTable);
            })
            .Case<qref::SetBasisStateOp>([&](qref::SetBasisStateOp basisOp) {
                // llvm::outs() << "SetBasisStateOp:   " << basisOp << "\n";
                initQubitsState(basisOp, currentAbst, wireTable);
            })
            .Case<qref::AllocOp>([&](qref::AllocOp allocOp) {
                // llvm::outs() << "AllocOp:   " << allocOp << "\n";
            })
            .Case<qref::AllocQubitOp>([&](qref::AllocQubitOp allocQbOp) {
                // llvm::outs() << "AllocQubitOp:  " << allocQbOp << "\n";
            })
            .Case<qref::DeallocOp>([&](qref::DeallocOp deallocOp) {
                // llvm::outs() << "DeallocOp:   " << deallocOp << "\n";
                // deallocateRegister(deallocOp.getQreg());
            })
            .Case<qref::DeallocQubitOp>([&](qref::DeallocQubitOp deallocQbOp) {
                // llvm::outs() << "DeallocQubitOp:   " << deallocQbOp << "\n";
                // deallocateQubit(deallocQbOp.getQubit());
            })
            .Case<qref::GetOp>([&](qref::GetOp getOp) {
                // llvm::outs() << "GetOp: " << getOp << "\n";
            })
            .Case<qref::MeasureOp>([&](qref::MeasureOp measureOp) {
                // llvm::outs() << "MeasureOp:  " << measureOp << "\n";
            })
            .Case<qref::GlobalPhaseOp>([&](qref::GlobalPhaseOp gpOp) {
                // llvm::outs() << "GlobalPhaseOp: " << gpOp << "\n";
            })
            .Case<mlir::scf::YieldOp>([&](mlir::scf::YieldOp yieldOp) {
                // llvm::outs() << "YieldOp:  " << yieldOp << "\n";
                
            })
            .Default([&](mlir::Operation *unknownOp) {
                // Handle or ignore operations that don't affect phases (e.g., standard arithmetic)
                // llvm::outs() << "UnknownOp: " << *unknownOp << "\n";
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
        if (funcOp->hasAttrOfType<mlir::UnitAttr>("quantum.node")) {
            plan.mainProgramAbst = funcAbst; // is it a deep copy?
        }
        procedureSummaries[funcName] = RegionSummary(RegionType::Procedure, funcAbst);

        // Functions typically have a single block in their body region
        // if (!funcOp.getBody().empty()) {
        // }
    }

    void dumpSummaries(llvm::raw_ostream &os = llvm::outs())
    {
        os << "\nAll Summaries:\n";
        for (auto &[funcName, summary] : procedureSummaries) {
            os << funcName << "\n";
            os << summary << "\n";
        }
    }
};

struct PhaseFolder {
  public:
    PhaseFolder(PhaseFoldingPlan &plan) : plan(plan) {}

  private:
    PhaseFoldingPlan &plan;

    // --- Rotation Angle Computations ---

    double extractConstAngle(mlir::ValueRange params)
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

    double phaseAngle(qref::CustomOp &op)
    {
        double c = (op.getAdjointFlag() ? -1 : 1);
        double angle = rotAngle(gateWithName(op.getGateName()));

        return ((angle != UNKNOWN_ANGLE ? angle : extractConstAngle(op.getParams())) * c);
    }

    double netBundleAngle(const GateBundle &contributors)
    {
        double sum = 0.0;
        for (GateID id : contributors.zeroAffineGates) {
            sum += phaseAngle(plan.phaseOps[id]);
        }
        for (GateID id : contributors.oneAffineGates) {
            sum -= phaseAngle(plan.phaseOps[id]);
        }
        return sum;
    }

    double normalizeAngle(double angle)
    { // returns equivalent angle between [-PI, PI]
        return std::remainder(angle, 2 * PI);
    }

    // --- IR Modifications ---
    
    void eraseOp(qref::CustomOp &op)
    {
        assert(op.getCtrlQubits().empty() && op.getCtrlValues().empty()); // move to somewhere better
        plan.stats.updateModifications(gateWithName(op.getGateName()), -1);
        op.erase();
        op = {};
    }

    void replaceOpWith(qref::CustomOp &curOp, Gate newGate, bool isAdjoint, double mergedAngle)
    {
        plan.stats.updateModifications(gateWithName(curOp.getGateName()), -1);

        mlir::IRRewriter rewriter(curOp.getContext());
        rewriter.setInsertionPoint(curOp.getOperation());

        llvm::SmallVector<mlir::Value, 1> params;
        if (newGate == Gate::RZ) {
            params.push_back(arith::ConstantOp::create(
                rewriter, curOp.getLoc(), rewriter.getF64FloatAttr(mergedAngle)));
        }

        auto newOp = rewriter.replaceOpWithNewOp<qref::CustomOp>(
            curOp, 
            /*params=*/ params,
            /*qubits=*/ curOp.getQubits(), 
            /*gate_name=*/GATE_NAME[static_cast<size_t>(newGate)], 
            /*adjoint=*/ isAdjoint,
            curOp.getCtrlQubits(),
            curOp.getCtrlValues());
        curOp = newOp;

        plan.stats.updateModifications(newGate, +1);
    }

    void foldIntoTarget(qref::CustomOp &targetOp, double mergedAngle)
    {
        double normAngle = normalizeAngle(mergedAngle);
        bool isAdjoint = (normAngle < 0.0);
        normAngle = std::abs(normAngle);
        Gate gate = gateWithAngle(normAngle);

        if (gate == Gate::I) {
            eraseOp(targetOp);
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
            eraseOp(op);
        }
    }

    void removeGates(auto gateIDs, std::optional<GateID> skipID = std::nullopt) 
    {
        for (GateID id : gateIDs) {
            if (id != skipID) {
                removePhaseOp(plan.phaseOps[id]);
            }
        }
    }

    void foldBundle(GateBundle &bundle) 
    {
        if (bundle.gateCount() <= 1)    return;

        double mergedAngle = netBundleAngle(bundle);
        if (!bundle.isMergeTargetAffineZero()) {
            mergedAngle = -mergedAngle;
        }
                
        GateID targetOpID = bundle.getMergeTarget();
        foldIntoTarget(plan.phaseOps[targetOpID], mergedAngle);
        removeGates(bundle.getAllGates(), targetOpID);
    }

  public:
    void foldPhases()
    {
        for (auto &[parity, contributors] : plan.mainProgramAbst.phases.activeBundles) {
            if (parity.isTrivial()) {
                removeGates(contributors.zeroAffineGates);
                contributors.zeroAffineGates.clear();
            }
            foldBundle(contributors);
        }

        for (GateBundle &bundle : plan.mainProgramAbst.phases.orphanBundles) {
            foldBundle(bundle);
        }
    }
};



}   // namespace
    
namespace catalyst {
namespace qref {

#define GEN_PASS_DECL_PHASEFOLDINGPASS
#define GEN_PASS_DEF_PHASEFOLDINGPASS
#include "QRef/Transforms/Passes.h.inc"

struct PhaseFoldingPass : public impl::PhaseFoldingPassBase<PhaseFoldingPass> {
    using impl::PhaseFoldingPassBase<PhaseFoldingPass>::PhaseFoldingPassBase;

    void runOnOperation() override 
    {  
        llvm::outs() << "Hello phase-folding world!\n";

        mlir::ModuleOp rootModule = dyn_cast<mlir::ModuleOp>(getOperation());

        std::unique_ptr<AbstractionTracer> tracer;
        if (trace_abstraction) {
            std::string moduleName = rootModule.getName() ? rootModule.getName()->str() : "unnamed";
            tracer = std::make_unique<AbstractionTracer>(
                "phase_folding_trace_" + moduleName + ".txt");
        }

        PhaseFoldingPlan plan;
        PhaseAnalyzer analyzer(plan, tracer.get());
        for (auto func : rootModule.getOps<mlir::func::FuncOp>()) {
            // llvm::outs() << "FuncOp:\n" << func << "\n";
            if (!func.isExternal()) {   // only analyze functions with bodies
                analyzer.analyzeFuncOp(func);
            }
        }
    
        PhaseFolder folder(plan);
        folder.foldPhases();

        // analyzer.dumpSummaries();
        // plan.stats.reportStats();

        if (report_stats) {
            std::string moduleName = rootModule.getName() ? rootModule.getName()->str() : "unnamed";
            plan.writeReport("phase_folding_report_" + moduleName + ".txt");
        }
    }
};

} // namespace qref
} // namespace catalyst

// if seeing allocOp, change the state to |0>
// if seeing state preparation, change to x
    // can state preparation be called in the middle of the circuit?
