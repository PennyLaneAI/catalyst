#define DEBUG_TYPE "phase-folding"

#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // For arith::ConstantOp
#include "llvm/ADT/StringRef.h"
// #include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h" // For llvm::concat<>

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

#include "SymbolicAnalysis/SymbolicCircuit.h"
#include "SymbolicAnalysis/Gate.h"

#include <vector>
#include <cassert>
#include <cmath> // for std::fmod, std::abs

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_PHASEFOLDINGPASS
#define GEN_PASS_DEF_PHASEFOLDINGPASS
#include "Quantum/Transforms/Passes.h.inc"

struct PhaseFoldingPass : impl::PhaseFoldingPassBase<PhaseFoldingPass> {
    using PhaseFoldingPassBase::PhaseFoldingPassBase;
    
    llvm::DenseMap<mlir::Value, size_t> ssaToWireMap;   // handle multiple registers
    std::vector<quantum::CustomOp> phaseOps;

    size_t getIndexFromExtractOp(mlir::Value& value) {
        if (auto extractOp = llvm::cast<quantum::ExtractOp>(value.getDefiningOp())) {
            auto staticIdx = extractOp.getIdxAttr();
            if (staticIdx.has_value()) {
                return static_cast<size_t>(staticIdx.value_or(0));
            } else {
                // auto dynamicIdx = extractOp.getIdx();
                llvm::errs() << "Error: Dynamic qubit extraction indices are not supported.\n";
                assert(false);
            }
        } else {
            llvm::errs() << "Error: Found an untracked qubit not originating from a quantum.extract.\n";
            assert(false);
        }
    }

    llvm::SmallVector<size_t, 4> getQubitIndices(llvm::ArrayRef<mlir::Value> ins, const llvm::ArrayRef<mlir::OpResult>& outs) {     
        assert(ins.size() == outs.size());
        size_t n = ins.size();

        llvm::SmallVector<size_t, 4> indices;
        indices.reserve(n);

        for (size_t i = 0; i < n; i++) {
            mlir::Value inValue = ins[i];
            mlir::OpResult outValue = outs[i];
            
            size_t index;
            auto it = ssaToWireMap.find(inValue);
            if (it != ssaToWireMap.end()) { // inVal is the result of a prev. op.
                index = it->second;
                ssaToWireMap.erase(it);
            } else {                        // first gate on the wire
                index = getIndexFromExtractOp(inValue);
            }

            indices.push_back(index);
            ssaToWireMap[outValue] = index;
        }
        return indices;
    }

    llvm::SmallVector<size_t, 4> convertIndicesBase(llvm::ArrayRef<size_t> indices) {
        llvm::SmallVector<size_t, 4> oneBasedIndices;
        oneBasedIndices.reserve(indices.size());
        
        for (size_t idx : indices) {
            oneBasedIndices.push_back(idx + 1);
        }
        return oneBasedIndices;
    }


    Gate gateFromName(llvm::StringRef gateName) {
        for (size_t i = 0; i < PRIMITIV_GATES_COUNT; i++) {
            if ((GATE_NAME[i] == gateName)) {
                return static_cast<Gate>(i);
            }
        }
        return Gate::U;
    }

    Gate extractCliffTGate(quantum::CustomOp& op) {
        Gate gate = gateFromName(op.getGateName());
        if (!op.getInCtrlQubits().empty() || !op.getInCtrlValues().empty()) {
            if (isPhaseGate(gate)) {
                return Gate::I; // C-Rz gates don't alter state space, but alter phase space non-linearly, which I'm not going to track for now, but should be trackable using xy = x + y - (x \oplus y).
            }
            // if (gate == Gate::X && op.getInCtrlQubits().size() == 1 && op.getInCtrlValues().empty()) {
            //     return Gate::CNOT;
            // }    should pass getInCtrlQubit as qubitIn. will do it later.
            return Gate::U;                                       
        }
        return gate;
    }


    double getPhase(quantum::CustomOp& op) {
        double c = (op.getAdjointFlag() ? -1 : 1);
        double angle = rotAngle(gateFromName(op.getGateName()));

        return ((angle != UNKNOWN_ANGLE ? angle : extractRZAngle(op.getParams())) * c);
    }

    double extractRZAngle(mlir::ValueRange params) {
        if (params.empty()) { return 0.0; }
        
        mlir::FloatAttr floatAttr;
        if (mlir::matchPattern(params.front(), mlir::m_Constant(&floatAttr))) {            
            return floatAttr.getValueAsDouble();
        } else {    // dynamic param
            return 0.0;
        }
    }

    double sumAngles(const PhaseBucket& contributors) {
        double sum = 0.0;
        for (GateID id : contributors.zeroAffineRZs) {
            sum += getPhase(phaseOps[id]);
        }
        for (GateID id : contributors.oneAffineRZs) {
            sum -= getPhase(phaseOps[id]);
        }
        return sum;
    }

    double normalizeAngle(double angle) {
        double quot = std::floor(angle / (2 * PI));
        double res = angle - ((2 * PI) * quot);
        if (res >= (2 * PI)) {
            return 0.0;
        } 
        if (res < 0.0) {
            return res + (2 * PI);
        }
        return res;

        // return std::fmod(angle, 2 * PI);
    }


    void removePhaseOp(quantum::CustomOp& op) {
        Gate gate = gateFromName(op.getGateName());
        insertedGateCount[static_cast<size_t>(gate)]--;

        if (gate == Gate::Y) {
            op.setGateName(GATE_NAME[static_cast<size_t>(Gate::X)]);
            insertedGateCount[static_cast<size_t>(Gate::X)]++;
        } else {
            killOp(op);
        }       
    }

    void updateTargetOp(quantum::CustomOp& targetOp, double sumAngle) {
        quantum::CustomOp lastOp = targetOp;
        mlir::OpBuilder builder(targetOp.getContext());

        double normAngle = normalizeAngle(sumAngle);
        
        // consider - angles?
        if (normAngle >= rotAngle(Gate::Z)) {
            quantum::CustomOp zOp = injectOpAfter(builder, lastOp, Gate::Z, false, 0.0);
            lastOp = zOp;
            normAngle -= rotAngle(Gate::Z);
        }
        if (normAngle >= rotAngle(Gate::S)) {
            quantum::CustomOp sOp = injectOpAfter(builder, lastOp, Gate::S, false, 0.0);
            lastOp = sOp;
            normAngle -= rotAngle(Gate::S);
        }
        if (normAngle == rotAngle(Gate::T)) {
            quantum::CustomOp sOp = injectOpAfter(builder, lastOp, Gate::T, false, 0.0);
            lastOp = sOp;
            normAngle -= rotAngle(Gate::T);
        }
        if (normAngle != 0.0) {
            quantum::CustomOp rzOp = injectOpAfter(builder, lastOp, Gate::RZ, false, normAngle);
            lastOp = rzOp;
        }

        removePhaseOp(targetOp);
    }
    // Put discretization option non and off

    // Test with multiple insertions
    quantum::CustomOp injectOpAfter(mlir::OpBuilder& builder, quantum::CustomOp& preOp, Gate gate, bool isAdjoint, double angle) {
        builder.setInsertionPointAfter(preOp.getOperation());
        mlir::Value qIn = preOp.getQubitResults()[0];
        mlir::Value angleVal;
        if (gate == Gate::RZ) {
            angleVal = arith::ConstantOp::create(builder, preOp.getLoc(), builder.getF64FloatAttr(angle));
        }    
        
        quantum::CustomOp newOp = quantum::CustomOp::create(
                            builder,
                            preOp.getLoc(),
            /*gate_name=*/  GATE_NAME[static_cast<size_t>(gate)],
            /*in_qubits=*/  mlir::ValueRange({qIn}),
            /*params=*/     (gate == Gate::RZ) ? mlir::ValueRange({angleVal}) : mlir::ValueRange({}),
            /*adjoint=*/    isAdjoint
        );

        qIn.replaceUsesWithIf(newOp.getQubitResults()[0], 
            [&](mlir::OpOperand& use) {
                return use.getOwner() != newOp.getOperation();
            }
        );

        insertedGateCount[static_cast<size_t>(gate)]++;
        return newOp;
    }

    void killOp(quantum::CustomOp& op) {
        assert(op.getInCtrlQubits().empty() && op.getInCtrlValues().empty());

        mlir::ValueRange qIns = op.getInQubits(); // get in_qubit
        op.replaceAllUsesWith(qIns);
        op.erase();
    }

    
    void phaseAnalysis(SymbolicCircuit& symCirc) {
        GateID gateID = -1;
        getOperation()->walk([&](Operation *op) {
            // if (isa<AllocOp>(op)){

            // }
            // if (auto allocOp = dyn_cast<AllocOp>(op)){

            // }
            if (CustomOp customOp = dyn_cast<CustomOp>(op)) {
                // getting affected wires
                llvm::SmallVector<size_t, 4> qubitIndices = convertIndicesBase(getQubitIndices(customOp.getQubitOperands(), customOp.getQubitResults())); // getQubitIndices(customOp.getInQubits(), customOp.getOutQubits())

                // tracking phase gates
                Gate gate = extractCliffTGate(customOp);
                if (isPhaseGate(gate)) {
                    phaseOps.push_back(customOp);
                    gateID++;
                }
                initialGateCount[static_cast<size_t>(gate)]++;

                // updating symbolic circuit
                symCirc.applyGate(gate, customOp.getAdjointFlag(), qubitIndices, gateID);
            }
        });
    }

    void phaseMerge(SymbolicCircuit& symCirc) {
        for (auto& [parity, contributors] : symCirc.phasePoly.terms) {
            if (contributors.gateCount() > 1) {
                double angleSum = sumAngles(contributors);
                GateID targetOpID = contributors.mergeTarget(); 

                if (!contributors.isMergeTargetAffineZero()) {
                    angleSum *= -1.0;
                }
                updateTargetOp(phaseOps[targetOpID], angleSum);                

                for (GateID id : llvm::concat<GateID>(contributors.zeroAffineRZs, contributors.oneAffineRZs)) {
                    if (id != targetOpID) {
                        removePhaseOp(phaseOps[id]);
                    }
                }
            }
        }
    }

    void reportStats() {
        for (size_t i = 0; i < PRIMITIV_GATES_COUNT; i++) {
            if (insertedGateCount[i] != 0) {
                llvm::outs() << GATE_NAME[i] << ": initial-> " << initialGateCount[i] 
                    << ",  final-> " << (initialGateCount[i] + insertedGateCount[i]) 
                    << ". difference-> " << insertedGateCount[i] << "\n";
            }
        }
    }

    void runOnOperation() override {
        llvm::outs() << "Hello phase-folding world!\n";

        SymbolicCircuit symCirc = SymbolicCircuit();
        llvm::outs() << symCirc << "\n";

        phaseAnalysis(symCirc);
        llvm::outs() << symCirc << "\n";

        phaseMerge(symCirc);
        reportStats();        
    }
};


} // namespace quantum
} // namespace catalyst


// Currently ignoring any blocks or dynamic allocations, only capturing pure quantum circuits.

/*
TODO: 
test S, Z, and Y

initialize qubitNum when got a register allocation op

testttt  (make sure same parities with different affine consts work fine)

ancila initialization (0/1)

is the current 1-based indexing of qubits in SymbolicCircuit good?
is the current merge phase efficient?
*/