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
// #include "llvm/ADT/STLExtras.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

#include "SymbolicAnalysis/SymbolicCircuit.h"

#include <vector>
#include <cassert>

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

    static constexpr double PI = llvm::numbers::pi;
    static constexpr size_t NATIVE_GATE_NUM = 12;
    static constexpr StringLiteral GATE_NAMES[] = { "Identity", "Hadamard", "PauliX", "PauliY", "PauliZ", "S", "T", "RZ", "CNOT", "SWAP", "_", "GlobalPhase" };
    
    llvm::DenseMap<mlir::Value, size_t> SSAToWireMap;
    std::vector<quantum::CustomOp> RotationOps;

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
            auto it = SSAToWireMap.find(inValue);
            if (it != SSAToWireMap.end()) { // inVal is the result of a prev. op.
                index = it->second;
                SSAToWireMap.erase(it);
            } else {                        // first gate on the wire
                index = getIndexFromExtractOp(inValue);
            }

            indices.push_back(index);
            SSAToWireMap[outValue] = index;
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

    SymbolicCircuit::Gate gateFromOp(llvm::StringRef gateName) {
        for (size_t i = 0; i < NATIVE_GATE_NUM; i++) {
            if (GATE_NAMES[i] == gateName) {
                return static_cast<SymbolicCircuit::Gate>(i);
            }
        }
        return SymbolicCircuit::U;
    }

    double getRotationAngle(quantum::CustomOp& op) {
        double c = (op.getAdjointFlag() ? -1 : 1);
        switch (gateFromOp(op.getGateName())) {
        case SymbolicCircuit::Z:    return PI * c;
        case SymbolicCircuit::S:    return (PI / 2.0) * c;
        case SymbolicCircuit::T:    return (PI / 4.0) * c;
        case SymbolicCircuit::RZ:   return getRZAngle(op.getParams()) * c;
        default:    return 0.0;
        }
    }

    double getRZAngle(mlir::ValueRange params) {
        if (params.empty()) { return 0.0; }
        
        mlir::FloatAttr floatAttr;
        if (mlir::matchPattern(params.front(), mlir::m_Constant(&floatAttr))) {            
            return floatAttr.getValueAsDouble();
        } else {    // dynamic param
            return 0.0;
        }
    }

    double sumAngles(const Term& contributors) {
        double sum = 0.0;
        for (GateID id : contributors.gateRefPol_0) {
            sum += getRotationAngle(RotationOps[id]);
        }
        for (GateID id : contributors.gateRefPol_1) {
            sum -= getRotationAngle(RotationOps[id]);
        }
        return sum; //  % (2 * PI);
    }

    void updateRemainGate(GateID remainGateID, double angle) {
        quantum::CustomOp& remainGate = RotationOps[remainGateID];

        mlir::OpBuilder builder(remainGate);
        mlir::Value angleVal = arith::ConstantOp::create(builder, remainGate.getLoc(), builder.getF64FloatAttr(angle));
        mlir::Value qIn = remainGate.getQubitOperands()[0];
        mlir::Value qOut = remainGate.getQubitResults()[0];

        auto resultGate = quantum::CustomOp::create(
            builder, remainGate.getLoc(),
            /*gate_name=*/GATE_NAMES[SymbolicCircuit::RZ],
            /*in_qubits=*/mlir::ValueRange({qIn}),
            /*params=*/mlir::ValueRange({angleVal}),
            /*adjoint=*/false
            );

        qOut.replaceAllUsesWith(resultGate.getQubitResults()[0]);
        remainGate.erase();
    }

    void removeGate(GateID id) {
        quantum::CustomOp& op = RotationOps[id];

        mlir::Value qIn = op.getQubitOperands()[0];
        mlir::Value qOut = op.getQubitResults()[0];

        qOut.replaceAllUsesWith(qIn);
        op.erase();
    }

    void runOnOperation() override {
        llvm::outs() << "Hello phase-folding world!\n";

        SymbolicCircuit symCirc = SymbolicCircuit();
        llvm::outs() << symCirc << "\n";

        // Phase Analysis
        GateID gateID = -1;
        getOperation()->walk([&](quantum::CustomOp op) {
            // getting affected wires
            llvm::SmallVector<size_t, 4> qubitIndices = convertIndicesBase(getQubitIndices(op.getQubitOperands(), op.getQubitResults()));

            // tracking Rz gates
            SymbolicCircuit::Gate gate = gateFromOp(op.getGateName());
            if (SymbolicCircuit::isRZ(gate)) {
                RotationOps.push_back(op);
                gateID++;
            }

            // updating symbolic circuit
            symCirc.applyGate(gate, op.getAdjointFlag(), qubitIndices, gateID);
        });

        llvm::outs() << symCirc << "\n";

        // Phase Merge
        for (const auto& [parity, contributors] : symCirc.phasePoly.terms) {
            if (contributors.gateNum() > 1) {
                double angleSum = sumAngles(contributors);

                GateID remainGateID = contributors.getHead(); 

                // update the remainGate
                if (!contributors.isHead0()) {
                    angleSum *= -1.0;
                }
                updateRemainGate(remainGateID, angleSum);                

                // erase other gates
                for (GateID id : contributors.gateRefPol_0) {
                    if (id != remainGateID) {
                        removeGate(id);
                    }
                }
                for (GateID id : contributors.gateRefPol_1) {
                    if (id != remainGateID) {
                        removeGate(id);
                    }
                } // TODO: merge 2 fors
            }
        }
        
    }
};


} // namespace quantum
} // namespace catalyst


// Currently ignoring any blocks or dynamic allocations, only capturing pure quantum circuits.
// first, I ignore Z and S gates. then they can be merged too!

// TODO: make angles % (2*PI)