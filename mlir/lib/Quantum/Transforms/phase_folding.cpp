#define DEBUG_TYPE "phase-folding"

#include "llvm/Support/Debug.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Math/IR/Math.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

#include "SymbolicAnalysis/SymbolicCircuit.h"
// #include "SymbolicAnalysis/Term.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // For arith::ConstantOp
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

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

    llvm::DenseMap<mlir::Value, size_t> ssaToQubitMap;
    std::vector<quantum::CustomOp*> idToOpMap;

    // TODO: make ins and outs of the same parent type if possible.
    llvm::SmallVector<size_t, 4> getQubitIndices(llvm::ArrayRef<mlir::Value> ins, const llvm::ArrayRef<mlir::OpResult>& outs) {     
        assert(ins.size() == outs.size());
        size_t n = ins.size();

        llvm::SmallVector<size_t, 4> indices;
        indices.reserve(n);

        for (size_t i = 0; i < n; i++) {
            mlir::Value inValue = ins[i];
            mlir::OpResult outValue = outs[i];
            size_t index;

            auto it = ssaToQubitMap.find(inValue);
            if (it != ssaToQubitMap.end()) {
                index = it->second;
                ssaToQubitMap.erase(it);
            } else if (auto extractOp = llvm::cast<quantum::ExtractOp>(inValue.getDefiningOp())) {
                auto staticIdx = extractOp.getIdxAttr();
                // auto dynamicIdx = extractOp.getIdx();

                if (staticIdx.has_value()) {
                    index = static_cast<size_t>(staticIdx.value_or(0));
                } else {
                    llvm::errs() << "Error: Dynamic qubit extraction indices are not supported.\n";
                    assert(false);
                }
            } else {
                llvm::errs() << "Error: Found an untracked qubit not originating from a quantum.extract.\n";
                assert(false);
            }
            indices.push_back(index);
            ssaToQubitMap[outValue] = index;
        }
        return indices;
    }

    SymbolicCircuit::Gate extractGateFromOp(quantum::CustomOp op) {
        return llvm::StringSwitch<SymbolicCircuit::Gate>(op.getGateName())
                    .Case("CNOT", SymbolicCircuit::CNOT)
                    .Case("PauliX", SymbolicCircuit::X)
                    .Case("PauliY", SymbolicCircuit::Y)
                    .Case("PauliZ", SymbolicCircuit::Z)
                    .Case("T", SymbolicCircuit::T)
                    .Case("S", SymbolicCircuit::S)
                    .Case("RZ", SymbolicCircuit::RZ)
                    .Case("SWAP", SymbolicCircuit::SWAP)
                    .Case("Identity", SymbolicCircuit::I)
                    .Case("Hadamard", SymbolicCircuit::H)
                    .Case("GlobalPhase", SymbolicCircuit::GP)
                    .Default(SymbolicCircuit::U);
    }

    void runOnOperation() override {
        llvm::outs() << "Hello phase-folding world!\n";

        SymbolicCircuit symCirc = SymbolicCircuit();
        llvm::outs() << symCirc << "\n";

        GateID gateID = -1;
        getOperation()->walk([&](quantum::CustomOp op) {
            // getting affected wires
            llvm::SmallVector<size_t, 4> qubitIndices = getQubitIndices(op.getQubitOperands(), op.getQubitResults());

            // tracking Rz gates
            SymbolicCircuit::Gate gate = extractGateFromOp(op);
            if (SymbolicCircuit::isRZ(gate)) {
                idToOpMap.push_back(&op);
                gateID++;

                llvm::outs() << gateID << ": \n";
                for (auto operand : op.getParams()) {
                    llvm::outs() << "  *" << operand << "\n";
                }
            }

            // updating symbolic circuit
            symCirc.applyGate(gate, op.getAdjointFlag(), qubitIndices, gateID);
            
            llvm::outs() << "-------------------------\n";
        });

        llvm::outs() << symCirc << "\n";

        // for (const auto& [parity, term] : symCirc.phasePoly.poly) {
        //     // auto angle = 0;

        //     if (term.gateNum() > 1) {
        //         for (GateID id : term.gateRefPol_0) {
        //             quantum::CustomOp* op = idToOpMap[id];
        //             llvm::outs() << id << ": \n";
        //             // for (auto operand : op->getParams()) {
        //             //     llvm::outs() << "  *" << operand << "\n";
        //             // }
        //         }
        //         for (GateID id : term.gateRefPol_1) {
        //             llvm::outs() << id << " ";
        //         }
        //         llvm::outs() << "should be merged.\n";
        //     }
        // }
    }
};


} // namespace quantum
} // namespace catalyst


// Currently ignoring any blocks or dynamic allocations, only capturing pure quantum circuits.

// for merging, should probably have another walk.
// first, I ignore Z and S gates. then they can be merged too!

// llvm::outs() << "Location:\n";
// llvm::outs() << "  " << op.getLoc() << "\n";
// llvm::outs() << "Operands:\n";
// for (auto operand : op.getOperands()) {
//     llvm::outs() << "  " << operand << "\n";
// }
// op->dump();