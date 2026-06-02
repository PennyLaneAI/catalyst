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
    using Gate = SymbolicCircuit::Gate;

    llvm::DenseMap<mlir::Value, size_t> ssaToQubitMap;

    // TODO: make ins and outs of the same parent type if possible.
    std::vector<size_t> getQubitIndices(const std::vector<mlir::Value>& ins, const std::vector<mlir::OpResult>& outs) {     
        assert(ins.size() == outs.size());
        size_t n = ins.size();

        std::vector<size_t> indices;
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

    Gate extractGateFromOp(quantum::CustomOp op) {
        Gate gate = llvm::StringSwitch<Gate>(op.getGateName())
                    .Case("CNOT", SymbolicCircuit::CNOT)
                    .Case("PauliX", SymbolicCircuit::X)
                    .Case("PauliY", SymbolicCircuit::Y)
                    .Case("PauliZ", SymbolicCircuit::Z)
                    .Case("T", SymbolicCircuit::T)   // T^\dag?
                    .Case("S", SymbolicCircuit::S)   // S^\dag?
                    .Case("RZ", SymbolicCircuit::RZ)
                    .Case("SWAP", SymbolicCircuit::SWAP)
                    .Case("Identity", SymbolicCircuit::I)
                    .Case("Hadamard", SymbolicCircuit::H)
                    .Case("GlobalPhase", SymbolicCircuit::GP)
                    .Default(SymbolicCircuit::U);
        if (gate == SymbolicCircuit::Y && op.getAdjointFlag()) {
            gate = SymbolicCircuit::Y_dag;
        }
        return gate;
    }
    void runOnOperation() override {
        llvm::outs() << "Hello phase-folding world!\n";

        SymbolicCircuit circ = SymbolicCircuit();
        llvm::outs() << circ << "\n";

        int l = 0;
        getOperation()->walk([&](quantum::CustomOp op) {
            std::vector<mlir::Value> ins = op.getQubitOperands();
            std::vector<mlir::OpResult> outs = op.getQubitResults();
            std::vector<size_t> qubitIndices = getQubitIndices(ins, outs);

            Gate gate = extractGateFromOp(op);

            circ.applyGate(gate, &qubitIndices, l++);
            // circ.applyGate(gate, qubitIndices, op.getLoc().getAsOpaquePointer());
            // maybe passing op itself instead of loc?


            llvm::outs() << "Location:\n";
            llvm::outs() << "  " << op.getLoc() << "\n";

            // llvm::outs() << "Operands:\n";
            // for (auto operand : op.getOperands()) {
            //     llvm::outs() << "  " << operand << "\n";
            // }

            // op->dump();
            // op.getOperands();
            llvm::outs() << "-------------------------\n";
        });

        llvm::outs() << circ << "\n";
    }
};


} // namespace quantum
} // namespace catalyst


// about initializing map
// an ExtractOp can be either dynamic or static, which has either getIdx() (of type Value) or getIdxAttr() (of type uint64_t) respectively. 
// check Paul's PR on reference-semantic to get an idea of how handling dynamic case.

// Currently ignoring any blocks or dynamic allocations, only capturing pure quantum circuits.

// for merging, should probably have another walk.