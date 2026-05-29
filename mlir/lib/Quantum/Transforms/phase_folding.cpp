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

    void runOnOperation() override {
        llvm::errs() << "Hello phase-folding world!\n";

        SymbolicCircuit circ = SymbolicCircuit(2);
        llvm::errs() << circ << "\n";

        int l = 1;

        circ.applyGateCNOT(1, 2);
        llvm::errs() << circ << "\n";

        circ.applyGateRZ(2, l++);
        llvm::errs() << circ << "\n";

    }
};


} // namespace quantum
} // namespace catalyst
