#define DEBUG_TYPE "remove-global-phases"

#include "llvm/Support/Debug.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Pass/Pass.h"

using namespace llvm;
using namespace mlir;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DECL_REMOVEGLOBALPHASESPASS
#define GEN_PASS_DEF_REMOVEGLOBALPHASESPASS
#include "Quantum/Transforms/Passes.h.inc"

struct RemoveGlobalPhasesPass : public impl::RemoveGlobalPhasesPassBase<RemoveGlobalPhasesPass>{
    using impl::RemoveGlobalPhasesPassBase<RemoveGlobalPhasesPass>::RemoveGlobalPhasesPassBase;

    void runOnOperation() final {
        llvm::errs() << "RemoveGlobalPhasesPass Executed\n";
    }
};
    
} // namespace quantum
} // namespace catalyst