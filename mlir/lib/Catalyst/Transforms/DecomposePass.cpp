#define DEBUG_TYPE "myhelloworld"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_DECOMPOSEPASS
#define GEN_PASS_DECL_DECOMPOSEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DecomposePass : public impl::DecomposePassBase<DecomposePass> {
    using impl::DecomposePassBase<DecomposePass>::DecomposePassBase;

    void runOnOperation() override { llvm::errs() << "Decompose Pass called successfully\n"; }
};

std::unique_ptr<Pass> createDecomposePass() { return std::make_unique<DecomposePass>(); }

} // namespace catalyst