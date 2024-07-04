#define DEBUG_TYPE "myhelloToronto"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_MYHELLOTORONTOPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct MyHelloTorontoPass : public impl::MyHelloTorontoPassBase<MyHelloTorontoPass> {
    using impl::MyHelloTorontoPassBase<MyHelloTorontoPass>::MyHelloTorontoPassBase;

    void runOnOperation() override { llvm::errs() << "Hello Toronto!\n"; }
};

std::unique_ptr<Pass> createMyHelloTorontoPass() { return std::make_unique<MyHelloTorontoPass>(); }

} // namespace catalyst