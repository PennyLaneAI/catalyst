#define DEBUG_TYPE "myhelloworld"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_MYHELLOWORLDPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct MyHelloWorldPass : public impl::MyHelloWorldPassBase<MyHelloWorldPass> {
    using impl::MyHelloWorldPassBase<MyHelloWorldPass>::MyHelloWorldPassBase;

    void runOnOperation() override { llvm::errs() << "Hello world!\n"; }
};

std::unique_ptr<Pass> createMyHelloWorldPass() { return std::make_unique<MyHelloWorldPass>(); }

} // namespace catalyst