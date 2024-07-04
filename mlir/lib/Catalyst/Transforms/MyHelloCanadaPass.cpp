#define DEBUG_TYPE "myhelloCanada"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_MYHELLOCANADAPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct MyHelloCanadaPass : public impl::MyHelloCanadaPassBase<MyHelloCanadaPass> {
    using impl::MyHelloCanadaPassBase<MyHelloCanadaPass>::MyHelloCanadaPassBase;

    void runOnOperation() override { llvm::errs() << "Hello Canada!\n"; }
};

std::unique_ptr<Pass> createMyHelloCanadaPass() { return std::make_unique<MyHelloCanadaPass>(); }

} // namespace catalyst