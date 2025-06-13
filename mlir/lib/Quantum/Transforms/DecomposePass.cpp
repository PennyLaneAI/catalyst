#define DEBUG_TYPE "decompose"
#include <iostream>

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_DECOMPOSEPASS
#define GEN_PASS_DECL_DECOMPOSEPASS
#include "Quantum/Transforms/Passes.h.inc"

struct SleepOnceOnCustomOp : public OpRewritePattern<quantum::CustomOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(quantum::CustomOp op, PatternRewriter &rewriter) const override
    {

        llvm::errs() << "CustomOp matched. Sleeping for 10 seconds...\n";
        sleep(10);
        return failure();
    }
};

struct DecomposePass : public impl::DecomposePassBase<DecomposePass> {
    using impl::DecomposePassBase<DecomposePass>::DecomposePassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(llvm::dbgs() << "Decompose Pass running\n");

        RewritePatternSet patterns(&getContext());
        patterns.add<SleepOnceOnCustomOp>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDecomposePass() { return std::make_unique<DecomposePass>(); }

} // namespace catalyst