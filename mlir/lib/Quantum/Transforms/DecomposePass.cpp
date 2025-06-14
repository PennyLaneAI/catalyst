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

struct DecomposeCustomOp : public OpRewritePattern<quantum::CustomOp> {
    using mlir::OpRewritePattern<quantum::CustomOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(quantum::CustomOp op, PatternRewriter &rewriter) const override
    {
        llvm::outs() << "Replacing a CustomOp at ";
        op->getLoc().print(llvm::outs());
        llvm::outs() << "\n";

        if (op->hasAttr("already decomposed"))
            return failure();

        auto loc = op.getLoc();

        ValueRange outQubits = op.getOutQubits();
        ValueRange outCtrlQubits = op.getOutCtrlQubits();
        ValueRange params = op.getParams();
        ValueRange inQubits = op.getInQubits();
        StringRef gateName = op.getGateName();
        bool adjoint = op.getAdjoint();
        ValueRange inCtrlQubits = op.getInCtrlQubits();
        ValueRange inCtrlValues = op.getInCtrlValues();

        if (gateName == rewriter.getStringAttr("RX")) {
            gateName = rewriter.getStringAttr("RY");
        }

        auto newOp = rewriter.create<quantum::CustomOp>(
            loc, TypeRange(outQubits.getTypes()), TypeRange(outCtrlQubits.getTypes()), params,
            inQubits, gateName, adjoint, inCtrlQubits, inCtrlValues);

        newOp->setAttr("already decomposed", rewriter.getUnitAttr());
        rewriter.replaceOp(op, newOp->getResults());
        return success();
    }
};

struct DecomposePass : public impl::DecomposePassBase<DecomposePass> {
    using impl::DecomposePassBase<DecomposePass>::DecomposePassBase;

    void runOnOperation() final
    {
        LLVM_DEBUG(llvm::dbgs() << "Decompose Pass running\n");

        RewritePatternSet patterns(&getContext());
        patterns.add<DecomposeCustomOp>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDecomposePass() { return std::make_unique<DecomposePass>(); }

} // namespace catalyst