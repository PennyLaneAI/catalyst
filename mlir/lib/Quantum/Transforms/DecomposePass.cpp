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
    using mlir::OpRewritePattern<quantum::CustomOp>::OpRewritePattern;

    // The above boilerplate instructs the pattern to be applied
    // to all operations of type `CustomOp` in the input mlir

    // quantum::CustomOp createSimpleOneBitGate(StringRef gateName, const Value &inQubit,
    //                                          const Value &outQubit, mlir::IRRewriter &builder,
    //                                          Location &loc,
    //                                          const quantum::CustomOp &insert_after_gate)
    // {
    //     OpBuilder::InsertionGuard insertionGuard(builder);
    //     builder.setInsertionPointAfter(insert_after_gate);
    //     quantum::CustomOp newGate =
    //         builder.create<quantum::CustomOp>(loc,
    //                                           /*out_qubits=*/mlir::TypeRange({outQubit.getType()}),
    //                                           /*out_ctrl_qubits=*/mlir::TypeRange(),
    //                                           /*params=*/mlir::ValueRange(),
    //                                           /*in_qubits=*/mlir::ValueRange({inQubit}),
    //                                           /*gate_name=*/gateName,
    //                                           /*adjoint=*/false,
    //                                           /*in_ctrl_qubits=*/mlir::ValueRange(),
    //                                           /*in_ctrl_values=*/mlir::ValueRange());

    //     return newGate;
    // }

    LogicalResult matchAndRewrite(quantum::CustomOp op, PatternRewriter &rewriter) const override
    {
        llvm::errs() << "Replacing a CustomOp at ";
        op->getLoc().print(llvm::errs());
        llvm::errs() << "\n";

        if (op->hasAttr("processed"))
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

        auto newOp = rewriter.create<quantum::CustomOp>(
            loc,
            TypeRange(outQubits.getTypes()), // result types
            TypeRange(outCtrlQubits.getTypes()), params, inQubits, gateName, adjoint, inCtrlQubits,
            inCtrlValues);

        newOp->setAttr("processed", rewriter.getUnitAttr());
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
        patterns.add<SleepOnceOnCustomOp>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDecomposePass() { return std::make_unique<DecomposePass>(); }

} // namespace catalyst