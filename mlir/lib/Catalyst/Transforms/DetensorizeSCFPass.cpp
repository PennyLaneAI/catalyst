#define DEBUG_TYPE "myhelloworld"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_DETENSORIZESCFPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetensorizeSCFPass : public impl::DetensorizeSCFPassBase<DetensorizeSCFPass> {
    using impl::DetensorizeSCFPassBase<DetensorizeSCFPass>::DetensorizeSCFPassBase;

    void runOnOperation() override
    {
        mlir::Operation *root = getOperation();
        IRRewriter rewriter(root->getContext());
        mlir::Operation *ifop{nullptr};
        mlir::Operation *yieldop{nullptr};
        Type type;
        auto isScalarTensor = [](Value v) {
            if (!::llvm::isa<TensorType>(v.getType())) {
                return false;
            }
            return dyn_cast<TensorType>(v.getType()).getRank() == 0;
        };
        root->walk([&](scf::IfOp op) {
            op->walk([&](scf::YieldOp yield) {
                llvm::errs() << "........................................\n";
                llvm::errs() << yield << '\n';
                for (mlir::Value res : yield.getResults()) {
                    llvm::errs() << res << '\n';
                    llvm::errs() << res.getType() << '\n';

                    if (isScalarTensor(res)) {
                        auto *defop = res.getDefiningOp();
                        llvm::errs() << *defop << '\n';
                        llvm::errs() << isa<tensor::FromElementsOp>(*defop) << '\n';
                        llvm::errs() << defop->getOperand(0) << '\n';
                        rewriter.replaceAllUsesWith(res, defop->getOperand(0));
                        rewriter.eraseOp(defop);
                        type = defop->getOperand(0).getType();
                        llvm::errs() << type << '\n';
                    }
                }
                llvm::errs() << "........................................\n";
                yieldop = yield;
            });
            llvm::errs() << type << '\n';

            llvm::errs() << "---------------------------------------\n";
            rewriter.modifyOpInPlace(op, [&]() {
                op.getResult(0).setType(type);
                llvm::errs() << op.getResult(0).getType() << '\n';
                op.getResult(1).setType(type);
                llvm::errs() << op.getResult(1).getType() << '\n';
            });
            llvm::errs() << "---------------------------------------\n";

            rewriter.setInsertionPointAfter(op);
            auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                op->getLoc(), RankedTensorType::get({}, op->getResult(1).getType()),
                op->getResult(1));
            rewriter.replaceUsesWithIf(
                op->getResult(1), from_elem_op->getResult(0),
                [&](OpOperand &operand) { return operand.getOwner() != from_elem_op; });
            ifop = op;
        });

        llvm::errs() << "|||||||||||||||||| Final IR ||||||||||||||||||\n";
        root->walk([&](mlir::Operation *op) { llvm::errs() << *op << '\n'; });
        llvm::errs() << "|||||||||||||||||| Final IR ||||||||||||||||||\n";
    }
};

std::unique_ptr<Pass> createDetensorizeSCFPass() { return std::make_unique<DetensorizeSCFPass>(); }

} // namespace catalyst