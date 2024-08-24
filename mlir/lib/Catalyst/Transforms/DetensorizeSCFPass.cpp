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
        auto isFromElementScalarTensor = [](Value v) {
            if (!::llvm::isa<TensorType>(v.getType())) {
                return false;
            }
            if (dyn_cast<TensorType>(v.getType()).getRank() != 0) {
                return false;
            }
            return isa<tensor::FromElementsOp>(*v.getDefiningOp());
        };
        // Find scf.if
        root->walk([&](scf::IfOp if_op) {
            bool first_yield = true;
            // Find scf.yield
            if_op->walk([&](scf::YieldOp yield_op) {
                // Loop over results
                std::size_t i_result = 0;
                for (mlir::Value result : yield_op.getResults()) {
                    if (isFromElementScalarTensor(result)) {
                        // Remove from_element op and substitute operand
                        mlir::Operation *from_element_op = result.getDefiningOp();
                        rewriter.replaceAllUsesWith(result, from_element_op->getOperand(0));
                        rewriter.modifyOpInPlace(if_op, [&]() {
                            if_op.getResult(i_result).setType(
                                from_element_op->getOperand(0).getType());
                        });
                        rewriter.eraseOp(from_element_op);
                        if (first_yield) {
                            // Create a from_element op after scf.if
                            rewriter.setInsertionPointAfter(if_op);
                            auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                                if_op->getLoc(),
                                RankedTensorType::get({}, if_op->getResult(i_result).getType()),
                                if_op->getResult(i_result));
                            // Substitute scf.if results with from_element results
                            rewriter.replaceUsesWithIf(
                                if_op->getResult(i_result), from_elem_op->getResult(0),
                                [&](OpOperand &operand) {
                                    return operand.getOwner() != from_elem_op;
                                });
                        }
                    }
                    i_result += 1;
                }
                first_yield = false;
            });
        });

        llvm::errs() << *root << '\n';

        // Find scf.for
        root->walk<WalkOrder::PreOrder>([&](scf::ForOp for_op) {
            llvm::errs() << for_op << '\n';
            bool has_for = false;
            // for_op->walk([&](scf::ForOp op) {
            //     has_for = true;
            // });
            if (!has_for) {
            // Find scf.yield
            scf::YieldOp yield_op = cast<scf::YieldOp>(*for_op.getBody()->getTerminator());
            // for_op->walk<WalkOrder::PreOrder>([&](scf::YieldOp yield_op) {
                llvm::errs() << yield_op << '\n';
                // Loop over results
                std::size_t i_result = 0;
                for (mlir::Value result : yield_op.getResults()) {
                llvm::errs() << result << '\n';
                    if (isFromElementScalarTensor(result)) {
                llvm::errs() << "Is a scalar tensor" << '\n';
                        // Detensorize operand and modify type
                        {
                            auto iter_arg = for_op.getOperand(3 + i_result);
                            // llvm::errs() << iter_arg << '\n'; 
                            mlir::Operation *from_elem_op = iter_arg.getDefiningOp();
                            rewriter.replaceAllUsesWith(iter_arg, from_elem_op->getOperand(0));
                            rewriter.modifyOpInPlace(for_op, [&]() {
                                for_op.getRegionIterArg(i_result).setType(
                                    from_elem_op->getOperand(0).getType());
                            });
                            rewriter.eraseOp(from_elem_op);
                        }
                        // Tensorize argument in the for loop block
                        {
                            // llvm::errs() << for_op.getBody() << '\n';
                            rewriter.setInsertionPointToStart(for_op.getBody());
                            auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                                for_op->getLoc(),
                                RankedTensorType::get({},
                                                      for_op.getRegionIterArg(i_result).getType()),
                                for_op.getRegionIterArg(i_result));
                            // Substitute scf.for arguments with from_element results
                            rewriter.replaceUsesWithIf(
                                for_op.getRegionIterArg(i_result), from_elem_op->getResult(0),
                                [&](OpOperand &operand) {
                                    return operand.getOwner() != from_elem_op;
                                });
                        }
                        // Remove from_element op and substitute argument and operand
                        {
                            mlir::Operation *from_elem_op = result.getDefiningOp();
                            rewriter.replaceAllUsesWith(result, from_elem_op->getOperand(0));
                            rewriter.modifyOpInPlace(for_op, [&]() {
                                for_op.getResult(i_result).setType(
                                    from_elem_op->getOperand(0).getType());
                            });
                            rewriter.eraseOp(from_elem_op);
                        }
                        // Create a from_element op after scf.for
                        { 
                            rewriter.setInsertionPointAfter(for_op);
                            auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                                for_op->getLoc(),
                                RankedTensorType::get({}, for_op->getResult(i_result).getType()),
                                for_op->getResult(i_result));
                            // Substitute scf.for results with from_element results
                            rewriter.replaceUsesWithIf(
                                for_op->getResult(i_result), from_elem_op->getResult(0),
                                [&](OpOperand &operand) {
                                    return operand.getOwner() != from_elem_op;
                                });
                        }
                    }
                    i_result += 1;
                }
            // });
            // llvm::errs() << for_op << '\n';
            // for (auto args : for_op.getOperands()) {
            //     llvm::errs() << args << '\n';
            // }
            // for (auto args : for_op.getRegionIterArgs()) {
            //     llvm::errs() << args << '\n';
            // }
            // for (auto args : for_op.getResults()) {
            //     llvm::errs() << args << '\n';
            // }
            }
            llvm::errs() << *root << '\n';
        });

        // llvm::errs() << "|||||||||||||||||| Final IR ||||||||||||||||||\n";
        // root->walk([&](mlir::Operation *op) { llvm::errs() << *op << '\n'; });
        // llvm::errs() << "|||||||||||||||||| Final IR ||||||||||||||||||\n";
    }
};

std::unique_ptr<Pass> createDetensorizeSCFPass() { return std::make_unique<DetensorizeSCFPass>(); }

} // namespace catalyst