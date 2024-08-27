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
        auto isScalarTensor = [](Value v) {
            if (!::llvm::isa<TensorType>(v.getType())) {
                return false;
            }
            return dyn_cast<TensorType>(v.getType()).getRank() == 0;
        };
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

        // llvm::errs() << *root << '\n';

        // Find scf.for
        root->walk([&](scf::ForOp for_op) {
            // llvm::errs() << for_op << '\n';

            std::size_t i_operand = 0;
            for (mlir::Value operand : for_op.getOperands()) {
                if (isScalarTensor(operand)) {
                    // llvm::errs() << operand << '(' << i_operand << ')' << '\n';

                    // Detensorize operand and modify type
                    {
                        // auto operand = for_op.getOperand(3 + i_operand);
                        // llvm::errs() << operand << '\n';
                        // mlir::Operation *operand_def = operand.getDefiningOp();
                        // rewriter.setInsertionPointAfter(operand_def);
                        rewriter.setInsertionPoint(for_op);
                        mlir::Operation *extract_op = rewriter.create<tensor::ExtractOp>(for_op->getLoc(), operand, ValueRange{});
                        // rewriter.replaceAllUsesWith(operand, extract_op->getResult(0));
                        rewriter.replaceUsesWithIf(
                            operand, extract_op->getResult(0),
                            [&](OpOperand &op) { return op.getOwner() != extract_op; });
                        rewriter.modifyOpInPlace(for_op, [&]() {
                            for_op.getRegionIterArg(i_operand - 3).setType(
                                extract_op->getResult(0).getType());
                        });
                    }
                    // Tensorize argument in the for loop block
                    {
                        // llvm::errs() << for_op.getBody() << '\n';
                        rewriter.setInsertionPointToStart(for_op.getBody());
                        auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                            for_op->getLoc(),
                            RankedTensorType::get({}, for_op.getRegionIterArg(i_operand - 3).getType()),
                            for_op.getRegionIterArg(i_operand - 3));
                        // Substitute scf.for arguments with from_element results
                        rewriter.replaceUsesWithIf(
                            for_op.getRegionIterArg(i_operand - 3), from_elem_op->getResult(0),
                            [&](OpOperand &operand) { return operand.getOwner() != from_elem_op; });
                    }
                }
                i_operand += 1;
            }
            // llvm::errs() << for_op << '\n';

            scf::YieldOp yield_op = cast<scf::YieldOp>(*for_op.getBody()->getTerminator());
            // llvm::errs() << yield_op << '\n';
            // Loop over results
            std::size_t i_result = 0;
            for (mlir::Value result : yield_op.getOperands()) {
                // llvm::errs() << result << '\n';
                if (isScalarTensor(result)) {
                    // llvm::errs() << "Is a scalar tensor" << '\n';
                    // // Detensorize operand and modify type
                    // {
                    //     auto iter_arg = for_op.getOperand(3 + i_result);
                    //     // llvm::errs() << iter_arg << '\n';
                    //     mlir::Operation *from_elem_op = iter_arg.getDefiningOp();
                    //     rewriter.replaceAllUsesWith(iter_arg, from_elem_op->getOperand(0));
                    //     rewriter.modifyOpInPlace(for_op, [&]() {
                    //         for_op.getRegionIterArg(i_result).setType(
                    //             from_elem_op->getOperand(0).getType());
                    //     });
                    //     rewriter.eraseOp(from_elem_op);
                    // }
                    // // Tensorize argument in the for loop block
                    // {
                    //     // llvm::errs() << for_op.getBody() << '\n';
                    //     rewriter.setInsertionPointToStart(for_op.getBody());
                    //     auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                    //         for_op->getLoc(),
                    //         RankedTensorType::get({},
                    //         for_op.getRegionIterArg(i_result).getType()),
                    //         for_op.getRegionIterArg(i_result));
                    //     // Substitute scf.for arguments with from_element results
                    //     rewriter.replaceUsesWithIf(
                    //         for_op.getRegionIterArg(i_result), from_elem_op->getResult(0),
                    //         [&](OpOperand &operand) { return operand.getOwner() != from_elem_op;
                    //         });
                    // }

                    // Remove from_element op and substitute argument and operand
                    {
                        
                        rewriter.setInsertionPoint(yield_op);
                        mlir::Operation *extract_op = rewriter.create<tensor::ExtractOp>(yield_op->getLoc(), result, ValueRange{});
                        rewriter.replaceUsesWithIf(
                            result, extract_op->getResult(0),
                            [&](OpOperand &op) { return op.getOwner() != extract_op; });
                        rewriter.modifyOpInPlace(yield_op, [&]() {
                            yield_op->getOperand(i_result).setType(
                                extract_op->getResult(0).getType());
                        });

                        // mlir::Operation *from_elem_op = result.getDefiningOp();
                        // rewriter.replaceAllUsesWith(result, from_elem_op->getOperand(0));
                        rewriter.modifyOpInPlace(for_op, [&]() {
                            for_op.getResult(i_result).setType(
                                extract_op->getResult(0).getType());
                        });
                        // rewriter.eraseOp(from_elem_op);
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
                            [&](OpOperand &operand) { return operand.getOwner() != from_elem_op;
                            });
                    }
                }
                i_result += 1;
            }

            // llvm::errs() << for_op << '\n';
            // llvm::errs() << *root << '\n';
        });

        // llvm::errs() << "|||||||||||||||||| Final IR ||||||||||||||||||\n";
        // root->walk([&](mlir::Operation *op) { llvm::errs() << *op << '\n'; });
        // llvm::errs() << "|||||||||||||||||| Final IR ||||||||||||||||||\n";
    }
};

std::unique_ptr<Pass> createDetensorizeSCFPass() { return std::make_unique<DetensorizeSCFPass>(); }

} // namespace catalyst