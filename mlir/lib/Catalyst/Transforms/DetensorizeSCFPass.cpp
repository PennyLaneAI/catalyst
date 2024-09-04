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

        // Find scf.if
        root->walk([&](scf::IfOp if_op) {
            bool first_yield = false;
            if_op->walk([&](scf::YieldOp yield_op) {
                // Loop over results
                std::size_t i_result = 0;
                for (mlir::Value operand : yield_op.getOperands()) {
                    if (isScalarTensor(operand)) {
                        auto if_result = if_op->getResult(i_result);
                        // Detensorize operand: extract tensor element before yielding
                        {
                            rewriter.setInsertionPoint(yield_op);
                            mlir::Operation *extract_op = rewriter.create<tensor::ExtractOp>(
                                yield_op->getLoc(), operand, ValueRange{});
                            rewriter.replaceUsesWithIf(
                                operand, extract_op->getResult(0),
                                [&](OpOperand &op) { return op.getOwner() == yield_op; });
                            const mlir::Type type = extract_op->getResult(0).getType();
                            rewriter.modifyOpInPlace(
                                yield_op, [&]() { yield_op->getOperand(i_result).setType(type); });
                            rewriter.modifyOpInPlace(if_op, [&]() { if_result.setType(type); });
                        }
                        // Retensorize operand: reconstruct tensor after the for body
                        if (first_yield) {
                            rewriter.setInsertionPointAfter(if_op);
                            auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                                if_op->getLoc(), RankedTensorType::get({}, if_result.getType()),
                                if_result);
                            rewriter.replaceUsesWithIf(
                                if_result, from_elem_op->getResult(0), [&](OpOperand &operand) {
                                    return !isa<tensor::FromElementsOp>(operand.getOwner());
                                });
                        }
                    }
                    i_result += 1;
                }
                first_yield = true;
            });
        });

        // Find scf.for
        root->walk([&](scf::ForOp for_op) {
            // Loop over operands
            std::size_t i_operand = 0;
            for (mlir::Value operand : for_op.getOperands()) {
                if (isScalarTensor(operand)) {
                    const std::size_t i_argument = i_operand - 3;
                    // Detensorize operand: extract tensor element before for operation and replace
                    // operand
                    {
                        rewriter.setInsertionPoint(for_op);
                        mlir::Operation *extract_op = rewriter.create<tensor::ExtractOp>(
                            for_op->getLoc(), operand, ValueRange{});
                        rewriter.replaceUsesWithIf(
                            operand, extract_op->getResult(0),
                            [&](OpOperand &op) { return op.getOwner() == for_op; });
                        rewriter.modifyOpInPlace(for_op, [&]() {
                            for_op.getRegionIterArg(i_argument)
                                .setType(extract_op->getResult(0).getType());
                        });
                    }
                    // Retensorize argument: reconstruct tensor at the beginning of the for body
                    {
                        rewriter.setInsertionPointToStart(for_op.getBody());
                        auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                            for_op->getLoc(),
                            RankedTensorType::get({},
                                                  for_op.getRegionIterArg(i_argument).getType()),
                            for_op.getRegionIterArg(i_argument));
                        rewriter.replaceUsesWithIf(
                            for_op.getRegionIterArg(i_argument), from_elem_op->getResult(0),
                            [&](OpOperand &op) { return op.getOwner() != from_elem_op; });
                    }
                }
                i_operand += 1;
            }
            scf::YieldOp yield_op = cast<scf::YieldOp>(*for_op.getBody()->getTerminator());
            // Loop over results
            std::size_t i_result = 0;
            for (mlir::Value result : yield_op.getOperands()) {
                if (isScalarTensor(result)) {
                    // Detensorize result: extract tensor element before yielding
                    {
                        rewriter.setInsertionPoint(yield_op);
                        mlir::Operation *extract_op = rewriter.create<tensor::ExtractOp>(
                            yield_op->getLoc(), result, ValueRange{});
                        rewriter.replaceUsesWithIf(
                            result, extract_op->getResult(0),
                            [&](OpOperand &op) { return op.getOwner() != extract_op; });
                        rewriter.modifyOpInPlace(yield_op, [&]() {
                            yield_op->getOperand(i_result).setType(
                                extract_op->getResult(0).getType());
                        });
                        rewriter.modifyOpInPlace(for_op, [&]() {
                            for_op.getResult(i_result).setType(extract_op->getResult(0).getType());
                        });
                    }
                    // Retensorize result: reconstruct tensor after the for body
                    {
                        rewriter.setInsertionPointAfter(for_op);
                        auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                            for_op->getLoc(),
                            RankedTensorType::get({}, for_op->getResult(i_result).getType()),
                            for_op->getResult(i_result));
                        rewriter.replaceUsesWithIf(
                            for_op->getResult(i_result), from_elem_op->getResult(0),
                            [&](OpOperand &operand) { return operand.getOwner() != from_elem_op; });
                    }
                }
                i_result += 1;
            }
        });
    }
};

std::unique_ptr<Pass> createDetensorizeSCFPass() { return std::make_unique<DetensorizeSCFPass>(); }

} // namespace catalyst