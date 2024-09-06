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

    void detensorizeForOp(scf::ForOp forOp, IRRewriter &rewriter)
    {

        auto isScalarTensor = [](Value v) {
            if (!::llvm::isa<TensorType>(v.getType())) {
                return false;
            }
            return dyn_cast<TensorType>(v.getType()).getRank() == 0;
        };

        // 1. Find scalar tensors and extract element
        SmallVector<std::size_t> indices;
        SmallVector<Value> newIterOperands;
        std::size_t index = 0;
        for (OpOperand &opOperand : forOp.getInitArgsMutable()) {
            if (isScalarTensor(opOperand.get())) {
                rewriter.setInsertionPoint(forOp);
                mlir::Operation *extract_op = rewriter.create<tensor::ExtractOp>(
                    forOp->getLoc(), opOperand.get(), ValueRange{});
                newIterOperands.push_back(extract_op->getResult(0));
                indices.push_back(index);
                index += 1;
                continue;
            }
            newIterOperands.push_back(opOperand.get());
            index += 1;
        }

        // 2. Create the new ForOp shell.
        scf::ForOp newForOp =
            rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(),
                                        forOp.getUpperBound(), forOp.getStep(), newIterOperands);
        newForOp->setAttrs(forOp->getAttrs());
        Block &newBlock = newForOp.getRegion().front();
        SmallVector<Value> newBlockTransferArgs(newBlock.getArguments().begin(),
                                                newBlock.getArguments().end());

        // 3. Copy body
        Block &oldBlock = forOp.getRegion().front();
        rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);

        // 4. Retensorize arguments at the beginning of the body
        rewriter.setInsertionPoint(&newBlock, newBlock.begin());
        for (std::size_t index : indices) {
            auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                newForOp->getLoc(),
                RankedTensorType::get({}, newForOp.getRegionIterArg(index).getType()),
                newForOp.getRegionIterArg(index));
            rewriter.replaceUsesWithIf(
                newForOp.getRegionIterArg(index), from_elem_op->getResult(0),
                [&](OpOperand &op) { return !isa<tensor::FromElementsOp>(op.getOwner()); });
        }

        // 5. Extract scalar tensor elements and detensorize terminator
        auto clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
        SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
        rewriter.setInsertionPoint(clonedYieldOp);
        for (std::size_t index : indices) {
            mlir::Operation *extract_op = rewriter.create<tensor::ExtractOp>(
                clonedYieldOp->getLoc(), clonedYieldOp->getOperand(index), ValueRange{});
            newYieldOperands[index] = extract_op->getResult(0);
        }
        rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
        rewriter.eraseOp(clonedYieldOp);

        // 6. Retensorize results after for loop
        rewriter.setInsertionPointAfter(forOp);
        for (std::size_t index : indices) {
            mlir::Value for_result = newForOp->getResult(index);
            auto from_elem_op = rewriter.create<tensor::FromElementsOp>(
                newForOp->getLoc(), RankedTensorType::get({}, for_result.getType()), for_result);
            rewriter.replaceUsesWithIf(
                forOp->getResult(index), from_elem_op->getResult(0),
                [&](OpOperand &op) { return !isa<tensor::FromElementsOp>(op.getOwner()); });
        }

        rewriter.replaceOp(forOp, newForOp);
    }

    void detensorizeIfOp(scf::IfOp if_op, IRRewriter &rewriter)
    {
        auto isScalarTensor = [](Value v) {
            if (!::llvm::isa<TensorType>(v.getType())) {
                return false;
            }
            return dyn_cast<TensorType>(v.getType()).getRank() == 0;
        };

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
                            if_result, from_elem_op->getResult(0), [&](OpOperand &op) {
                                return !isa<tensor::FromElementsOp>(op.getOwner());
                            });
                    }
                }
                i_result += 1;
            }
            first_yield = true;
        });
    }

    void runOnOperation() override
    {
        mlir::Operation *root = getOperation();
        IRRewriter rewriter(root->getContext());

        // llvm::errs() << *root << '\n';

        root->walk([&](scf::ForOp for_op) { detensorizeForOp(for_op, rewriter); });

        // llvm::errs() << *root << '\n';

        root->walk([&](scf::IfOp if_op) { detensorizeIfOp(if_op, rewriter); });
    }
};

std::unique_ptr<Pass> createDetensorizeSCFPass() { return std::make_unique<DetensorizeSCFPass>(); }

} // namespace catalyst