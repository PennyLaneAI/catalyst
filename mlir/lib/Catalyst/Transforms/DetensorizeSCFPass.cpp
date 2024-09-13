#define DEBUG_TYPE "myhelloworld"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace {
bool isScalarTensor(Value v)
{
    Type type = v.getType();
    if (!isa<TensorType>(type)) {
        return false;
    }
    return cast<TensorType>(type).getRank() == 0;
};

template <typename ScfOp> bool hasScalarTensorOperand(ScfOp op)
{
    for (Value result : op->getOperands()) {
        if (isScalarTensor(result)) {
            return true;
        }
    }
    return false;
}

template <typename ScfOp> bool hasScalarTensorResult(ScfOp op)
{
    for (Value result : op->getResults()) {
        if (isScalarTensor(result)) {
            return true;
        }
    }
    return false;
}

struct DetensorizeForOp : public OpRewritePattern<scf::ForOp> {
    using OpRewritePattern<scf::ForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override
    {
        if (!hasScalarTensorOperand(forOp) && !hasScalarTensorResult(forOp)) {
            return failure();
        }

        // 1. Find scalar tensors and extract element
        SmallVector<std::size_t> newIterOperandsIndices;
        SmallVector<Value> newIterOperands;
        for (const auto &it : llvm::enumerate(forOp.getInitArgsMutable())) {
            OpOperand &opOperand = it.value();
            if (isScalarTensor(opOperand.get())) {
                OpBuilder::InsertionGuard g(rewriter);
                rewriter.setInsertionPoint(forOp);
                Value value = rewriter.create<tensor::ExtractOp>(forOp->getLoc(), opOperand.get(),
                                                                 ValueRange{});
                newIterOperands.push_back(value);
                newIterOperandsIndices.push_back(it.index());
                continue;
            }
            newIterOperands.push_back(opOperand.get());
        }

        // 2. Create the new ForOp shell.
        OpBuilder::InsertionGuard forOpInsertionGuard(rewriter);
        rewriter.setInsertionPoint(forOp);
        scf::ForOp newForOp =
            rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(),
                                        forOp.getUpperBound(), forOp.getStep(), newIterOperands);
        newForOp->setAttrs(forOp->getAttrs());

        // 3. Copy body
        Block &newBlock = newForOp.getRegion().front();
        SmallVector<Value> newBlockTransferArgs(newBlock.getArguments().begin(),
                                                newBlock.getArguments().end());
        Block &oldBlock = forOp.getRegion().front();
        rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);

        // 4. Retensorize arguments at the beginning of the body
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(&newBlock, newBlock.begin());
            for (std::size_t index : newIterOperandsIndices) {
                Value value = rewriter.create<tensor::FromElementsOp>(
                    newForOp->getLoc(),
                    RankedTensorType::get({}, newForOp.getRegionIterArg(index).getType()),
                    newForOp.getRegionIterArg(index));
                rewriter.replaceUsesWithIf(
                    newForOp.getRegionIterArg(index), value,
                    [&](OpOperand &op) { return !isa<tensor::FromElementsOp>(op.getOwner()); });
            }
        }

        // 5. Extract scalar tensor elements and detensorize terminator
        {
            scf::YieldOp clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
            SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(clonedYieldOp);
            for (std::size_t index : newIterOperandsIndices) {
                newYieldOperands[index] = rewriter.create<tensor::ExtractOp>(
                    clonedYieldOp->getLoc(), clonedYieldOp->getOperand(index), ValueRange{});
            }
            rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
            rewriter.eraseOp(clonedYieldOp);
        }

        // 6. Retensorize results after for loop
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPointAfter(forOp);
            for (std::size_t index : newIterOperandsIndices) {
                Value for_result = newForOp->getResult(index);
                Value value = rewriter.create<tensor::FromElementsOp>(
                    newForOp->getLoc(), RankedTensorType::get({}, for_result.getType()),
                    for_result);
                rewriter.replaceUsesWithIf(forOp->getResult(index), value, [&](OpOperand &op) {
                    return !isa<tensor::FromElementsOp>(op.getOwner());
                });
            }
        }

        rewriter.replaceOp(forOp, newForOp);
        return success();
    }
};

struct DetensorizeIfOp : public OpRewritePattern<scf::IfOp> {
    using OpRewritePattern<scf::IfOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::IfOp ifOp, PatternRewriter &rewriter) const override
    {
        // Early exit if there are no results that could be replaced.
        if (ifOp.getNumResults() == 0) {
            return failure();
        }
        if (!hasScalarTensorResult(ifOp)) {
            return failure();
        }

        // 1. Extract tensor elements before yield op
        SmallVector<scf::YieldOp> yieldOps = {ifOp.thenYield(), ifOp.elseYield()};
        for (scf::YieldOp yield_op : yieldOps) {
            // Loop over yield operands
            for (const auto &it : llvm::enumerate(yield_op.getOperands())) {
                Value operand = it.value();
                // Detensorize operand: extract tensor element before yielding
                if (isScalarTensor(operand)) {
                    OpBuilder::InsertionGuard g(rewriter);
                    rewriter.setInsertionPoint(yield_op);
                    Value value = rewriter.create<tensor::ExtractOp>(yield_op->getLoc(), operand,
                                                                     ValueRange{});
                    yield_op.setOperand(it.index(), value);
                }
            }
        };

        // Collect the types for the new IfOp
        SmallVector<Type> newResultTypes;
        for (auto result : ifOp->getResults()) {
            if (isScalarTensor(result)) {
                newResultTypes.push_back(cast<TensorType>(result.getType()).getElementType());
                continue;
            }
            newResultTypes.push_back(result.getType());
        }

        // 2. Create the new IfOp
        OpBuilder::InsertionGuard ifOpInsertionGuard(rewriter);
        rewriter.setInsertionPoint(ifOp);
        auto newIfOp =
            rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes, ifOp.getCondition());
        rewriter.cloneRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                   newIfOp.getThenRegion().end());
        rewriter.cloneRegionBefore(ifOp.getElseRegion(), newIfOp.getElseRegion(),
                                   newIfOp.getElseRegion().end());

        // 3. Retensorize results after if op
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPointAfter(ifOp);
            for (auto results : llvm::zip(ifOp->getResults(), newIfOp->getResults())) {
                auto oldResult = std::get<0>(results);
                auto newResult = std::get<1>(results);
                if (isScalarTensor(oldResult)) {
                    Value value = rewriter.create<tensor::FromElementsOp>(
                        ifOp->getLoc(), RankedTensorType::get({}, newResult.getType()), newResult);
                    rewriter.replaceAllUsesWith(oldResult, value);
                }
            }
        }

        // 4. Replace if op with new if op
        rewriter.replaceOp(ifOp, newIfOp);
        return success();
    }
};

struct DetensorizeWhileOp : public OpRewritePattern<scf::WhileOp> {
    using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::WhileOp whileOp, PatternRewriter &rewriter) const override
    {
        if (!hasScalarTensorOperand(whileOp)) {
            return failure();
        }

        Block &beforeBlock = *whileOp.getBeforeBody();
        Block &afterBlock = *whileOp.getAfterBody();

        // 1. Find scalar tensors and extract elements
        SmallVector<std::size_t> newIterOperandsIndices;
        SmallVector<Value> newIterOperands;
        for (const auto &it : llvm::enumerate(whileOp.getOperands())) {
            Value opOperand = it.value();
            if (isScalarTensor(opOperand)) {
                OpBuilder::InsertionGuard g(rewriter);
                rewriter.setInsertionPoint(whileOp);
                Value value =
                    rewriter.create<tensor::ExtractOp>(whileOp->getLoc(), opOperand, ValueRange{});
                newIterOperands.push_back(value);
                newIterOperandsIndices.push_back(it.index());
                continue;
            }
            newIterOperands.push_back(opOperand);
        }

        SmallVector<std::size_t> newResultIndices;
        SmallVector<Type> newResultTypes;
        for (const auto &it : llvm::enumerate(whileOp.getResults())) {
            Value opOperand = it.value();
            if (isScalarTensor(opOperand)) {
                newResultTypes.push_back(cast<TensorType>(opOperand.getType()).getElementType());
                newResultIndices.push_back(it.index());
                continue;
            }
            newResultTypes.push_back(opOperand.getType());
        }

        // 2. Create the new WhileOp shell.
        OpBuilder::InsertionGuard newWhileOpInsertionGuard(rewriter);
        rewriter.setInsertionPoint(whileOp);
        scf::WhileOp newWhileOp =
            rewriter.create<scf::WhileOp>(whileOp.getLoc(), newResultTypes, newIterOperands,
                                          /*beforeBody*/ nullptr, /*afterBody*/ nullptr);

        // 3. Copy body
        Block &newBeforeBlock = *newWhileOp.getBeforeBody();
        rewriter.mergeBlocks(&beforeBlock, &newBeforeBlock, newBeforeBlock.getArguments());

        Block &newAfterBlock = *newWhileOp.getAfterBody();
        rewriter.mergeBlocks(&afterBlock, &newAfterBlock, newAfterBlock.getArguments());

        // 4. Retensorize arguments at the beginning of the bodies
        for (std::size_t index : newIterOperandsIndices) {
            {
                OpBuilder::InsertionGuard g(rewriter);
                rewriter.setInsertionPoint(&newBeforeBlock, newBeforeBlock.begin());
                Value value = rewriter.create<tensor::FromElementsOp>(
                    newWhileOp->getLoc(),
                    RankedTensorType::get({}, newBeforeBlock.getArgument(index).getType()),
                    newBeforeBlock.getArgument(index));
                rewriter.replaceUsesWithIf(
                    newBeforeBlock.getArgument(index), value,
                    [&](OpOperand &op) { return !isa<tensor::FromElementsOp>(op.getOwner()); });
            }
        }

        for (std::size_t index : newResultIndices) {
            {
                OpBuilder::InsertionGuard g(rewriter);
                rewriter.setInsertionPoint(&newAfterBlock, newAfterBlock.begin());
                Value value = rewriter.create<tensor::FromElementsOp>(
                    newWhileOp->getLoc(),
                    RankedTensorType::get({}, newAfterBlock.getArgument(index).getType()),
                    newAfterBlock.getArgument(index));
                rewriter.replaceUsesWithIf(
                    newAfterBlock.getArgument(index), value,
                    [&](OpOperand &op) { return !isa<tensor::FromElementsOp>(op.getOwner()); });
            }
        }

        // 4. Detensorize condition arguments
        {
            scf::ConditionOp condOp = newWhileOp.getConditionOp();
            SmallVector<Value> newCondOpArgs;
            OperandRange condOpArgs = condOp.getArgs();
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(condOp);
            for (const auto &it : llvm::enumerate(condOpArgs)) {
                Value condOpArg = it.value();
                if (isScalarTensor(condOpArg)) {
                    Value value = rewriter.create<tensor::ExtractOp>(condOp->getLoc(), condOpArg,
                                                                     ValueRange{});
                    newCondOpArgs.push_back(value);
                }
                else {
                    newCondOpArgs.emplace_back(condOpArg);
                }
            }
            rewriter.replaceOpWithNewOp<scf::ConditionOp>(condOp, condOp.getCondition(),
                                                          newCondOpArgs);
        }

        // 5. Extract scalar tensor elements and detensorize terminator
        {
            scf::YieldOp clonedYieldOp = cast<scf::YieldOp>(newAfterBlock.getTerminator());
            SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(clonedYieldOp);
            for (std::size_t index : newIterOperandsIndices) {
                newYieldOperands[index] = rewriter.create<tensor::ExtractOp>(
                    clonedYieldOp->getLoc(), clonedYieldOp->getOperand(index), ValueRange{});
            }
            rewriter.create<scf::YieldOp>(newWhileOp.getLoc(), newYieldOperands);
            rewriter.eraseOp(clonedYieldOp);
        }

        // 6. Retensorize results after for loop
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPointAfter(whileOp);
            for (std::size_t index : newResultIndices) {
                Value for_result = newWhileOp->getResult(index);
                Value value = rewriter.create<tensor::FromElementsOp>(
                    newWhileOp->getLoc(), RankedTensorType::get({}, for_result.getType()),
                    for_result);
                rewriter.replaceUsesWithIf(whileOp->getResult(index), value, [&](OpOperand &op) {
                    return !isa<tensor::FromElementsOp>(op.getOwner());
                });
            }
        }

        rewriter.replaceOp(whileOp, newWhileOp);
        return success();
    }
};

} // namespace

namespace catalyst {
#define GEN_PASS_DEF_DETENSORIZESCFPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetensorizeSCFPass : public impl::DetensorizeSCFPassBase<DetensorizeSCFPass> {
    using impl::DetensorizeSCFPassBase<DetensorizeSCFPass>::DetensorizeSCFPassBase;
    void runOnOperation() override
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<DetensorizeForOp>(patterns.getContext());
        patterns.add<DetensorizeIfOp>(patterns.getContext());
        patterns.add<DetensorizeWhileOp>(patterns.getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> createDetensorizeSCFPass() { return std::make_unique<DetensorizeSCFPass>(); }

} // namespace catalyst
