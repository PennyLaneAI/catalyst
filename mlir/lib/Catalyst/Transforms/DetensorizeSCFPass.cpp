#define DEBUG_TYPE "myhelloworld"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

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
        SmallVector<std::size_t> indices;
        SmallVector<Value> newIterOperands;
        std::size_t index = 0;
        for (OpOperand &opOperand : forOp.getInitArgsMutable()) {
            if (isScalarTensor(opOperand.get())) {
                rewriter.setInsertionPoint(forOp);
                Value value = rewriter.create<tensor::ExtractOp>(forOp->getLoc(), opOperand.get(),
                                                                 ValueRange{});
                newIterOperands.push_back(value);
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
            Value value = rewriter.create<tensor::FromElementsOp>(
                newForOp->getLoc(),
                RankedTensorType::get({}, newForOp.getRegionIterArg(index).getType()),
                newForOp.getRegionIterArg(index));
            rewriter.replaceUsesWithIf(newForOp.getRegionIterArg(index), value, [&](OpOperand &op) {
                return !isa<tensor::FromElementsOp>(op.getOwner());
            });
        }

        // 5. Extract scalar tensor elements and detensorize terminator
        scf::YieldOp clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
        SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
        rewriter.setInsertionPoint(clonedYieldOp);
        for (std::size_t index : indices) {
            newYieldOperands[index] = rewriter.create<tensor::ExtractOp>(
                clonedYieldOp->getLoc(), clonedYieldOp->getOperand(index), ValueRange{});
        }
        rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
        rewriter.eraseOp(clonedYieldOp);

        // 6. Retensorize results after for loop
        rewriter.setInsertionPointAfter(forOp);
        for (std::size_t index : indices) {
            Value for_result = newForOp->getResult(index);
            Value value = rewriter.create<tensor::FromElementsOp>(
                newForOp->getLoc(), RankedTensorType::get({}, for_result.getType()), for_result);
            rewriter.replaceUsesWithIf(forOp->getResult(index), value, [&](OpOperand &op) {
                return !isa<tensor::FromElementsOp>(op.getOwner());
            });
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
        // Collect the types for the new IfOp
        SmallVector<Type> newResultTypes;
        for (auto result : ifOp->getResults()) {
            if (isScalarTensor(result)) {
                newResultTypes.push_back(cast<TensorType>(result.getType()).getElementType());
            }
            else {
                newResultTypes.push_back(result.getType());
            }
        }
        // Extract tensor elements before yield op
        ifOp->walk([&](scf::YieldOp yield_op) {
            // Loop over yield operands
            std::size_t i_result = 0;
            for (Value operand : yield_op.getOperands()) {
                // Detensorize operand: extract tensor element before yielding
                if (isScalarTensor(operand)) {
                    PatternRewriter::InsertionGuard insertGuard(rewriter);
                    rewriter.setInsertionPoint(yield_op);
                    Value value = rewriter.create<tensor::ExtractOp>(yield_op->getLoc(), operand,
                                                                     ValueRange{});
                    yield_op.setOperand(i_result, value);
                }
                i_result += 1;
            }
        });

        // 2. Create the new IfOp
        PatternRewriter::InsertionGuard ifOpInsertGuard(rewriter);
        rewriter.setInsertionPoint(ifOp);
        auto newIfOp =
            rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes, ifOp.getCondition());
        rewriter.cloneRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                   newIfOp.getThenRegion().end());
        rewriter.cloneRegionBefore(ifOp.getElseRegion(), newIfOp.getElseRegion(),
                                   newIfOp.getElseRegion().end());

        // 3. Retensorize results after if op
        {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointAfter(ifOp);
            for (auto results : llvm::zip(ifOp->getResults(), newIfOp->getResults())) {
                auto oldResult = std::get<0>(results);
                auto newResult = std::get<1>(results);
                Value value = rewriter.create<tensor::FromElementsOp>(
                    ifOp->getLoc(), RankedTensorType::get({}, newResult.getType()), newResult);
                rewriter.replaceAllUsesWith(oldResult, value);
            }
        }

        // 4. Replace if op with new if op
        rewriter.replaceOp(ifOp, newIfOp);
        return success();
    }
};
} // namespace

namespace catalyst {
#define GEN_PASS_DEF_DETENSORIZESCFPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetensorizeSCFPass : public impl::DetensorizeSCFPassBase<DetensorizeSCFPass> {
    using impl::DetensorizeSCFPassBase<DetensorizeSCFPass>::DetensorizeSCFPassBase;

    void detensorizeForOp(scf::ForOp forOp, IRRewriter &rewriter)
    {
        // 1. Find scalar tensors and extract element
        SmallVector<std::size_t> indices;
        SmallVector<Value> newIterOperands;
        std::size_t index = 0;
        for (OpOperand &opOperand : forOp.getInitArgsMutable()) {
            if (isScalarTensor(opOperand.get())) {
                rewriter.setInsertionPoint(forOp);
                Value value = rewriter.create<tensor::ExtractOp>(forOp->getLoc(), opOperand.get(),
                                                                 ValueRange{});
                newIterOperands.push_back(value);
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
            Value value = rewriter.create<tensor::FromElementsOp>(
                newForOp->getLoc(),
                RankedTensorType::get({}, newForOp.getRegionIterArg(index).getType()),
                newForOp.getRegionIterArg(index));
            rewriter.replaceUsesWithIf(newForOp.getRegionIterArg(index), value, [&](OpOperand &op) {
                return !isa<tensor::FromElementsOp>(op.getOwner());
            });
        }

        // 5. Extract scalar tensor elements and detensorize terminator
        scf::YieldOp clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
        SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
        rewriter.setInsertionPoint(clonedYieldOp);
        for (std::size_t index : indices) {
            newYieldOperands[index] = rewriter.create<tensor::ExtractOp>(
                clonedYieldOp->getLoc(), clonedYieldOp->getOperand(index), ValueRange{});
        }
        rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
        rewriter.eraseOp(clonedYieldOp);

        // 6. Retensorize results after for loop
        rewriter.setInsertionPointAfter(forOp);
        for (std::size_t index : indices) {
            Value for_result = newForOp->getResult(index);
            Value value = rewriter.create<tensor::FromElementsOp>(
                newForOp->getLoc(), RankedTensorType::get({}, for_result.getType()), for_result);
            rewriter.replaceUsesWithIf(forOp->getResult(index), value, [&](OpOperand &op) {
                return !isa<tensor::FromElementsOp>(op.getOwner());
            });
        }

        rewriter.replaceOp(forOp, newForOp);
    }

    void detensorizeIfOp(scf::IfOp ifOp, IRRewriter &rewriter)
    {
        if (!hasScalarTensorResult(ifOp)) {
            return;
        }
        // Collect the types for the new IfOp
        SmallVector<Type> newResultTypes;
        for (auto result : ifOp->getResults()) {
            if (isScalarTensor(result)) {
                newResultTypes.push_back(cast<TensorType>(result.getType()).getElementType());
            }
            else {
                newResultTypes.push_back(result.getType());
            }
        }
        // Extract tensor elements before yield op
        ifOp->walk([&](scf::YieldOp yield_op) {
            // Loop over yield operands
            std::size_t i_result = 0;
            for (Value operand : yield_op.getOperands()) {
                // Detensorize operand: extract tensor element before yielding
                if (isScalarTensor(operand)) {
                    PatternRewriter::InsertionGuard insertGuard(rewriter);
                    rewriter.setInsertionPoint(yield_op);
                    Value value = rewriter.create<tensor::ExtractOp>(yield_op->getLoc(), operand,
                                                                     ValueRange{});
                    yield_op.setOperand(i_result, value);
                }
                i_result += 1;
            }
        });

        // 2. Create the new IfOp
        PatternRewriter::InsertionGuard ifOpInsertGuard(rewriter);
        rewriter.setInsertionPoint(ifOp);
        auto newIfOp =
            rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes, ifOp.getCondition());
        rewriter.cloneRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                   newIfOp.getThenRegion().end());
        rewriter.cloneRegionBefore(ifOp.getElseRegion(), newIfOp.getElseRegion(),
                                   newIfOp.getElseRegion().end());

        // 3. Retensorize results after if op
        {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointAfter(ifOp);
            for (auto results : llvm::zip(ifOp->getResults(), newIfOp->getResults())) {
                auto oldResult = std::get<0>(results);
                auto newResult = std::get<1>(results);
                Value value = rewriter.create<tensor::FromElementsOp>(
                    ifOp->getLoc(), RankedTensorType::get({}, newResult.getType()), newResult);
                rewriter.replaceAllUsesWith(oldResult, value);
            }
        }

        // 4. Replace if op with new if op
        rewriter.replaceOp(ifOp, newIfOp);
    }

    void detensorizeWhileOp(Operation *root, scf::WhileOp whileOp, IRRewriter &rewriter)
    {
        SmallVector<Type> newResultTypes;
        for (auto result : whileOp->getResults()) {
            if (isScalarTensor(result)) {
                newResultTypes.push_back(cast<TensorType>(result.getType()).getElementType());
            }
            else {
                newResultTypes.push_back(result.getType());
            }
        }

        // 1. Find scalar tensors and extract element
        SmallVector<std::size_t> indices;
        SmallVector<Value> newIterOperands;
        std::size_t index = 0;
        for (Value opOperand : whileOp.getOperands()) {
            if (isScalarTensor(opOperand)) {
                rewriter.setInsertionPoint(whileOp);
                Value value =
                    rewriter.create<tensor::ExtractOp>(whileOp->getLoc(), opOperand, ValueRange{});
                newIterOperands.push_back(value);
                indices.push_back(index);
                index += 1;
                continue;
            }
            newIterOperands.push_back(opOperand);
            index += 1;
        }

        // 2. Create the new WhileOp shell.
        scf::WhileOp newWhileOp =
            rewriter.create<scf::WhileOp>(whileOp.getLoc(), newResultTypes, newIterOperands,
                                          /*beforeBody*/ nullptr, /*afterBody*/ nullptr);

        // 3. Copy body
        Block &beforeBlock = *whileOp.getBeforeBody();
        Block &afterBlock = *whileOp.getAfterBody();
        Block &newBeforeBlock = *newWhileOp.getBeforeBody();
        Block &newAfterBlock = *newWhileOp.getAfterBody();
        rewriter.mergeBlocks(&beforeBlock, &newBeforeBlock, newBeforeBlock.getArguments());
        rewriter.mergeBlocks(&afterBlock, &newAfterBlock, newAfterBlock.getArguments());

        // 4. Retensorize arguments at the beginning of the body
        rewriter.setInsertionPoint(&newBeforeBlock, newBeforeBlock.begin());
        for (std::size_t index : indices) {

            Value value = rewriter.create<tensor::FromElementsOp>(
                newWhileOp->getLoc(),
                RankedTensorType::get({}, newBeforeBlock.getArgument(index).getType()),
                newBeforeBlock.getArgument(index));
            rewriter.replaceUsesWithIf(newBeforeBlock.getArgument(index), value,
                                       [&](OpOperand &op) {
                                           return !isa<tensor::FromElementsOp>(op.getOwner()) &&
                                                  !isa<scf::WhileOp>(op.getOwner());
                                       });
        }

        scf::ConditionOp condOp = newWhileOp.getConditionOp();
        SmallVector<Value> newCondOpArgs;
        OperandRange condOpArgs = condOp.getArgs();
        for (const auto &it : llvm::enumerate(condOpArgs)) {
            Value condOpArg = it.value();
            if (isScalarTensor(condOpArg)) {
                OpBuilder::InsertionGuard g(rewriter);
                rewriter.setInsertionPoint(condOp);
                Value value =
                    rewriter.create<tensor::ExtractOp>(condOp->getLoc(), condOpArg, ValueRange{});
                newCondOpArgs.push_back(value);
            }
            else {
                newCondOpArgs.emplace_back(condOpArg);
            }
        }

        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(condOp);
            rewriter.replaceOpWithNewOp<scf::ConditionOp>(condOp, condOp.getCondition(),
                                                          newCondOpArgs);
        }

        rewriter.setInsertionPoint(&newAfterBlock, newAfterBlock.begin());
        for (std::size_t index : indices) {
            Value value = rewriter.create<tensor::FromElementsOp>(
                newWhileOp->getLoc(),
                RankedTensorType::get({}, newAfterBlock.getArgument(index).getType()),
                newAfterBlock.getArgument(index));
            rewriter.replaceUsesWithIf(newAfterBlock.getArgument(index), value, [&](OpOperand &op) {
                return !isa<tensor::FromElementsOp>(op.getOwner());
            });
        }

        // 5. Extract scalar tensor elements and detensorize terminator
        scf::YieldOp clonedYieldOp = cast<scf::YieldOp>(newAfterBlock.getTerminator());
        SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
        rewriter.setInsertionPoint(clonedYieldOp);
        for (std::size_t index : indices) {
            newYieldOperands[index] = rewriter.create<tensor::ExtractOp>(
                clonedYieldOp->getLoc(), clonedYieldOp->getOperand(index), ValueRange{});
        }
        rewriter.create<scf::YieldOp>(newWhileOp.getLoc(), newYieldOperands);
        rewriter.eraseOp(clonedYieldOp);

        // llvm::errs() << "========================================" << '\n';
        // llvm::errs() << *root << '\n';
        // llvm::errs() << "========================================" << '\n';

        // 6. Retensorize results after for loop
        rewriter.setInsertionPointAfter(whileOp);
        for (std::size_t index : indices) {
            Value for_result = newWhileOp->getResult(index);
            Value value = rewriter.create<tensor::FromElementsOp>(
                newWhileOp->getLoc(), RankedTensorType::get({}, for_result.getType()), for_result);
            rewriter.replaceUsesWithIf(whileOp->getResult(index), value, [&](OpOperand &op) {
                return !isa<tensor::FromElementsOp>(op.getOwner());
            });
        }

        rewriter.replaceOp(whileOp, newWhileOp);
    }

    void runOnOperation() override
    {
        Operation *root = getOperation();

        IRRewriter rewriter(root->getContext());
        root->walk([&](scf::ForOp for_op) { detensorizeForOp(for_op, rewriter); });
        root->walk([&](scf::IfOp if_op) { detensorizeIfOp(if_op, rewriter); });
        root->walk([&](scf::WhileOp while_op) { detensorizeWhileOp(root, while_op, rewriter); });

        // MLIRContext *context = &getContext();
        // ConversionTarget target(*context);
        // RewritePatternSet patterns(context);

        // target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

        // target.addDynamicallyLegalOp<scf::ForOp>(
        //     [&](scf::ForOp op) { return !hasScalarTensorOperand(op) &&
        //     !hasScalarTensorResult(op); });
        // patterns.add<DetensorizeForOp>(patterns.getContext());

        // target.addDynamicallyLegalOp<scf::IfOp>(
        //     [&](scf::IfOp op) { return !hasScalarTensorResult(op); });
        // patterns.add<DetensorizeIfOp>(patterns.getContext());

        // if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        //     signalPassFailure();

        // llvm::errs() << *root << '\n';
    }
};

std::unique_ptr<Pass> createDetensorizeSCFPass() { return std::make_unique<DetensorizeSCFPass>(); }

} // namespace catalyst
