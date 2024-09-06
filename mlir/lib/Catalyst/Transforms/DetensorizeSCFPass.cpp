#define DEBUG_TYPE "myhelloworld"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
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
} // namespace

namespace catalyst {
#define GEN_PASS_DEF_DETENSORIZESCFPASS
#include "Catalyst/Transforms/Passes.h.inc"

static Value sourceMaterializationCallback(OpBuilder &builder, Type type, ValueRange inputs,
                                           Location loc)
{
    assert(inputs.size() == 1);
    auto inputType = inputs[0].getType();
    if (isa<TensorType>(inputType))
        return nullptr;

    // A detensored value is converted back by creating a new tensor from its
    // element(s).
    return builder.create<tensor::FromElementsOp>(loc, RankedTensorType::get({}, inputType),
                                                  inputs[0]);
}

class DetensorizeTypeConverter : public TypeConverter {
  public:
    DetensorizeTypeConverter()
    {
        addConversion([](Type type) { return type; });

        // A TensorType that can be detensorized, is converted to the underlying
        // element type.
        addConversion([](TensorType tensorType) -> Type {
            if (tensorType.getRank() == 0)
                return tensorType.getElementType();

            return tensorType;
        });

        // A tensor value is detensorized by extracting its element(s).
        addTargetMaterialization(
            [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value {
                return builder.create<tensor::ExtractOp>(loc, inputs[0], ValueRange{});
            });

        addSourceMaterialization(sourceMaterializationCallback);
        addArgumentMaterialization(sourceMaterializationCallback);
    }
};

class DetensorizeForOp : public OpConversionPattern<scf::ForOp> {
  public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(scf::ForOp forOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        llvm::errs() << "........................................" << '\n';
        llvm::errs() << forOp << '\n';
        llvm::errs() << "........................................" << '\n';

        // 1. Find scalar tensors and extract element
        SmallVector<std::size_t> indices;
        SmallVector<Value> newIterOperands;
        std::size_t index = 0;
        rewriter.setInsertionPoint(forOp);
        for (OpOperand &opOperand : forOp.getInitArgsMutable()) {
            if (isScalarTensor(opOperand.get())) {
                tensor::ExtractOp extract_op = rewriter.create<tensor::ExtractOp>(
                    forOp->getLoc(), opOperand.get(), ValueRange{});
                newIterOperands.push_back(extract_op->getResult(0));
                indices.push_back(index);
                index += 1;
                continue;
            }
            newIterOperands.push_back(opOperand.get());
            index += 1;
        }

        if (newIterOperands.empty()) {
            return failure();
        }

        // 2. Create the new ForOp shell.
        scf::ForOp newForOp =
            rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(),
                                        forOp.getUpperBound(), forOp.getStep(), newIterOperands);
        newForOp->setAttrs(forOp->getAttrs());
        Block &newBlock = newForOp.getRegion().front();
        SmallVector<Value> newBlockTransferArgs(newBlock.getArguments().begin(),
                                                newBlock.getArguments().end());

        llvm::errs() << "........................................" << '\n';
        llvm::errs() << newForOp << '\n';
        llvm::errs() << "........................................" << '\n';

        // 3. Copy body
        Block &oldBlock = forOp.getRegion().front();
        rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);

        llvm::errs() << "........................................" << '\n';
        llvm::errs() << newForOp << '\n';
        llvm::errs() << "........................................" << '\n';

        // 4. Retensorize arguments at the beginning of the body
        rewriter.setInsertionPoint(&newBlock, newBlock.begin());
        for (std::size_t index : indices) {
            tensor::FromElementsOp from_elem_op = rewriter.create<tensor::FromElementsOp>(
                newForOp->getLoc(),
                RankedTensorType::get({}, newForOp.getRegionIterArg(index).getType()),
                newForOp.getRegionIterArg(index));
            rewriter.replaceUsesWithIf(
                newForOp.getRegionIterArg(index), from_elem_op->getResult(0),
                [&](OpOperand &op) { return !isa<tensor::FromElementsOp>(op.getOwner()); });
        }

        llvm::errs() << "........................................" << '\n';
        llvm::errs() << newForOp << '\n';
        llvm::errs() << "........................................" << '\n';

        // 5. Extract scalar tensor elements and detensorize terminator
        scf::YieldOp clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
        SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
        rewriter.setInsertionPoint(clonedYieldOp);
        for (std::size_t index : indices) {
            tensor::ExtractOp extract_op = rewriter.create<tensor::ExtractOp>(
                clonedYieldOp->getLoc(), clonedYieldOp->getOperand(index), ValueRange{});
            newYieldOperands[index] = extract_op->getResult(0);
        }
        rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
        rewriter.eraseOp(clonedYieldOp);

        llvm::errs() << "........................................" << '\n';
        llvm::errs() << newForOp << '\n';
        llvm::errs() << "........................................" << '\n';

        // 6. Retensorize results after for loop
        rewriter.setInsertionPointAfter(forOp);
        for (std::size_t index : indices) {
            Value for_result = newForOp->getResult(index);
            tensor::FromElementsOp from_elem_op = rewriter.create<tensor::FromElementsOp>(
                newForOp->getLoc(), RankedTensorType::get({}, for_result.getType()), for_result);
            rewriter.replaceUsesWithIf(
                forOp->getResult(index), from_elem_op->getResult(0),
                [&](OpOperand &op) { return !isa<tensor::FromElementsOp>(op.getOwner()); });
        }

        llvm::errs() << "........................................" << '\n';
        llvm::errs() << newForOp << '\n';
        llvm::errs() << "........................................" << '\n';

        rewriter.replaceOp(forOp, newForOp);

        return success();
    }
};

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

    void detensorizeIfOp(scf::IfOp if_op, IRRewriter &rewriter)
    {
        if_op->walk([&](scf::YieldOp yield_op) {
            // Loop over results
            std::size_t i_result = 0;
            for (Value operand : yield_op.getOperands()) {
                if (isScalarTensor(operand)) {
                    auto if_result = if_op->getResult(i_result);
                    // Detensorize operand: extract tensor element before yielding
                    {
                        rewriter.setInsertionPoint(yield_op);
                        Value value = rewriter.create<tensor::ExtractOp>(yield_op->getLoc(),
                                                                         operand, ValueRange{});
                        yield_op.setOperand(i_result, value);
                        rewriter.modifyOpInPlace(if_op,
                                                 [&]() { if_result.setType(value.getType()); });
                    }
                    // Retensorize operand: reconstruct tensor after the for body
                    {
                        rewriter.setInsertionPointAfter(if_op);
                        Value value = rewriter.create<tensor::FromElementsOp>(
                            if_op->getLoc(), RankedTensorType::get({}, if_result.getType()),
                            if_result);
                        rewriter.replaceUsesWithIf(if_result, value, [&](OpOperand &op) {
                            return !isa<tensor::FromElementsOp>(op.getOwner());
                        });
                    }
                }
                i_result += 1;
            }
        });
    }

    void runOnOperation() override
    {
        Operation *root = getOperation();
        IRRewriter rewriter(root->getContext());

        // MLIRContext *context = &getContext();
        // ConversionTarget target(*context);
        // target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        // target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp forOp) {
        //     std::size_t count = 0;
        //     for (OpOperand &opOperand : forOp.getInitArgsMutable()) {
        //         if (isScalarTensor(opOperand.get())) {
        //             count += 1;
        //             continue;
        //         }
        //     }
        //     return !count;
        // });
        // DetensorizeTypeConverter typeConverter;

        // RewritePatternSet patterns(context);
        // patterns.add<DetensorizeForOp>(typeConverter, patterns.getContext());
        // if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
        //     signalPassFailure();
        // llvm::errs() << *root << '\n';

        root->walk([&](scf::ForOp for_op) { detensorizeForOp(for_op, rewriter); });

        // llvm::errs() << *root << '\n';

        root->walk([&](scf::IfOp if_op) { detensorizeIfOp(if_op, rewriter); });
    }
};

std::unique_ptr<Pass> createDetensorizeSCFPass() { return std::make_unique<DetensorizeSCFPass>(); }

} // namespace catalyst
