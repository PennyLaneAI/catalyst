#define DEBUG_TYPE "detensorize-func-boundary"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystDialect.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

static bool isZeroDRankedTensor(Type type)
{
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
        return rankedType.getRank() == 0;
    }
    return false;
}

static Type getScalarType(Type type)
{
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
        if (rankedType.getRank() == 0) {
            return rankedType.getElementType();
        }
    }
    return type;
}

struct DetensorizeFuncPattern : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override
    {
        if (funcOp->hasAttr("llvm.emit_c_interface")) {
            return failure();
        }

        FunctionType funcType = funcOp.getFunctionType();
        SmallVector<Type> newArgTypes;
        SmallVector<Type> newResultTypes;
        bool needsConversion = false;

        for (Type type : funcType.getInputs()) {
            if (isZeroDRankedTensor(type))
                needsConversion = true;
            newArgTypes.push_back(getScalarType(type));
        }
        for (Type type : funcType.getResults()) {
            if (isZeroDRankedTensor(type))
                needsConversion = true;
            newResultTypes.push_back(getScalarType(type));
        }

        if (!needsConversion) {
            return failure();
        }

        // Collect all attributes of the original function
        SmallVector<NamedAttribute> newAttrs;
        for (const NamedAttribute &attr : funcOp->getAttrs()) {
            if (attr.getName() != funcOp.getSymNameAttrName() &&
                attr.getName() != funcOp.getFunctionTypeAttrName()) {
                newAttrs.push_back(attr);
            }
        }

        // Create the new function with the updated signature and preserved attributes
        auto newFuncType = FunctionType::get(getContext(), newArgTypes, newResultTypes);
        auto newFuncOp =
            rewriter.create<func::FuncOp>(funcOp.getLoc(), funcOp.getName(), newFuncType, newAttrs);

        // Create the entry block with the new argument types
        Block *newEntryBlock = newFuncOp.addEntryBlock();
        rewriter.setInsertionPointToStart(newEntryBlock);

        // Map the old block arguments to the new values.
        IRMapping mapper;
        Block &oldEntryBlock = funcOp.front();
        for (unsigned i = 0; i < oldEntryBlock.getNumArguments(); ++i) {
            Value oldArg = oldEntryBlock.getArgument(i);
            Value newArg = newEntryBlock->getArgument(i);

            if (isZeroDRankedTensor(oldArg.getType())) {
                auto fromElementsOp = rewriter.create<tensor::FromElementsOp>(
                    funcOp.getLoc(), oldArg.getType(), newArg);
                mapper.map(oldArg, fromElementsOp.getResult());
            }
            else {
                mapper.map(oldArg, newArg);
            }
        }

        // Clone the operations from the old function's body into the new one
        for (Operation &op : oldEntryBlock.getOperations()) {
            rewriter.clone(op, mapper);
        }

        rewriter.replaceOp(funcOp, newFuncOp.getOperation());
        return success();
    }
};

struct DetensorizeReturnPattern : public OpRewritePattern<func::ReturnOp> {
    using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::ReturnOp returnOp, PatternRewriter &rewriter) const override
    {
        auto funcOp = returnOp->getParentOfType<func::FuncOp>();
        FunctionType funcType = funcOp.getFunctionType();
        bool needsConversion = false;

        for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
            if (isZeroDRankedTensor(returnOp.getOperand(i).getType()) &&
                returnOp.getOperand(i).getType() != funcType.getResult(i)) {
                needsConversion = true;
                break;
            }
        }

        if (!needsConversion) {
            return failure();
        }

        SmallVector<Value> newOperands;
        newOperands.reserve(returnOp.getNumOperands());
        for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
            Value operand = returnOp.getOperand(i);
            if (isZeroDRankedTensor(operand.getType()) &&
                operand.getType() != funcType.getResult(i)) {
                auto extractOp =
                    rewriter.create<tensor::ExtractOp>(returnOp.getLoc(), operand, ValueRange{});
                newOperands.push_back(extractOp.getResult());
            }
            else {
                newOperands.push_back(operand);
            }
        }

        rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, newOperands);
        return success();
    }
};

struct DetensorizeCallPattern : public OpRewritePattern<func::CallOp> {
    using OpRewritePattern<func::CallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::CallOp callOp, PatternRewriter &rewriter) const override
    {
        auto module = callOp->getParentOfType<ModuleOp>();
        auto funcOp = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
        if (!funcOp) {
            return failure();
        }

        FunctionType funcType = funcOp.getFunctionType();
        if (callOp.getCalleeType() == funcType) {
            return failure();
        }

        rewriter.setInsertionPoint(callOp);
        SmallVector<Value> newOperands;
        for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
            Value operand = callOp.getOperand(i);
            if (isZeroDRankedTensor(operand.getType()) &&
                operand.getType() != funcType.getInput(i)) {
                auto extractOp =
                    rewriter.create<tensor::ExtractOp>(callOp.getLoc(), operand, ValueRange{});
                newOperands.push_back(extractOp.getResult());
            }
            else {
                newOperands.push_back(operand);
            }
        }

        auto newCallOp = rewriter.create<func::CallOp>(callOp.getLoc(), funcOp, newOperands);

        SmallVector<Value> newResults;
        for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
            Value oldResult = callOp.getResult(i);
            Value newResult = newCallOp.getResult(i);
            if (isZeroDRankedTensor(oldResult.getType()) &&
                oldResult.getType() != newResult.getType()) {
                auto fromElementsOp = rewriter.create<tensor::FromElementsOp>(
                    callOp.getLoc(), oldResult.getType(), newResult);
                newResults.push_back(fromElementsOp.getResult());
            }
            else {
                newResults.push_back(newResult);
            }
        }

        rewriter.replaceOp(callOp, newResults);
        return success();
    }
};

struct FoldExtractFromElementsPattern : public OpRewritePattern<tensor::ExtractOp> {
    using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                  PatternRewriter &rewriter) const override
    {
        auto fromElementsOp = extractOp.getTensor().getDefiningOp<tensor::FromElementsOp>();
        if (!fromElementsOp) {
            return failure();
        }

        if (fromElementsOp.getElements().size() == 1 && extractOp.getIndices().empty()) {
            rewriter.replaceOp(extractOp, fromElementsOp.getElements()[0]);
            return success();
        }

        return failure();
    }
};

struct FoldFromElementsExtractPattern : public OpRewritePattern<tensor::FromElementsOp> {
    using OpRewritePattern<tensor::FromElementsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::FromElementsOp fromElementsOp,
                                  PatternRewriter &rewriter) const override
    {
        if (fromElementsOp.getElements().size() != 1) {
            return failure();
        }

        auto extractOp = fromElementsOp.getElements()[0].getDefiningOp<tensor::ExtractOp>();
        if (!extractOp) {
            return failure();
        }

        if (extractOp.getTensor().getType() == fromElementsOp.getType()) {
            rewriter.replaceOp(fromElementsOp, extractOp.getTensor());
            return success();
        }

        return failure();
    }
};

namespace catalyst {
#define GEN_PASS_DEF_DETENSORIZEFUNCTIONBOUNDARYPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetensorizeFunctionBoundaryPass
    : public impl::DetensorizeFunctionBoundaryPassBase<DetensorizeFunctionBoundaryPass> {
    using impl::DetensorizeFunctionBoundaryPassBase<
        DetensorizeFunctionBoundaryPass>::DetensorizeFunctionBoundaryPassBase;
    void runOnOperation() override
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);

        patterns.add<DetensorizeFuncPattern, DetensorizeReturnPattern, DetensorizeCallPattern,
                     FoldExtractFromElementsPattern, FoldFromElementsExtractPattern>(context);

        GreedyRewriteConfig config;
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDetensorizeFunctionBoundaryPass()
{
    return std::make_unique<DetensorizeFunctionBoundaryPass>();
}

} // namespace catalyst
