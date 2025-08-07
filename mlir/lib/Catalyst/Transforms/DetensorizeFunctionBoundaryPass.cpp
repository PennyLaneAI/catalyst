#define DEBUG_TYPE "detensorize-func-boundary"

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

namespace {
bool isScalarTensor(Type type)
{
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
        return rankedType.getRank() == 0;
    }
    return false;
}

Type getScalarOrOriginalType(Type type)
{
    if (isScalarTensor(type)) {
        return dyn_cast<RankedTensorType>(type).getElementType();
    }
    else {
        return type;
    }
}

bool hasScalarTensorArguments(func::FuncOp funcOp)
{
    for (Type type : funcOp.getFunctionType().getInputs()) {
        if (isScalarTensor(type)) {
            return true;
        }
    }
    return false;
}

bool hasScalarTensorResults(func::FuncOp funcOp)
{
    for (Type type : funcOp.getFunctionType().getResults()) {
        if (isScalarTensor(type)) {
            return true;
        }
    }
    return false;
}

struct DetensorizeFuncPattern : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::FuncOp funcOp, PatternRewriter &rewriter) const override
    {
        // Skip for main function
        if (funcOp->hasAttr("llvm.emit_c_interface")) {
            return failure();
        }

        // Skip if function does not have scalar tensor arguments or results
        if (!hasScalarTensorArguments(funcOp) && !hasScalarTensorResults(funcOp)) {
            return failure();
        }

        // Skip if not used directly by func call (e.g. gradient VJP or mitigation ZNE)
        auto module = funcOp->getParentOfType<ModuleOp>();
        if (auto uses = mlir::SymbolTable::getSymbolUses(funcOp, module)) {
            for (const mlir::SymbolTable::SymbolUse &use : *uses) {
                if (!isa<func::CallOp>(use.getUser())) {
                    return failure();
                }
            }
        }

        // Skip if used by gradient
        if (funcOp->hasAttr("diff_method")) {
            return failure();
        }

        // Collect the argument and result types
        FunctionType funcType = funcOp.getFunctionType();
        SmallVector<Type> newArgTypes;
        SmallVector<Type> newResultTypes;

        for (Type type : funcType.getInputs()) {
            newArgTypes.push_back(getScalarOrOriginalType(type));
        }
        for (Type type : funcType.getResults()) {
            newResultTypes.push_back(getScalarOrOriginalType(type));
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

        // Map the old block arguments to the new values
        IRMapping mapper;
        Block *oldEntryBlock = &funcOp.front();
        for (unsigned i = 0; i < oldEntryBlock->getNumArguments(); ++i) {
            Value oldArg = oldEntryBlock->getArgument(i);
            Value newArg = newEntryBlock->getArgument(i);

            if (isScalarTensor(oldArg.getType())) {
                // Insert a FromElements op to bridge scalar argument to tensor
                auto fromElementsOp = rewriter.create<tensor::FromElementsOp>(
                    funcOp.getLoc(), oldArg.getType(), newArg);
                mapper.map(oldArg, fromElementsOp.getResult());
            }
            else {
                mapper.map(oldArg, newArg);
            }
        }

        // Clone the operations from the old function's body into the new one
        for (Operation &op : oldEntryBlock->getOperations()) {
            rewriter.clone(op, mapper);
        }

        rewriter.replaceOp(funcOp, newFuncOp.getOperation());
        return success();
    }
};

bool hasScalarTensorOperandAndMismatchFuncResultType(func::ReturnOp returnOp)
{
    auto funcOp = returnOp->getParentOfType<func::FuncOp>();
    FunctionType funcType = funcOp.getFunctionType();

    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        auto returnOperandType = returnOp.getOperand(i).getType();
        auto funcResultType = funcType.getResult(i);
        if (isScalarTensor(returnOperandType) && returnOperandType != funcResultType) {
            return true;
        }
    }
    return false;
}

struct DetensorizeReturnPattern : public OpRewritePattern<func::ReturnOp> {
    using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::ReturnOp returnOp, PatternRewriter &rewriter) const override
    {
        if (!hasScalarTensorOperandAndMismatchFuncResultType(returnOp)) {
            return failure();
        }

        // Create new return operation with detensorized operands
        SmallVector<Value> newOperands;
        newOperands.reserve(returnOp.getNumOperands());
        auto funcOp = returnOp->getParentOfType<func::FuncOp>();
        FunctionType funcType = funcOp.getFunctionType();

        for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
            Value operand = returnOp.getOperand(i);
            if (isScalarTensor(operand.getType()) && operand.getType() != funcType.getResult(i)) {
                // Insert a tensor extract operation to bridge scalar tensor to scalar for return
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
        // Only update the call op if it differs from the function signature
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
            if (isScalarTensor(operand.getType()) && operand.getType() != funcType.getInput(i)) {
                // Insert extract op if the operand is converted from tensor to scalar
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
            if (isScalarTensor(oldResult.getType()) && oldResult.getType() != newResult.getType()) {
                // Insert FromElement op if result is converted from tensor to scalar
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
} // namespace

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

        patterns.add<DetensorizeFuncPattern>(context);
        patterns.add<DetensorizeReturnPattern>(context);
        patterns.add<DetensorizeCallPattern>(context);

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
