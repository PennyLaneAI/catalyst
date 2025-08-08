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

bool hasScalarTensorSignature(func::FuncOp funcOp)
{
    for (Type type : funcOp.getFunctionType().getInputs()) {
        if (isScalarTensor(type)) {
            return true;
        }
    }
    for (Type type : funcOp.getFunctionType().getResults()) {
        if (isScalarTensor(type)) {
            return true;
        }
    }
    return false;
}

struct DetensorizeCallSitePattern : public OpRewritePattern<func::CallOp> {
    using OpRewritePattern<func::CallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::CallOp callOp, PatternRewriter &rewriter) const override
    {

        auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callOp, callOp.getCalleeAttr());
        auto module = callOp->getParentOfType<ModuleOp>();

        if (!funcOp || funcOp->hasAttr("llvm.emit_c_interface") ||
            !hasScalarTensorSignature(funcOp) || funcOp->hasAttr("qnode")) {
            return failure();
        }

        // Create detensorized FuncOp if it does not already exist
        std::string newFuncName = funcOp.getName().str() + ".detensorized";
        auto newFuncOp = module.lookupSymbol<func::FuncOp>(newFuncName);

        if (!newFuncOp) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToEnd(module.getBody());

            // Create the new function with a detensorized signature.
            FunctionType funcType = funcOp.getFunctionType();
            SmallVector<Type> newArgTypes, newResultTypes;
            for (Type type : funcType.getInputs()) {
                newArgTypes.push_back(getScalarOrOriginalType(type));
            }
            for (Type type : funcType.getResults()) {
                newResultTypes.push_back(getScalarOrOriginalType(type));
            }
            auto newFuncType = FunctionType::get(getContext(), newArgTypes, newResultTypes);

            // Collect all attributes from the original function
            SmallVector<NamedAttribute> newAttrs;
            for (const NamedAttribute &attr : funcOp->getAttrs()) {
                if (attr.getName() == funcOp.getSymNameAttrName() ||
                    attr.getName() == funcOp.getFunctionTypeAttrName()) {
                    continue;
                }
                newAttrs.push_back(attr);
            }

            // Create the new function, passing the collected attributes
            newFuncOp = rewriter.create<func::FuncOp>(funcOp.getLoc(), newFuncName, newFuncType, newAttrs);

            Block *newEntryBlock = newFuncOp.addEntryBlock();

            // Map old arguments to new arguments 
            IRMapping mapper;
            rewriter.setInsertionPointToStart(newEntryBlock);
            for (const auto &it : llvm::enumerate(funcOp.getArguments())) {
                Value oldArg = it.value();
                Value newArg = newEntryBlock->getArgument(it.index());

                if (isScalarTensor(oldArg.getType())) {
                    // Insert a FromElementsOp if the old argument is a scalar tensor
                    auto fromElementsOp = rewriter.create<tensor::FromElementsOp>(
                        newArg.getLoc(), oldArg.getType(), newArg);
                    mapper.map(oldArg, fromElementsOp.getResult());
                }
                else {
                    mapper.map(oldArg, newArg);
                }
            }

            // Clone the operations from the body of old function (excluding the old return)
            rewriter.setInsertionPointToEnd(newEntryBlock);
            for (Operation &op : funcOp.front().without_terminator()) {
                rewriter.clone(op, mapper);
            }

            // Create the new return operation
            auto oldReturnOp = cast<func::ReturnOp>(funcOp.front().getTerminator());
            SmallVector<Value> newReturnOperands;
            newReturnOperands.reserve(oldReturnOp.getNumOperands());
            for (Value operand : oldReturnOp.getOperands()) {
                Value newOperand = mapper.lookup(operand);
                if (isScalarTensor(newOperand.getType())) {
                    // Insert ExtractOp if the operand is a scalar tensor
                    auto extractOp = rewriter.create<tensor::ExtractOp>(
                        oldReturnOp.getLoc(), newOperand, ValueRange{});
                    newReturnOperands.push_back(extractOp.getResult());
                }
                else {
                    newReturnOperands.push_back(newOperand);
                }
            }
            rewriter.create<func::ReturnOp>(oldReturnOp.getLoc(), newReturnOperands);
        }

        // Rewrite the original call site to use the new detensorized function.
        rewriter.setInsertionPoint(callOp);
        SmallVector<Value> newOperands;
        for (Value operand : callOp.getOperands()) {
            // Insert ExtractOp if the old operand is a scalar tensor to bridge the detensorized function
            if (isScalarTensor(operand.getType())) {
                auto extractOp =
                    rewriter.create<tensor::ExtractOp>(callOp.getLoc(), operand, ValueRange{});
                newOperands.push_back(extractOp.getResult());
            }
            else {
                newOperands.push_back(operand);
            }
        }

        auto newCallOp = rewriter.create<func::CallOp>(callOp.getLoc(), newFuncOp, newOperands);

        SmallVector<Value> newResults;
        for (size_t i = 0; i < callOp.getNumResults(); ++i) {
            Value oldResult = callOp.getResult(i);
            Value newResult = newCallOp.getResult(i);
            if (isScalarTensor(oldResult.getType())) {
                // Insert a FromElementsOp if the old result is a scalar tensor to bridge the detensorized function
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

        patterns.add<DetensorizeCallSitePattern>(context);

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
