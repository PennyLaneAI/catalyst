// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Quantum/IR/QuantumOps.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Gradient/IR/GradientOps.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"
#include "Quantum/Transforms/annotate_function.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

bool isAnnotated(FunctionOpInterface op, const char *attr)
{
    return (bool)(op->getAttrOfType<UnitAttr>(attr));
}

bool invalidGradientOperation(FunctionOpInterface op)
{
    ModuleOp mod = op->getParentOfType<ModuleOp>();
    auto res = op.walk([&](Operation *o) {
        if (isa<MeasureOp>(o) || isa<catalyst::CustomCallOp>(o)) {
            return WalkResult::interrupt();
        }
        else if (auto callbackCall = dyn_cast<catalyst::CallbackCallOp>(o)) {
            bool hasCustomDerivative = false;
            auto callee = callbackCall.getCalleeAttr();
            auto callback =
                SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(callbackCall, callee);
            bool inactive = callback.getResultTypes().empty();
            if (inactive) {
                return WalkResult::advance();
            }
            auto uses = *SymbolTable::getSymbolUses(callback, mod);
            for (SymbolTable::SymbolUse use : uses) {
                Operation *user = use.getUser();
                if (auto customGrad = dyn_cast<catalyst::gradient::CustomGradOp>(user)) {
                    hasCustomDerivative |= customGrad.getCalleeAttr() == callee;
                }
                if (hasCustomDerivative)
                    break;
            }
            if (!hasCustomDerivative) {
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return res.wasInterrupted();
}

bool successfulMatchLeaf(FunctionOpInterface op)
{
    return !isAnnotated(op, hasInvalidGradientOp) && invalidGradientOperation(op);
}

void annotate(FunctionOpInterface op, PatternRewriter &rewriter, const char *attr)
{
    op->setAttr(attr, rewriter.getUnitAttr());
}

struct AnnotateFunctionPattern : public OpInterfaceRewritePattern<FunctionOpInterface> {
    using OpInterfaceRewritePattern<FunctionOpInterface>::OpInterfaceRewritePattern;

    LogicalResult matchAndRewrite(FunctionOpInterface op, PatternRewriter &rewriter) const override;
};

LogicalResult AnnotateFunctionPattern::matchAndRewrite(FunctionOpInterface op,
                                                       PatternRewriter &rewriter) const
{
    if (!successfulMatchLeaf(op)) {
        return failure();
    }

    annotate(op, rewriter, hasInvalidGradientOp);
    return success();
}

std::optional<FunctionOpInterface> getFuncOp(const CallGraphNode *node, CallGraph &cg)
{
    std::optional<FunctionOpInterface> funcOp = std::nullopt;
    if (node == cg.getExternalCallerNode())
        // if we don't know who called us, return nullopt
        return funcOp;
    if (node == cg.getUnknownCalleeNode())
        // if we don't know the callee, return nullopt
        return funcOp;
    auto *callableRegion = node->getCallableRegion();
    funcOp = cast<FunctionOpInterface>(callableRegion->getParentOp());
    return funcOp;
}

std::optional<FunctionOpInterface> getCallee(CallGraphNode::Edge edge, CallGraph &cg)
{
    CallGraphNode *callee = edge.getTarget();
    return getFuncOp(callee, cg);
}

bool anyCalleeIsAnnotated(FunctionOpInterface op, const char *attr, CallGraph &cg)
{
    Region &region = op->getRegion(0);
    CallGraphNode *node = cg.lookupNode(&region);
    assert(node && "An incorrect region was used to look up a node in the callgraph.");
    for (auto i = node->begin(), e = node->end(); i != e; ++i) {
        std::optional<FunctionOpInterface> maybeCallee = getCallee(*i, cg);
        // An indirect call.
        // This will not happen in the current version of Catalyst as all calls are direct.
        // Which calls would be indirect?
        // Those which are defined via a pointer address. E.g.,
        // auto func = &foo;
        // or virtual functions.
        // If we don't know which function we are calling, it is safest to assume that the function
        // may be annotated.
        // We can get better precision by using one of the many callgraph analyses.
        // See Sundaresan, Vijay, et al. "Practical virtual method call resolution for Java." ACM
        // SIGPLAN Notices 35.10 (2000): 264-280.
        if (!maybeCallee)
            return true;

        FunctionOpInterface calleeOp = maybeCallee.value();
        if (isAnnotated(calleeOp, attr))
            return true;
    }
    return false;
}

bool successfulMatchNode(FunctionOpInterface op, const char *attr, CallGraph &cg)
{
    return !isAnnotated(op, attr) && anyCalleeIsAnnotated(op, attr, cg);
}

struct PropagateAnnotationPattern : public OpInterfaceRewritePattern<FunctionOpInterface> {
    PropagateAnnotationPattern(MLIRContext *ctx, CallGraph &cg)
        : OpInterfaceRewritePattern<FunctionOpInterface>(ctx), callgraph(cg)
    {
    }

    LogicalResult matchAndRewrite(FunctionOpInterface op, PatternRewriter &rewriter) const override;

  private:
    CallGraph &callgraph;
};

LogicalResult PropagateAnnotationPattern::matchAndRewrite(FunctionOpInterface op,
                                                          PatternRewriter &rewriter) const
{
    if (!successfulMatchNode(op, hasInvalidGradientOp, callgraph)) {
        return failure();
    }

    annotate(op, rewriter, hasInvalidGradientOp);
    return success();
}

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_ANNOTATEFUNCTIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct AnnotateFunctionPassVerified
    : public PassWrapper<AnnotateFunctionPassVerified, OperationPass<>> {
    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        CallGraph &cg = getAnalysis<CallGraph>();
        patterns.add<AnnotateFunctionPattern>(context);
        patterns.add<PropagateAnnotationPattern>(context, cg);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

struct AnnotateFunctionPass : impl::AnnotateFunctionPassBase<AnnotateFunctionPass> {
    using AnnotateFunctionPassBase::AnnotateFunctionPassBase;

    void runOnOperation() final
    {
        MLIRContext *ctx = &getContext();
        auto pm = mlir::PassManager::on<mlir::ModuleOp>(ctx);
        pm.addPass(std::make_unique<AnnotateFunctionPassVerified>());
        pm.enableVerifier(true);
        if (failed(pm.run(getOperation()))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createAnnotateFunctionPass()
{
    return std::make_unique<AnnotateFunctionPass>();
}

} // namespace catalyst
