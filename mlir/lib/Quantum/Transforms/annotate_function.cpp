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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"
#include "Quantum/Transforms/annotate_function.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

bool isAnnotated(func::FuncOp op, const char *attr)
{
    return (bool)(op->getAttrOfType<UnitAttr>(attr));
}

bool invalidGradientOperation(func::FuncOp op)
{
    auto res = op.walk([](Operation *o) {
        if (dyn_cast<MeasureOp>(o) || dyn_cast<catalyst::InactiveCallbackOp>(o) ||
            dyn_cast<catalyst::ActiveCallbackOp>(o) || dyn_cast<catalyst::CustomCallOp>(o)) {
            return WalkResult::interrupt();
        }
        else {
            return WalkResult::advance();
        }
    });
    return res.wasInterrupted();
}

bool successfulMatchLeaf(func::FuncOp op)
{
    return !isAnnotated(op, hasInvalidGradientOp) && invalidGradientOperation(op);
}

void annotate(func::FuncOp op, PatternRewriter &rewriter, const char *attr)
{
    op->setAttr(attr, rewriter.getUnitAttr());
}

struct AnnotateFunctionPattern : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;
};

LogicalResult AnnotateFunctionPattern::match(func::FuncOp op) const
{
    return successfulMatchLeaf(op) ? success() : failure();
}

void AnnotateFunctionPattern::rewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
    annotate(op, rewriter, hasInvalidGradientOp);
}

std::optional<func::FuncOp> getFuncOp(const CallGraphNode *node, CallGraph &cg)
{
    std::optional<func::FuncOp> funcOp = std::nullopt;
    if (node == cg.getExternalCallerNode())
        // if we don't know who called us, return nullopt
        return funcOp;
    if (node == cg.getUnknownCalleeNode())
        // if we don't know the callee, return nullopt
        return funcOp;
    auto *callableRegion = node->getCallableRegion();
    funcOp = cast<func::FuncOp>(callableRegion->getParentOp());
    return funcOp;
}

std::optional<func::FuncOp> getCallee(CallGraphNode::Edge edge, CallGraph &cg)
{
    CallGraphNode *callee = edge.getTarget();
    return getFuncOp(callee, cg);
}

bool anyCalleeIsAnnotated(func::FuncOp op, const char *attr, CallGraph &cg)
{
    Region &region = op.getRegion();
    CallGraphNode *node = cg.lookupNode(&region);
    assert(node && "An incorrect region was used to look up a node in the callgraph.");
    for (auto i = node->begin(), e = node->end(); i != e; ++i) {
        std::optional<func::FuncOp> maybeCallee = getCallee(*i, cg);
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

        func::FuncOp calleeOp = maybeCallee.value();
        if (isAnnotated(calleeOp, attr))
            return true;
    }
    return false;
}

bool successfulMatchNode(func::FuncOp op, const char *attr, CallGraph &cg)
{
    return !isAnnotated(op, attr) && anyCalleeIsAnnotated(op, attr, cg);
}

struct PropagateAnnotationPattern : public OpRewritePattern<func::FuncOp> {
    PropagateAnnotationPattern(MLIRContext *ctx, CallGraph &cg)
        : OpRewritePattern<func::FuncOp>(ctx), callgraph(cg)
    {
    }

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;

  private:
    CallGraph &callgraph;
};

LogicalResult PropagateAnnotationPattern::match(func::FuncOp op) const
{
    return successfulMatchNode(op, hasInvalidGradientOp, callgraph) ? success() : failure();
}

void PropagateAnnotationPattern::rewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
    annotate(op, rewriter, hasInvalidGradientOp);
}

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_ANNOTATEFUNCTIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct AnnotateFunctionPass : impl::AnnotateFunctionPassBase<AnnotateFunctionPass> {
    using AnnotateFunctionPassBase::AnnotateFunctionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        CallGraph &cg = getAnalysis<CallGraph>();
        patterns.add<AnnotateFunctionPattern>(context);
        patterns.add<PropagateAnnotationPattern>(context, cg);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createAnnotateFunctionPass()
{
    return std::make_unique<AnnotateFunctionPass>();
}

} // namespace catalyst
