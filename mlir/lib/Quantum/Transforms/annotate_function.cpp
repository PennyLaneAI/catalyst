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

using namespace mlir;
using namespace catalyst::quantum;

namespace {

static constexpr const char *hasMeasureAttrName = "catalyst.invalidGradientOperation";

bool isAnnotated(func::FuncOp op, const char *attr)
{
    return (bool)(op->getAttrOfType<UnitAttr>(attr));
}

bool invalidGradientOperation(func::FuncOp op)
{
    auto res = op.walk([](Operation *o) {
        if (dyn_cast<MeasureOp>(o) || dyn_cast<catalyst::PythonCallOp>(o) ||
            dyn_cast<catalyst::CustomCallOp>(o)) {
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
    return !isAnnotated(op, hasMeasureAttrName) && invalidGradientOperation(op);
}

void annotate(func::FuncOp op, PatternRewriter &rewriter, const char *attr)
{
    op->setAttr(attr, rewriter.getUnitAttr());
}

struct AnnotateFunctionTransform : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;
};

LogicalResult AnnotateFunctionTransform::match(func::FuncOp op) const
{
    return successfulMatchLeaf(op) ? success() : failure();
}

void AnnotateFunctionTransform::rewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
    annotate(op, rewriter, hasMeasureAttrName);
}

std::optional<func::FuncOp> getFuncOp(const CallGraphNode *node, CallGraph &cg)
{
    std::optional<func::FuncOp> funcOp = std::nullopt;
    if (node == cg.getExternalCallerNode())
        return funcOp;
    if (node == cg.getUnknownCalleeNode())
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
    // TODO: ICE if we do not find the node.
    for (auto i = node->begin(), e = node->end(); i != e; ++i) {
        std::optional<func::FuncOp> maybeCallee = getCallee(*i, cg);
        // An indirect call
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

struct PropagateAnnotationTransform : public OpRewritePattern<func::FuncOp> {

    PropagateAnnotationTransform(MLIRContext *ctx, CallGraph &cg)
        : OpRewritePattern<func::FuncOp>(ctx), callgraph(cg)
    {
    }

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;

  private:
    CallGraph &callgraph;
};

LogicalResult PropagateAnnotationTransform::match(func::FuncOp op) const
{
    return successfulMatchNode(op, hasMeasureAttrName, callgraph) ? success() : failure();
}

void PropagateAnnotationTransform::rewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
    annotate(op, rewriter, hasMeasureAttrName);
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
        patterns.add<AnnotateFunctionTransform>(context);
        patterns.add<PropagateAnnotationTransform>(context, cg);

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
