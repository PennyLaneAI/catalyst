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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;

namespace {

static constexpr llvm::StringRef qnodeAttr = "qnode";
static constexpr llvm::StringRef scheduleInvokeAttr = "catalyst.preInvoke";

bool hasQnodeAttribute(LLVM::LLVMFuncOp funcOp) { return funcOp->hasAttr(qnodeAttr); }

bool isAsync(LLVM::LLVMFuncOp funcOp) {
    if (!funcOp->hasAttr("passthrough")) return false;

    auto haystack = funcOp->getAttrOfType<ArrayAttr>("passthrough");
    auto needle = StringAttr::get(funcOp.getContext(), "presplitcoroutine");
    for (auto maybeNeedle : haystack) {
       if (maybeNeedle == needle) return true;
    }

    return false;
}

LLVM::LLVMFuncOp getCaller(LLVM::CallOp callOp)
{
    return callOp->getParentOfType<LLVM::LLVMFuncOp>();
}

std::optional<LLVM::LLVMFuncOp> getCalleeSafe(LLVM::CallOp callOp)
{
    std::optional<LLVM::LLVMFuncOp> callee;
    auto calleeAttr = callOp.getCalleeAttr();
    auto caller = getCaller(callOp);
    if (!calleeAttr) {
        callee = std::nullopt;
    }
    else {
        callee = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(caller, calleeAttr);
    }
    return callee;
}

void scheduleCallToInvoke(LLVM::CallOp callOp, PatternRewriter &rewriter)
{
    rewriter.updateRootInPlace(callOp,
                               [&] { callOp->setAttr(scheduleInvokeAttr, rewriter.getUnitAttr()); });
}


struct DetectCallsInAsyncRegionsTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

LogicalResult DetectCallsInAsyncRegionsTransform::match(LLVM::CallOp callOp) const
{
    std::optional<LLVM::LLVMFuncOp> candidate = getCalleeSafe(callOp);
    if (!candidate) return failure();

    LLVM::LLVMFuncOp callee = candidate.value();
    auto caller = getCaller(callOp);
    bool validCandidate = callee->hasAttr(qnodeAttr) \
			  && !callOp->hasAttr(scheduleInvokeAttr) \
			  && isAsync(caller);
    return validCandidate ? success() : failure();
}

void DetectCallsInAsyncRegionsTransform::rewrite(LLVM::CallOp callOp, PatternRewriter &rewriter) const
{
    scheduleCallToInvoke(callOp, rewriter);
}

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_ANNOTATEASYNCCALLWITHINVOKEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct AnnotateAsyncCallWithInvokePass : impl::AnnotateAsyncCallWithInvokePassBase<AnnotateAsyncCallWithInvokePass> {
    using AnnotateAsyncCallWithInvokePassBase::AnnotateAsyncCallWithInvokePassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<DetectCallsInAsyncRegionsTransform>(context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createAnnotateAsyncCallWithInvokePass() { return std::make_unique<AnnotateAsyncCallWithInvokePass>(); }

} // namespace catalyst
