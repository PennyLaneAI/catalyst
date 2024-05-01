// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;

namespace {
static constexpr llvm::StringRef debugCallback = "catalyst.debugCallback";
struct AnnotateDebugCallbackPattern : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

LogicalResult AnnotateDebugCallbackPattern::match(LLVM::CallOp callOp) const {
    bool isDebugCallback = callOp->hasAttr(debugCallback);
    return isDebugCallback ? success() : failure();
}


void AnnotateDebugCallbackPattern::rewrite(LLVM::CallOp callOp, PatternRewriter &rewriter) const
{
    rewriter.updateRootInPlace(callOp, [&] { callOp->removeAttr(debugCallback); });
}

}

namespace catalyst {

#define GEN_PASS_DEF_ANNOTATEDEBUGCALLBACKASENZYMECONSTPASS
#define GEN_PASS_DECL_ANNOTATEDEBUGCALLBACKASENZYMECONSTPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct AnnotateDebugCallbackAsEnzymeConstPass : impl::AnnotateDebugCallbackAsEnzymeConstPassBase<AnnotateDebugCallbackAsEnzymeConstPass> {
    using AnnotateDebugCallbackAsEnzymeConstPassBase::AnnotateDebugCallbackAsEnzymeConstPassBase;

    void runOnOperation() final
    {

        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<AnnotateDebugCallbackPattern>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createAnnotateDebugCallbackAsEnzymeConstPass()
{
    return std::make_unique<AnnotateDebugCallbackAsEnzymeConstPass>();
}

} // namespace catalyst
