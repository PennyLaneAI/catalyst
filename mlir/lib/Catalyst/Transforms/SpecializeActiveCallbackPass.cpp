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

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;

namespace {

struct AddDeclarationToModulePattern : public OpRewritePattern<catalyst::ActiveCallbackOp> {
    using OpRewritePattern<catalyst::ActiveCallbackOp>::OpRewritePattern;

    LogicalResult match(catalyst::ActiveCallbackOp op) const override;
    void rewrite(catalyst::ActiveCallbackOp op, PatternRewriter &rewriter) const override;
};

LogicalResult AddDeclarationToModulePattern::match(catalyst::ActiveCallbackOp op) const
{
    return op.getSpecialized() ? failure() : success();
}

void AddDeclarationToModulePattern::rewrite(catalyst::ActiveCallbackOp op,
                                            PatternRewriter &rewriter) const
{
    rewriter.updateRootInPlace(op, [&] {
        auto specializedFakeName = rewriter.getStringAttr("hello");
        auto specializedFake = FlatSymbolRefAttr::get(specializedFakeName);
        op.setSpecializedAttr(specializedFake);
    });
    return;
}

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_SPECIALIZEACTIVECALLBACKPASS
#define GEN_PASS_DECL_SPECIALIZEACTIVECALLBACKPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct SpecializeActiveCallbackPass
    : impl::SpecializeActiveCallbackPassBase<SpecializeActiveCallbackPass> {
    using SpecializeActiveCallbackPassBase::SpecializeActiveCallbackPassBase;
    void runOnOperation() final
    {
        auto ctx = &getContext();
        RewritePatternSet patterns(ctx);
        patterns.add<AddDeclarationToModulePattern>(ctx);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createSpecializeActiveCallbackPass()
{
    return std::make_unique<SpecializeActiveCallbackPass>();
}
} // namespace catalyst
