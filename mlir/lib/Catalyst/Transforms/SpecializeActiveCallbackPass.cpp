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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst;

namespace {

std::string getSpecializedName(ActiveCallbackOp op)
{
    auto id = std::to_string(op.getIdentifier());
    return "active_callback_" + id;
}

struct AddDeclarationToModulePattern : public OpRewritePattern<ActiveCallbackOp> {
    using OpRewritePattern<ActiveCallbackOp>::OpRewritePattern;

    LogicalResult match(ActiveCallbackOp op) const override;
    void rewrite(ActiveCallbackOp op, PatternRewriter &rewriter) const override;
};

LogicalResult AddDeclarationToModulePattern::match(ActiveCallbackOp op) const
{
    return op.getSpecialized() ? failure() : success();
}

void AddDeclarationToModulePattern::rewrite(ActiveCallbackOp op, PatternRewriter &rewriter) const
{
    auto specializedName = getSpecializedName(op);
    auto specializedNameAttr = rewriter.getStringAttr(specializedName);
    rewriter.updateRootInPlace(op, [&] {
        auto specializedSym = FlatSymbolRefAttr::get(specializedNameAttr);
        op.setSpecializedAttr(specializedSym);
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
