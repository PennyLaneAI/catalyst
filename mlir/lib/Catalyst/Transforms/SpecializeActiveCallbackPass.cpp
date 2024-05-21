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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

LLVM::LLVMFuncOp lookupOrDeclareInactiveCallback(ActiveCallbackOp op, PatternRewriter &rewriter)
{
    std::string name = "inactive_callback";
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto ctx = rewriter.getContext();
    Type i64 = rewriter.getI64Type();

    bool isVarArg = true;
    Type voidType = LLVM::LLVMVoidType::get(ctx);
    LLVM::LLVMFuncOp inactiveCallback = mlir::LLVM::lookupOrCreateFn(
        moduleOp, name, {/*args=*/i64, i64, i64}, /*ret_type=*/voidType, isVarArg);
    return inactiveCallback;
}

LLVM::LLVMFuncOp lookupOrCreateSpecialized(ActiveCallbackOp op, PatternRewriter &rewriter)
{
    auto name = getSpecializedName(op);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto ctx = rewriter.getContext();
    Location loc = op.getLoc();
    SmallVector<Type> inputs;
    Type voidType = LLVM::LLVMVoidType::get(ctx);
    Type ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto argcLen = op.getInputs().size();
    auto retcLen = op.getResults().size();
    auto len = argcLen + retcLen;
    for (size_t i = 0; i < len; i++) {
        inputs.push_back(ptrTy);
    }

    auto funcOp = mlir::LLVM::lookupOrCreateFn(moduleOp, name, inputs, voidType, false);
    auto alreadyDefined = funcOp.getBody().begin() != funcOp.getBody().end();
    if (alreadyDefined) {
        return funcOp;
    }
    funcOp.setPrivate();

    Block *entryBlock = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    auto identAttr = op.getIdentifier();
    auto ident = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(identAttr));

    auto argcAttr = op.getNumberOriginalArg();
    long argcint = argcAttr ? argcAttr.value() : 0;
    auto argc = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(argcint));

    auto resultsSizeAttr = len - argcint;
    auto resultsSizeVal =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(resultsSizeAttr));

    SmallVector<Value> args = {ident, argc, resultsSizeVal};
    for (auto arg : entryBlock->getArguments()) {
        args.push_back(arg);
    }

    auto inactive = lookupOrDeclareInactiveCallback(op, rewriter);
    rewriter.create<LLVM::CallOp>(loc, inactive, args);
    SmallVector<Value> returns;
    rewriter.create<LLVM::ReturnOp>(loc, returns);

    return funcOp;
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
    lookupOrCreateSpecialized(op, rewriter);
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
