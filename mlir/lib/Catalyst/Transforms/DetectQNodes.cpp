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

static constexpr llvm::StringRef scheduleInvoke = "catalyst.preInvoke";
static constexpr llvm::StringRef personalityName = "__gxx_personality_v0";

bool hasQnodeAttribute(LLVM::LLVMFuncOp callOp) { return callOp->hasAttr("qnode"); }

LLVM::LLVMFuncOp getCaller(LLVM::CallOp callOp)
{
    return callOp->getParentOfType<LLVM::LLVMFuncOp>();
}

LLVM::LLVMFuncOp lookupOrCreatePersonality(ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    bool isVarArg = true;
    return mlir::LLVM::lookupOrCreateFn(moduleOp, personalityName, {}, i32Ty, isVarArg);
}

LLVM::LLVMFuncOp lookupOrCreateAbort(ModuleOp moduleOp)
{
    static constexpr llvm::StringRef abortName = "abort";
    MLIRContext *ctx = moduleOp.getContext();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(moduleOp, abortName, {}, voidTy);
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

std::tuple<Block *, Block *, Block *> getBlocks(LLVM::CallOp callOp, PatternRewriter &rewriter)
{
    // TODO: Maybe split this logic a bit?
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    Block *blockContainingCall = callOp->getBlock();
    rewriter.setInsertionPointAfter(callOp);
    Block *successBlock = rewriter.splitBlock(blockContainingCall, rewriter.getInsertionPoint());

    rewriter.setInsertionPoint(callOp);
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto nullOp = rewriter.create<LLVM::NullOp>(callOp.getLoc(), ptrTy);
    Block *unwindBlock = rewriter.createBlock(successBlock);

    rewriter.setInsertionPointToEnd(unwindBlock);
    bool isCleanUp = false;
    SmallVector<Value> operands{nullOp.getResult()};
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto structTy = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {ptrTy, i32Ty});
    rewriter.create<LLVM::LandingpadOp>(callOp.getLoc(), structTy, isCleanUp, operands);
    rewriter.create<LLVM::UnreachableOp>(callOp.getLoc());

    return std::tuple<Block *, Block *, Block *>(blockContainingCall, successBlock, unwindBlock);
}

void setPersonalityAttribute(LLVM::LLVMFuncOp callerOp, LLVM::LLVMFuncOp personality,
                             PatternRewriter &rewriter)
{
    rewriter.updateRootInPlace(callerOp, [&] {
        auto personalityAttr = FlatSymbolRefAttr::get(personality.getSymNameAttr());
        callerOp.setPersonalityAttr(personalityAttr);
    });
}

void transformCallToInvoke(LLVM::CallOp callOp, Block *successBlock, Block *failBlock,
                           PatternRewriter &rewriter)
{
    auto calleeAttr = callOp.getCalleeAttr();
    SmallVector<Value> unwindArgs;
    auto invokeOp = rewriter.create<LLVM::InvokeOp>(callOp.getLoc(), callOp.getResultTypes(),
                                                    calleeAttr, callOp.getOperands(), successBlock,
                                                    ValueRange(), failBlock, unwindArgs);
    rewriter.replaceOp(callOp, invokeOp);
}

struct DetectQnodeTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

LogicalResult DetectQnodeTransform::match(LLVM::CallOp callOp) const
{
    // Only match with direct calls to qnodes.
    // TODO: This should actually be about async.
    // Right now we use the `qnode` attribute to determine async regions.
    // But that might not be the case in the future,
    // So, change this whenever we no longer create async.execute operations based on qnode.
    std::optional<LLVM::LLVMFuncOp> candidate = getCalleeSafe(callOp);
    bool validCandidate = candidate && hasQnodeAttribute(candidate.value());
    return validCandidate ? success() : failure();
}

void DetectQnodeTransform::rewrite(LLVM::CallOp callOp, PatternRewriter &rewriter) const
{
    auto moduleOp = callOp->getParentOfType<ModuleOp>();
    auto personality = lookupOrCreatePersonality(moduleOp);
    auto abortFuncOp = lookupOrCreateAbort(moduleOp);
    auto caller = getCaller(callOp);

    setPersonalityAttribute(caller, personality, rewriter);
    setTransformedAttribute(callOp, rewriter);

    auto [callBlock, successBlock, failBlock] = getBlocks(callOp, rewriter);

    transformCallToInvoke(callOp, successBlock, failBlock, rewriter);
}

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_DETECTQNODEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetectQnodePass : impl::DetectQnodePassBase<DetectQnodePass> {
    using DetectQnodePassBase::DetectQnodePassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<DetectQnodeTransform>(context);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDetectQnodePass() { return std::make_unique<DetectQnodePass>(); }

} // namespace catalyst
