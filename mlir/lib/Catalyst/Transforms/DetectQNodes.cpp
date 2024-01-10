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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;

namespace {

static constexpr llvm::StringRef qnodeAttr = "qnode";
static constexpr llvm::StringRef abortName = "abort";
static constexpr llvm::StringRef unrecoverableErrorName =
    "__catalyst__host__rt__unrecoverable_error";
static constexpr llvm::StringRef scheduleInvokeAttr = "catalyst.preInvoke";
static constexpr llvm::StringRef personalityName = "__gxx_personality_v0";
static constexpr llvm::StringRef passthroughAttr = "passthrough";
static constexpr llvm::StringRef preHandleErrorAttrValue = "catalyst.preHandleError";
static constexpr llvm::StringRef mlirAsyncRuntimeCreateValueName = "mlirAsyncRuntimeCreateValue";
static constexpr llvm::StringRef mlirAsyncRuntimeCreateTokenName = "mlirAsyncRuntimeCreateToken";
static constexpr llvm::StringRef mlirAsyncRuntimeSetValueErrorName =
    "mlirAsyncRuntimeSetValueError";
static constexpr llvm::StringRef mlirAsyncRuntimeSetTokenErrorName =
    "mlirAsyncRuntimeSetTokenError";
static constexpr llvm::StringRef mlirAsyncRuntimeIsTokenErrorName = "mlirAsyncRuntimeIsTokenError";
static constexpr llvm::StringRef mlirAsyncRuntimeIsValueErrorName = "mlirAsyncRuntimeIsValueError";

bool hasQnodeAttribute(LLVM::LLVMFuncOp funcOp) { return funcOp->hasAttr(qnodeAttr); }

bool isScheduledForTransformation(LLVM::CallOp callOp)
{
    return callOp->hasAttr(scheduleInvokeAttr);
}

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

LLVM::LLVMFuncOp lookupOrCreateMlirAsyncRuntimeSetValueError(ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(moduleOp.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(moduleOp, mlirAsyncRuntimeSetValueErrorName, {ptrTy},
                                        voidTy);
}

LLVM::LLVMFuncOp lookupOrCreateMlirAsyncRuntimeSetTokenError(ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(moduleOp.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(moduleOp, mlirAsyncRuntimeSetTokenErrorName, {ptrTy},
                                        voidTy);
}

LLVM::LLVMFuncOp lookupOrCreateUnrecoverableError(ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(moduleOp, unrecoverableErrorName, {}, voidTy);
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

bool isFunctionNamed(LLVM::LLVMFuncOp funcOp, llvm::StringRef expectedName)
{
    llvm::StringRef observedName = funcOp.getSymName();
    return observedName.equals(expectedName);
}

bool isMlirAsyncRuntimeCreateValue(LLVM::LLVMFuncOp funcOp)
{
    return isFunctionNamed(funcOp, mlirAsyncRuntimeCreateValueName);
}

bool isMlirAsyncRuntimeCreateToken(LLVM::LLVMFuncOp funcOp)
{
    return isFunctionNamed(funcOp, mlirAsyncRuntimeCreateTokenName);
}

bool isMlirAsyncRuntimeIsTokenError(LLVM::LLVMFuncOp funcOp)
{
    return isFunctionNamed(funcOp, mlirAsyncRuntimeIsTokenErrorName);
}

bool isMlirAsyncRuntimeIsValueError(LLVM::LLVMFuncOp funcOp)
{
    return isFunctionNamed(funcOp, mlirAsyncRuntimeIsValueErrorName);
}

bool callsMlirAsyncRuntimeIsTokenError(LLVM::CallOp callOp)
{
    auto maybeCallee = getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return isMlirAsyncRuntimeIsTokenError(callee);
}

bool callsMlirAsyncRuntimeIsValueError(LLVM::CallOp callOp)
{
    auto maybeCallee = getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return isMlirAsyncRuntimeIsValueError(callee);
}

bool callsMlirAsyncRuntimeIsTokenError(Operation *possibleCall)
{
    if (!isa<LLVM::CallOp>(possibleCall))
        return false;

    LLVM::CallOp callOp = cast<LLVM::CallOp>(possibleCall);
    return callsMlirAsyncRuntimeIsTokenError(callOp);
}

bool callsMlirAsyncRuntimeIsValueError(Operation *possibleCall)
{
    if (!isa<LLVM::CallOp>(possibleCall))
        return false;

    LLVM::CallOp callOp = cast<LLVM::CallOp>(possibleCall);
    return callsMlirAsyncRuntimeIsValueError(callOp);
}

bool isAbort(LLVM::LLVMFuncOp funcOp) { return isFunctionNamed(funcOp, abortName); }

bool callsAbort(LLVM::CallOp callOp)
{
    auto maybeCallee = getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return isAbort(callee);
}

bool callsAbort(Operation *possibleCall)
{
    if (!isa<LLVM::CallOp>(possibleCall))
        return false;

    LLVM::CallOp callOp = cast<LLVM::CallOp>(possibleCall);
    return callsAbort(callOp);
}

std::tuple<std::vector<Value>, std::vector<Value>>
collectRefCountedTokensAndValues(LLVM::LLVMFuncOp funcOp)
{
    // Since we are guaranteed to be in an asynchronous execution function
    // we need to gather all values generated from mlirAsyncRuntimeCreateToken
    // and mlirAsyncRuntimeCreateValue.
    // Assumptions: these functions are not called indirectly
    std::vector<Value> collectedTokens;
    std::vector<Value> collectedValues;

    funcOp.walk([&](LLVM::CallOp call) {
        auto calleeMaybeIndirect = getCalleeSafe(call);
        if (!calleeMaybeIndirect)
            return;

        auto callee = calleeMaybeIndirect.value();
        bool tokens = isMlirAsyncRuntimeCreateToken(callee);
        bool values = isMlirAsyncRuntimeCreateValue(callee);
        bool skip = !tokens && !values;
        if (skip)
            return;

        if (tokens) {
            for (auto value : call.getResults()) {
                collectedTokens.push_back(value);
            }
        }

        if (values) {
            for (auto value : call.getResults()) {
                collectedValues.push_back(value);
            }
        }
    });
    return std::tuple<std::vector<Value>, std::vector<Value>>(collectedTokens, collectedValues);
}

void insertCallToMlirAsyncRuntimeErrorFunction(Value value, LLVM::LLVMFuncOp fnDecl,
                                               Block *failBlock, PatternRewriter &rewriter)
{
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToEnd(failBlock);
    SmallVector<Value> operands = {value};
    rewriter.create<LLVM::CallOp>(fnDecl.getLoc(), fnDecl, operands);
}

void insertErrorCalls(std::vector<Value> tokens, std::vector<Value> values, Block *failBlock,
                      PatternRewriter &rewriter)
{
    // At the fail block, it is guaranteed that all runtime values are available
    // but they
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToEnd(failBlock);
    auto landingPad = failBlock->begin();
    auto moduleOp = landingPad->getParentOfType<ModuleOp>();

    LLVM::LLVMFuncOp setTokenError = lookupOrCreateMlirAsyncRuntimeSetTokenError(moduleOp);
    for (auto token : tokens) {
        insertCallToMlirAsyncRuntimeErrorFunction(token, setTokenError, failBlock, rewriter);
    }

    LLVM::LLVMFuncOp setValueError = lookupOrCreateMlirAsyncRuntimeSetValueError(moduleOp);
    for (auto value : values) {
        insertCallToMlirAsyncRuntimeErrorFunction(value, setValueError, failBlock, rewriter);
    }
}

void insertBranchFromFailToSuccess(Block *fail, Block *success, PatternRewriter &rewriter)
{
    // The reason why we are unconditionally jumping from failure to success
    // is because the failure is communicated through the state of the runtime tokens
    // and values.
    //
    // The error will finally be managed by the main thread.
    // But this thread is expected to end normally.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToEnd(fail);

    auto landingPad = fail->begin();
    auto loc = landingPad->getLoc();

    rewriter.create<LLVM::BrOp>(loc, success);
}

struct DetectQnodeTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

struct DetectCallsInAsyncRegionsTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

struct RemoveAbortInsertCallTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

LogicalResult RemoveAbortInsertCallTransform::match(LLVM::CallOp callOp) const
{
    // TODO: Can we find the callers directly without looking at each call op?
    auto maybeCallee = getCalleeSafe(callOp);
    if (!maybeCallee)
        return failure();

    auto calleeFuncOp = maybeCallee.value();
    if (!calleeFuncOp->hasAttr(preHandleErrorAttrValue))
        return failure();

    return success();
}

void cleanupPreHandleErrorAttr(LLVM::LLVMFuncOp funcOp, PatternRewriter &rewriter)
{
    rewriter.updateRootInPlace(funcOp, [&] { funcOp->removeAttr(preHandleErrorAttrValue); });
}

void RemoveAbortInsertCallTransform::rewrite(LLVM::CallOp callOp, PatternRewriter &rewriter) const
{
    auto maybeCallee = getCalleeSafe(callOp);
    if (!maybeCallee)
        return;

    auto moduleOp = callOp->getParentOfType<ModuleOp>();
    auto unrecoverableError = lookupOrCreateUnrecoverableError(moduleOp);

    auto callee = maybeCallee.value();

    // At this moment we are at the call which may return an error.
    // So we need to get the results
    auto results = callOp.getResults();

    // Values to look for:
    SmallVector<Value> valuesToLookFor;
    // TODO: Assert that we have results
    assert(results.size() == 1);

    Value result = results.front();
    Type resultTy = result.getType();

    if (isa<LLVM::LLVMPointerType>(resultTy)) {
        valuesToLookFor.push_back(result);
    }
    else if (isa<LLVM::LLVMStructType>(resultTy)) {
        // How to refer to a value without using llvm.extract
        for (Operation *user : result.getUsers()) {
            if (isa<LLVM::ExtractValueOp>(user)) {
                valuesToLookFor.push_back(user->getResult(0));
            }
        }
    }
    else {
        // TODO: unreachable
    }

    SmallVector<Value> callResults;

    for (Value value : valuesToLookFor) {
        for (Operation *user : value.getUsers()) {
            // Use forward slices to prevent checking individual llvm.extract operations
            bool isCallToIsErrorToken = callsMlirAsyncRuntimeIsTokenError(user);
            bool isCallToIsValueToken = callsMlirAsyncRuntimeIsTokenError(user);
            bool isValid = isCallToIsErrorToken || isCallToIsValueToken;
            if (!isValid)
                continue;

            auto boolVal = user->getResult(0);
            callResults.push_back(boolVal);
        }
    }

    SmallVector<Value> potentialConditions(callResults.begin(), callResults.end());

    for (auto boolVal : callResults) {
        for (Operation *user : boolVal.getUsers()) {
            if (isa<LLVM::XOrOp>(user)) {
                auto xorResult = user->getResult(0);
                potentialConditions.push_back(xorResult);
            }
        }
    }

    SmallVector<Block *> dests;

    for (auto condition : potentialConditions) {
        for (Operation *user : condition.getUsers()) {
            if (isa<LLVM::CondBrOp>(user)) {
                LLVM::CondBrOp brOp = cast<LLVM::CondBrOp>(user);
                dests.push_back(brOp.getTrueDest());
                dests.push_back(brOp.getFalseDest());
            }
        }
    }

    SmallVector<LLVM::CallOp> aborts;

    for (Block *block : dests) {
        block->walk([&](Operation *op) {
            if (callsAbort(op)) {
                LLVM::CallOp abortCall = cast<LLVM::CallOp>(op);
                aborts.push_back(abortCall);
            }
        });
    }

    for (auto abort : aborts) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPoint(abort);
        auto callToHostRuntime =
            rewriter.create<LLVM::CallOp>(abort.getLoc(), unrecoverableError, abort.getOperands());
        rewriter.replaceOp(abort, callToHostRuntime);
    }

    cleanupPreHandleErrorAttr(callee, rewriter);
}

LogicalResult DetectQnodeTransform::match(LLVM::CallOp callOp) const
{
    // Only match with direct calls to qnodes.
    // TODO: This should actually be about async.
    // Right now we use the `qnode` attribute to determine async regions.
    // But that might not be the case in the future,
    // So, change this whenever we no longer create async.execute operations based on qnode.
    std::optional<LLVM::LLVMFuncOp> candidate = getCalleeSafe(callOp);
    bool validCandidate =
        candidate &&
        hasQnodeAttribute(candidate.value()) // This one guarantees that we are in async.
        && isScheduledForTransformation(callOp);
    return validCandidate ? success() : failure();
}

void scheduleAnalysisForErrorHandling(LLVM::LLVMFuncOp funcOp, PatternRewriter &rewriter)
{
    rewriter.updateRootInPlace(
        funcOp, [&] { funcOp->setAttr(preHandleErrorAttrValue, rewriter.getUnitAttr()); });
}

void DetectQnodeTransform::rewrite(LLVM::CallOp callOp, PatternRewriter &rewriter) const
{
    auto moduleOp = callOp->getParentOfType<ModuleOp>();
    auto personality = lookupOrCreatePersonality(moduleOp);
    auto abortFuncOp = lookupOrCreateAbort(moduleOp);
    auto caller = getCaller(callOp);

    setPersonalityAttribute(caller, personality, rewriter);

    auto [callBlock, successBlock, failBlock] = getBlocks(callOp, rewriter);

    transformCallToInvoke(callOp, successBlock, failBlock, rewriter);

    auto [tokens, values] = collectRefCountedTokensAndValues(caller);

    insertErrorCalls(tokens, values, failBlock, rewriter);
    insertBranchFromFailToSuccess(failBlock, successBlock, rewriter);
    scheduleAnalysisForErrorHandling(caller, rewriter);
}

void scheduleCallToInvoke(LLVM::CallOp callOp, PatternRewriter &rewriter)
{
    rewriter.updateRootInPlace(
        callOp, [&] { callOp->setAttr(scheduleInvokeAttr, rewriter.getUnitAttr()); });
}

bool isAsync(LLVM::LLVMFuncOp funcOp)
{
    if (!funcOp->hasAttr(passthroughAttr))
        return false;

    auto haystack = funcOp->getAttrOfType<ArrayAttr>(passthroughAttr);
    auto needle = StringAttr::get(funcOp.getContext(), "presplitcoroutine");
    for (auto maybeNeedle : haystack) {
        if (maybeNeedle == needle)
            return true;
    }

    return false;
}

LogicalResult DetectCallsInAsyncRegionsTransform::match(LLVM::CallOp callOp) const
{
    std::optional<LLVM::LLVMFuncOp> candidate = getCalleeSafe(callOp);
    if (!candidate)
        return failure();

    LLVM::LLVMFuncOp callee = candidate.value();
    auto caller = getCaller(callOp);
    bool validCandidate =
        callee->hasAttr(qnodeAttr) && !callOp->hasAttr(scheduleInvokeAttr) && isAsync(caller);
    return validCandidate ? success() : failure();
}

void DetectCallsInAsyncRegionsTransform::rewrite(LLVM::CallOp callOp,
                                                 PatternRewriter &rewriter) const
{
    scheduleCallToInvoke(callOp, rewriter);
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

        RewritePatternSet patterns1(context);
        patterns1.add<DetectCallsInAsyncRegionsTransform>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns1)))) {
            signalPassFailure();
        }

        RewritePatternSet patterns2(context);
        patterns2.add<DetectQnodeTransform>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns2)))) {
            signalPassFailure();
        }

        RewritePatternSet patterns3(context);
        patterns3.add<RemoveAbortInsertCallTransform>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns3)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDetectQnodePass() { return std::make_unique<DetectQnodePass>(); }

} // namespace catalyst
