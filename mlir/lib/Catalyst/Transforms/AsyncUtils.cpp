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

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "Catalyst/Transforms/AsyncUtils.h"

using namespace mlir;

// Constants
namespace AsyncUtilsConstants {

static constexpr llvm::StringRef qnodeAttr = "qnode";
static constexpr llvm::StringRef abortName = "abort";
static constexpr llvm::StringRef putsName = "puts";
static constexpr llvm::StringRef unrecoverableErrorName =
    "__catalyst__host__rt__unrecoverable_error";
static constexpr llvm::StringRef scheduleInvokeAttr = "catalyst.preInvoke";
static constexpr llvm::StringRef changeToUnreachable = "catalyst.unreachable";
static constexpr llvm::StringRef personalityName = "__gxx_personality_v0";
static constexpr llvm::StringRef passthroughAttr = "passthrough";
static constexpr llvm::StringRef sinkAttr = "catalyst.sink";
static constexpr llvm::StringRef preHandleErrorAttrValue = "catalyst.preHandleError";
static constexpr llvm::StringRef sourceOfRefCounts = "catalyst.sourceOfRefCounts";
static constexpr llvm::StringRef mlirAsyncRuntimeCreateValueName = "mlirAsyncRuntimeCreateValue";
static constexpr llvm::StringRef mlirAsyncRuntimeCreateTokenName = "mlirAsyncRuntimeCreateToken";
static constexpr llvm::StringRef mlirAsyncRuntimeSetValueErrorName =
    "mlirAsyncRuntimeSetValueError";
static constexpr llvm::StringRef mlirAsyncRuntimeSetTokenErrorName =
    "mlirAsyncRuntimeSetTokenError";
static constexpr llvm::StringRef mlirAsyncRuntimeIsTokenErrorName = "mlirAsyncRuntimeIsTokenError";
static constexpr llvm::StringRef mlirAsyncRuntimeIsValueErrorName = "mlirAsyncRuntimeIsValueError";
static constexpr llvm::StringRef mlirAsyncRuntimeAwaitTokenName = "mlirAsyncRuntimeAwaitToken";
static constexpr llvm::StringRef mlirAsyncRuntimeAwaitValueName = "mlirAsyncRuntimeAwaitValue";
static constexpr llvm::StringRef mlirAsyncRuntimeDropRefName = "mlirAsyncRuntimeDropRef";

}; // namespace AsyncUtilsConstants

bool AsyncUtils::hasChangeToUnreachableAttr(mlir::Operation *op)
{
    return op->hasAttr(AsyncUtilsConstants::changeToUnreachable);
}

bool AsyncUtils::isSink(mlir::Operation *op) { return op->hasAttr(AsyncUtilsConstants::sinkAttr); }

void AsyncUtils::cleanupSink(LLVM::CallOp op, PatternRewriter &rewriter)
{
    rewriter.modifyOpInPlace(op, [&] { op->removeAttr(AsyncUtilsConstants::sinkAttr); });
}

void AsyncUtils::cleanupSource(LLVM::CallOp source, PatternRewriter &rewriter)
{
    rewriter.modifyOpInPlace(source,
                             [&] { source->removeAttr(AsyncUtilsConstants::sourceOfRefCounts); });
}

void AsyncUtils::cleanupSource(SmallVector<LLVM::CallOp> &sources, PatternRewriter &rewriter)
{
    for (auto source : sources) {
        AsyncUtils::cleanupSource(source, rewriter);
    }
}

bool AsyncUtils::hasAbortInBlock(Block *block)
{
    bool returnVal = false;
    block->walk([&](LLVM::CallOp op) { returnVal |= AsyncUtils::callsAbort(op); });
    return returnVal;
}

bool AsyncUtils::hasPutsInBlock(Block *block)
{
    bool returnVal = false;
    block->walk([&](LLVM::CallOp op) { returnVal |= AsyncUtils::callsPuts(op); });
    return returnVal;
}

void AsyncUtils::annotateCallForSource(LLVM::CallOp callOp, PatternRewriter &rewriter)
{
    rewriter.modifyOpInPlace(callOp, [&] {
        callOp->setAttr(AsyncUtilsConstants::sourceOfRefCounts, rewriter.getUnitAttr());
    });
}

void AsyncUtils::annotateBrToUnreachable(LLVM::BrOp brOp, PatternRewriter &rewriter)
{
    rewriter.modifyOpInPlace(brOp, [&] {
        brOp->setAttr(AsyncUtilsConstants::changeToUnreachable, rewriter.getUnitAttr());
    });
}

void AsyncUtils::annotateCallsForSink(SmallVector<LLVM::CallOp> &calls, PatternRewriter &rewriter)
{
    for (auto call : calls) {
        AsyncUtils::scheduleLivenessAnalysis(call, rewriter);
    }
}

bool AsyncUtils::hasQnodeAttribute(LLVM::LLVMFuncOp funcOp)
{
    return funcOp->hasAttr(AsyncUtilsConstants::qnodeAttr);
}

bool AsyncUtils::hasPreHandleErrorAttr(LLVM::LLVMFuncOp funcOp)
{
    return funcOp->hasAttr(AsyncUtilsConstants::preHandleErrorAttrValue);
}

bool AsyncUtils::isScheduledForTransformation(LLVM::CallOp callOp)
{
    return callOp->hasAttr(AsyncUtilsConstants::scheduleInvokeAttr);
}

LLVM::LLVMFuncOp AsyncUtils::getCaller(LLVM::CallOp callOp)
{
    return callOp->getParentOfType<LLVM::LLVMFuncOp>();
}

LLVM::LLVMFuncOp AsyncUtils::lookupOrCreatePersonality(OpBuilder &b, ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    auto i32Ty = IntegerType::get(ctx, 32);
    bool isVarArg = true;
    return mlir::LLVM::lookupOrCreateFn(b, moduleOp, AsyncUtilsConstants::personalityName, {},
                                        i32Ty, isVarArg)
        .value();
}

LLVM::LLVMFuncOp AsyncUtils::lookupOrCreateAbort(OpBuilder &b, ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(b, moduleOp, AsyncUtilsConstants::abortName, {}, voidTy)
        .value();
}

LLVM::LLVMFuncOp AsyncUtils::lookupOrCreateAwaitTokenName(OpBuilder &b, ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(moduleOp.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(
               b, moduleOp, AsyncUtilsConstants::mlirAsyncRuntimeAwaitTokenName, {ptrTy}, voidTy)
        .value();
}

LLVM::LLVMFuncOp AsyncUtils::lookupOrCreateAwaitValueName(OpBuilder &b, ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(moduleOp.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(
               b, moduleOp, AsyncUtilsConstants::mlirAsyncRuntimeAwaitValueName, {ptrTy}, voidTy)
        .value();
}

LLVM::LLVMFuncOp AsyncUtils::lookupOrCreateDropRef(OpBuilder &b, ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(moduleOp.getContext());
    Type llvmInt64Type = IntegerType::get(moduleOp.getContext(), 64);
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(b, moduleOp,
                                        AsyncUtilsConstants::mlirAsyncRuntimeDropRefName,
                                        {ptrTy, llvmInt64Type}, voidTy)
        .value();
}

LLVM::LLVMFuncOp AsyncUtils::lookupOrCreateMlirAsyncRuntimeSetValueError(OpBuilder &b,
                                                                         ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(moduleOp.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(
               b, moduleOp, AsyncUtilsConstants::mlirAsyncRuntimeSetValueErrorName, {ptrTy}, voidTy)
        .value();
}

LLVM::LLVMFuncOp AsyncUtils::lookupOrCreateMlirAsyncRuntimeSetTokenError(OpBuilder &b,
                                                                         ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(moduleOp.getContext());
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(
               b, moduleOp, AsyncUtilsConstants::mlirAsyncRuntimeSetTokenErrorName, {ptrTy}, voidTy)
        .value();
}

LLVM::LLVMFuncOp AsyncUtils::lookupOrCreateUnrecoverableError(OpBuilder &b, ModuleOp moduleOp)
{
    MLIRContext *ctx = moduleOp.getContext();
    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    return mlir::LLVM::lookupOrCreateFn(b, moduleOp, AsyncUtilsConstants::unrecoverableErrorName,
                                        {}, voidTy)
        .value();
}

std::optional<LLVM::LLVMFuncOp> AsyncUtils::getCalleeSafe(LLVM::CallOp callOp)
{
    std::optional<LLVM::LLVMFuncOp> callee;
    auto calleeAttr = callOp.getCalleeAttr();
    auto caller = AsyncUtils::getCaller(callOp);
    if (!calleeAttr) {
        callee = std::nullopt;
    }
    else {
        callee = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(caller, calleeAttr);
    }
    return callee;
}

bool AsyncUtils::isFunctionNamed(LLVM::LLVMFuncOp funcOp, llvm::StringRef expectedName)
{
    llvm::StringRef observedName = funcOp.getSymName();
    return observedName == expectedName;
}

bool AsyncUtils::isMlirAsyncRuntimeCreateValue(LLVM::LLVMFuncOp funcOp)
{
    return AsyncUtils::isFunctionNamed(funcOp,
                                       AsyncUtilsConstants::mlirAsyncRuntimeCreateValueName);
}

bool AsyncUtils::isMlirAsyncRuntimeCreateToken(LLVM::LLVMFuncOp funcOp)
{
    return AsyncUtils::isFunctionNamed(funcOp,
                                       AsyncUtilsConstants::mlirAsyncRuntimeCreateTokenName);
}

bool AsyncUtils::isMlirAsyncRuntimeIsTokenError(LLVM::LLVMFuncOp funcOp)
{
    return AsyncUtils::isFunctionNamed(funcOp,
                                       AsyncUtilsConstants::mlirAsyncRuntimeIsTokenErrorName);
}

bool AsyncUtils::isMlirAsyncRuntimeIsValueError(LLVM::LLVMFuncOp funcOp)
{
    return AsyncUtils::isFunctionNamed(funcOp,
                                       AsyncUtilsConstants::mlirAsyncRuntimeIsValueErrorName);
}

bool AsyncUtils::callsSource(LLVM::CallOp callOp)
{
    return callOp->hasAttr(AsyncUtilsConstants::sourceOfRefCounts);
}

bool AsyncUtils::callsMlirAsyncRuntimeCreateToken(Operation *possibleCall)
{
    if (!isa<LLVM::CallOp>(possibleCall))
        return false;

    LLVM::CallOp callOp = cast<LLVM::CallOp>(possibleCall);
    return AsyncUtils::callsMlirAsyncRuntimeCreateToken(callOp);
}

bool AsyncUtils::callsMlirAsyncRuntimeCreateToken(LLVM::CallOp callOp)
{
    auto maybeCallee = AsyncUtils::getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return AsyncUtils::isMlirAsyncRuntimeCreateToken(callee);
}

bool AsyncUtils::callsMlirAsyncRuntimeCreateValue(LLVM::CallOp callOp)
{
    auto maybeCallee = AsyncUtils::getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return AsyncUtils::isMlirAsyncRuntimeCreateValue(callee);
}

bool AsyncUtils::callsMlirAsyncRuntimeIsTokenError(LLVM::CallOp callOp)
{
    auto maybeCallee = AsyncUtils::getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return AsyncUtils::isMlirAsyncRuntimeIsTokenError(callee);
}

bool AsyncUtils::callsMlirAsyncRuntimeIsValueError(LLVM::CallOp callOp)
{
    auto maybeCallee = AsyncUtils::getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return AsyncUtils::isMlirAsyncRuntimeIsValueError(callee);
}

bool AsyncUtils::callsMlirAsyncRuntimeIsTokenError(Operation *possibleCall)
{
    if (!isa<LLVM::CallOp>(possibleCall))
        return false;

    LLVM::CallOp callOp = cast<LLVM::CallOp>(possibleCall);
    return AsyncUtils::callsMlirAsyncRuntimeIsTokenError(callOp);
}

bool AsyncUtils::callsMlirAsyncRuntimeIsValueError(Operation *possibleCall)
{
    if (!isa<LLVM::CallOp>(possibleCall))
        return false;

    LLVM::CallOp callOp = cast<LLVM::CallOp>(possibleCall);
    return AsyncUtils::callsMlirAsyncRuntimeIsValueError(callOp);
}

bool AsyncUtils::isAbort(LLVM::LLVMFuncOp funcOp)
{
    return AsyncUtils::isFunctionNamed(funcOp, AsyncUtilsConstants::abortName);
}

bool AsyncUtils::isPuts(LLVM::LLVMFuncOp funcOp)
{
    return AsyncUtils::isFunctionNamed(funcOp, AsyncUtilsConstants::putsName);
}

bool AsyncUtils::callsAbort(LLVM::CallOp callOp)
{
    auto maybeCallee = AsyncUtils::getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return AsyncUtils::isAbort(callee);
}

bool AsyncUtils::callsPuts(LLVM::CallOp callOp)
{
    auto maybeCallee = AsyncUtils::getCalleeSafe(callOp);
    if (!maybeCallee)
        return false;

    auto callee = maybeCallee.value();
    return AsyncUtils::isPuts(callee);
}

bool AsyncUtils::callsAbort(Operation *possibleCall)
{
    if (!isa<LLVM::CallOp>(possibleCall))
        return false;

    LLVM::CallOp callOp = cast<LLVM::CallOp>(possibleCall);
    return callsAbort(callOp);
}

void AsyncUtils::cleanupPreHandleErrorAttr(LLVM::LLVMFuncOp funcOp, PatternRewriter &rewriter)
{
    rewriter.modifyOpInPlace(
        funcOp, [&] { funcOp->removeAttr(AsyncUtilsConstants::preHandleErrorAttrValue); });
}

void AsyncUtils::scheduleAnalysisForErrorHandling(LLVM::LLVMFuncOp funcOp,
                                                  PatternRewriter &rewriter)
{
    rewriter.modifyOpInPlace(funcOp, [&] {
        funcOp->setAttr(AsyncUtilsConstants::preHandleErrorAttrValue, rewriter.getUnitAttr());
    });
}

void AsyncUtils::scheduleLivenessAnalysis(LLVM::CallOp callOp, PatternRewriter &rewriter)
{
    rewriter.modifyOpInPlace(
        callOp, [&] { callOp->setAttr(AsyncUtilsConstants::sinkAttr, rewriter.getUnitAttr()); });
}

void AsyncUtils::scheduleCallToInvoke(LLVM::CallOp callOp, PatternRewriter &rewriter)
{
    rewriter.modifyOpInPlace(callOp, [&] {
        callOp->setAttr(AsyncUtilsConstants::scheduleInvokeAttr, rewriter.getUnitAttr());
    });
}

bool AsyncUtils::isAsync(LLVM::LLVMFuncOp funcOp)
{
    if (!funcOp->hasAttr(AsyncUtilsConstants::passthroughAttr)) {
        return false;
    }

    auto haystack = funcOp->getAttrOfType<ArrayAttr>(AsyncUtilsConstants::passthroughAttr);
    auto needle = StringAttr::get(funcOp.getContext(), "presplitcoroutine");
    auto it = std::find(haystack.begin(), haystack.end(), needle);
    return (it != haystack.end()) ? true : false;
}
