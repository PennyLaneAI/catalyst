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

#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Transforms/AsyncUtils.h"
#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;

namespace {

void collectValuesToLookFor(ResultRange &results, SmallVector<Value> &valuesToLookFor);
void collectValuesToLookFor(ResultRange &results, SmallVector<Value> &tokens,
                            SmallVector<Value> &values);
void collectResultsForMlirAsyncRuntimeErrorFunctions(SmallVector<Value> &values,
                                                     SmallVector<Value> &results);
void collectPotentialConditions(SmallVector<Value> &values, SmallVector<Value> &conditions);
void collectSuccessorBlocks(SmallVector<Value> &conditions, SmallVector<Block *> &aborts,
                            SmallVector<Block *> &success);
void collectPutsBlocks(SmallVector<Value> &conditions, SmallVector<Block *> &puts);
void collectCallsToAbortInBlocks(SmallVector<Block *> &blocks, SmallVector<LLVM::CallOp> &calls);
void RemoveCallsToPutsInBlocks(SmallVector<Block *> &blocks, PatternRewriter &rewriter);
void replaceCallsWithCallToTarget(SmallVector<LLVM::CallOp> &oldCallOps, LLVM::LLVMFuncOp target,
                                  SmallVector<LLVM::CallOp> &newCalls, PatternRewriter &rewriter);
void replaceTerminatorWithUnconditionalJumpToSuccessBlock(SmallVector<Block *> abortBlocks,
                                                          SmallVector<Block *> successBlocks,
                                                          PatternRewriter &rewriter);

std::tuple<Block *, Block *, Block *> getBlocks(LLVM::CallOp callOp, PatternRewriter &rewriter);
void setPersonalityAttribute(LLVM::LLVMFuncOp callerOp, LLVM::LLVMFuncOp personality,
                             PatternRewriter &rewriter);
LLVM::InvokeOp transformCallToInvoke(LLVM::CallOp callOp, Block *successBlock, Block *failBlock,
                                     PatternRewriter &rewriter);
std::tuple<std::vector<Value>, std::vector<Value>>
collectRefCountedTokensAndValues(LLVM::LLVMFuncOp funcOp);
void insertCallToMlirAsyncRuntimeErrorFunction(Value value, LLVM::LLVMFuncOp fnDecl,
                                               Block *failBlock, PatternRewriter &rewriter);
void insertErrorCalls(std::vector<Value> tokens, std::vector<Value> values, Block *failBlock,
                      PatternRewriter &rewriter);
void insertBranchFromFailToSuccessor(Block *fail, Block *success, PatternRewriter &rewriter);

/*
 * This pass is split into multiple patterns.
 *
 * The first pattern is DetectCallsInAsyncRegionsTransform.
 */

struct DetectCallsInAsyncRegionsTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

/*
 * DetectCallsInAsyncregionsTransform pattern will match
 */
LogicalResult DetectCallsInAsyncRegionsTransform::matchAndRewrite(LLVM::CallOp callOp,
                                                                  PatternRewriter &rewriter) const
{
    // Calls to direct functions
    //    llvm.call @callee() : () -> ()
    std::optional<LLVM::LLVMFuncOp> candidate = AsyncUtils::getCalleeSafe(callOp);
    if (!candidate)
        return failure();

    LLVM::LLVMFuncOp callee = candidate.value();
    // Where the callee is annotated with the qnode attribute
    //    llvm.func @callee() attributes { qnode }
    bool isQnode = AsyncUtils::hasQnodeAttribute(callee);

    /* And are called from an presplitcoroutine context
     *
     *     llvm.func @foo() attributes { passthrough = [presplitcoroutine] } {
     *         llvm.call @callee()
     *     }
     */
    auto caller = AsyncUtils::getCaller(callOp);
    bool validCandidate = isQnode && AsyncUtils::isAsync(caller);

    if (!validCandidate)
        return failure();

    bool hasBeenTransformed = AsyncUtils::isScheduledForTransformation(callOp);
    if (hasBeenTransformed)
        return failure();

    /* Will be transformed to add the attribute catalyst.preInvoke
     *     llvm.call @callee() { catalyst.preInvoke }
     */
    AsyncUtils::scheduleCallToInvoke(callOp, rewriter);
    return success();
}

/* Next, we run AddExceptionHandlingTransform */
struct AddExceptionHandlingTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

/* Here we only match with calls that have the { catalyst.preInvoke } annotations.
 * The reason behind this separation between the previous pattern and this one,
 * is that this pattern can potentially be reused as long as this single annotation is present.
 */
LogicalResult AddExceptionHandlingTransform::match(LLVM::CallOp callOp) const
{
    // The following is a valid match
    //     llvm.call @callee() { catalyst.preInvoke }
    bool validCandidate = AsyncUtils::isScheduledForTransformation(callOp);
    return validCandidate ? success() : failure();
}

void AddExceptionHandlingTransform::rewrite(LLVM::CallOp callOp, PatternRewriter &rewriter) const
{
    auto moduleOp = callOp->getParentOfType<ModuleOp>();
    // Here, we are adding a reference to the personality declaration.
    // From the documentation: https://llvm.org/docs/ExceptionHandling.html#exception-tables
    auto personality = AsyncUtils::lookupOrCreatePersonality(moduleOp);

    // We annotate the body of the function containing the callop to have a reference
    // to the personality.
    //
    //     llvm.func caller() attributes { personality = @__gxx_personality_v0 } {
    //         llvm.call @callee() { catalyst.preInvoke } : () -> ()
    //         llvm.return
    //     }
    auto caller = AsyncUtils::getCaller(callOp);
    setPersonalityAttribute(caller, personality, rewriter);

    // Then, the basic block where the call is located is separated into three.
    // I'll leave the three basic blocks in a broken state (without terminators)
    // but will still show their place in the code.
    //
    //     llvm.func caller() {
    //         llvm.call @callee() : () -> ()  // This is entry basic block.
    //     ^bbfail:
    //         // emptyBlock
    //     ^bbsuccess:
    //         llvm.return
    //     }

    // Right now the logic also adds the following ops which are useful, but maybe
    // we can separate the logic a bit.
    //
    //     llvm.func caller() {
    //         %null = llvm.mlir.null : !llvm.ptr
    //         llvm.call @callee() : () -> ()
    //     ^bbfail:
    //         %0 = llvm.landingpad (catch %null : !llvm.ptr) : !llvm.struct<(ptr<i8>, i32)>
    //     ^bbsuccess:
    //         llvm.return
    //     }
    //
    // The llvm.landingpad must be the first instruction of the failblock.
    // llvm.landingpad when catching %null is equivalent to a catch statement in C++
    // without specifying an exception. The argument can be non-null if one wants
    // to catch specific exceptions.
    auto [callBlock, successBlock, failBlock] = getBlocks(callOp, rewriter);

    // We now transform the call to invoke.
    // Invoke is a version of call that can handle exceptions.
    //
    //     llvm.func caller() {
    //         %null = llvm.mlir.null : !llvm.ptr
    //         llvm.invoke @callee() to ^bbsuccess unwind ^bbfail : () -> ()
    //     ^bbfail:
    //         %0 = llvm.landingpad (catch %null : !llvm.ptr) : !llvm.struct<(ptr<i8>, i32)>
    //     ^bbsuccess:
    //         llvm.return
    //     }
    auto invokeOp = transformCallToInvoke(callOp, successBlock, failBlock, rewriter);

    // Here we collect all values from functions that create runtime token or values.
    // These function calls are made in the same function.
    //
    //     llvm.func caller()
    //         // ...
    //         %0 = llvm.call @mlirAsyncRuntimeCreateToken() : () -> !llvm.ptr
    //         %1 = llvm.call @mlirAsyncRuntimeCreateValue() : () -> !llvm.ptr
    //         // ...
    //
    // These tokens and values are implemented in the MLIRAsyncEngineRuntime.
    // They contain a state which checks whether the result is:
    // Unavailable, Available, or Error.
    auto [tokens, values] = collectRefCountedTokensAndValues(caller);

    // We are setting all of these values to be errors in the fail block.
    //
    //     ^bbfail:
    //         %2 = llvm.landingpad (catch %null : !llvm.ptr) : !llvm.struct<(ptr<i8>, i32)>
    //         llvm.call @mlirAsyncRuntimeSetTokenError(%0) : (!llvm.ptr) -> ()
    //         llvm.call @mlirAsyncRuntimeSetValueError(%1) : (!llvm.ptr) -> ()
    insertErrorCalls(tokens, values, failBlock, rewriter);

    // The failBlock still has no successors.
    // And we need to return these tokens back to the caller so that they
    // can look into whether or not an error has occured.
    // We don't want to jump to the successBlock because the success block
    // will override the state in the vales to Available.
    // so we jump to the first successor.
    //
    // TODO:
    // Maybe a better answer here would be to copy the contents of the success block
    // but remove calls to mlirAsyncRuntimeEmplaceValue/Token which are the only
    // instructions in that block. This is more general, but it looks unnecessary as
    // guaranteed by the generated code for the async lowerings.
    if (successBlock->hasNoSuccessors()) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToEnd(failBlock);
        rewriter.create<LLVM::UnreachableOp>(invokeOp->getLoc());
    }
    else {
        auto successor = successBlock->getSuccessor(0);

        // This is roughly what the function looks like after transformation
        //     llvm.func caller() {
        //         %0 = llvm.call @mlirAsyncRuntimeCreateToken() : () -> !llvm.ptr
        //         %1 = llvm.call @mlirAsyncRuntimeCreateValue() : () -> !llvm.ptr
        //         %null = llvm.mlir.null : !llvm.ptr
        //         llvm.invoke @callee() to ^bbsuccess unwind ^bbfail : () -> ()
        //     ^bbfail:
        //         %3 = llvm.landingpad (catch %null : !llvm.ptr) : !llvm.struct<(ptr<i8>, i32)>
        //         llvm.call @mlirAsyncRuntimeSetTokenError(%0) : (!llvm.ptr) -> ()
        //         llvm.call @mlirAsyncRuntimeSetValueError(%1) : (!llvm.ptr) -> ()
        //     ^bbsuccess:
        //         llvm.call @mlirAsyncRuntimeEmplaceToken(%0) : (!llvm.ptr) -> ()
        //         llvm.call @mlirAsyncRuntimeEmplaceValue(%1) : (!llvm.ptr) -> ()
        //         llvm.br ^termination
        //     ^termination:
        //         // some cleanup and return
        //     }
        insertBranchFromFailToSuccessor(failBlock, successor, rewriter);
    }

    // We will now annotate caller for the next stage.
    //
    //     llvm.func caller() attributes { catalyst.preHandleError }
    AsyncUtils::scheduleAnalysisForErrorHandling(caller, rewriter);
}

/* The next step is to inspect callers of the previous caller.
 * So, in other words, we will be inspecting the callers of functions annotated with {
 * catalyst.preHandleError }
 */
struct RemoveAbortAndPutsInsertCallTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

// In this pattern we are looking for function calls to functions annotated
// with the { catalyst.preHandleError } attribute.
//
//    llvm.func @async_execute_fn() attributes { catalyst.preHandleError }
//    // ... snip ...
//    %results = call @async_execute_fn()
//
// These functions return async values or tokens.
LogicalResult RemoveAbortAndPutsInsertCallTransform::match(LLVM::CallOp callOp) const
{
    auto maybeCallee = AsyncUtils::getCalleeSafe(callOp);
    if (!maybeCallee)
        return failure();

    // llvm.func @callee() attributes { catalyst.preHandleError }
    auto calleeFuncOp = maybeCallee.value();
    bool hasAttr = AsyncUtils::hasPreHandleErrorAttr(calleeFuncOp);
    if (!hasAttr)
        return failure();

    return success();
}

void RemoveAbortAndPutsInsertCallTransform::rewrite(LLVM::CallOp callOp,
                                                    PatternRewriter &rewriter) const
{
    auto maybeCallee = AsyncUtils::getCalleeSafe(callOp);
    if (!maybeCallee)
        return;

    // Here, we are declaring an external function which is available in the Catalyst runtime.
    //     llvm.func @__catalyst__host__rt__unrecoverable_error()
    auto moduleOp = callOp->getParentOfType<ModuleOp>();
    auto unrecoverableError = AsyncUtils::lookupOrCreateUnrecoverableError(moduleOp);

    auto callee = maybeCallee.value();
    rewriter.updateRootInPlace(callee, [&] { callee.setLinkage(LLVM::Linkage::Internal); });

    // llvm.func @async_execute_fn() attributes { catalyst.preHandleError }
    // %results = call @async_execute_fn()
    auto results = callOp.getResults();

    // Here, we need to decide what we are looking for.
    // Because we are in low level code, %results actually is typed
    // as either a ptr or a tuple of ptrs. If it is a ptr, it means
    // that the @async_execute_region had no returns and it is only
    // returning a token to find out if the function finished running.
    // This would be in the case of the following asynchrnous qnode, for example:
    //
    // @qml.qnode()
    // def foo():
    //   return
    //
    // If it is a struct of ptrs, the first element of the struct will always be the token,
    // and the rest are the promises.
    // So, we need to decide if we are looking for just the token, or the token and promises.:
    SmallVector<Value> valuesToLookFor;
    // TODO: Assert that we have results
    assert(results.size() == 1);
    collectValuesToLookFor(results, valuesToLookFor);

    // Here, we are collecting the results of functions calls to
    //     %3 = llvm.call @mlirAsyncRuntimeIsTokenError(%0) : (!llvm.ptr) -> i1
    //     %4 = llvm.call @mlirAsyncRuntimeIsValueError(%1) : (!llvm.ptr) -> i1
    // These functions take a token or value and return a boolean on whether the
    // state is an error.
    //     callResults = { %3, %4 }
    SmallVector<Value> callResults;
    collectResultsForMlirAsyncRuntimeErrorFunctions(valuesToLookFor, callResults);

    // These values may be used as conditions in branches.
    // However, the values may be used as operands in other operations.
    // For example, the lowering rules for the async runtime use
    //     %one = llvm.constant 1 : i64
    //     %5 = llvm.xor %3, %1 : i1
    //     // ... snip ...
    //     %6 = llvm.xor %4, %1 : i1
    // So we must also append these to collectResults.
    //     potentialConditions = { %3, %4, %5, %6 }
    SmallVector<Value> potentialConditions(callResults.begin(), callResults.end());
    collectPotentialConditions(callResults, potentialConditions);

    // Now, these values will be used as operands in conditional branches.
    // We need to decide which of the destination of the branches are successes and which ones are
    // aborts. We do so by looking into the block. If the block contains a call into an @abort
    // function, it is an abort block. Otherwise, it is a success block.
    //
    //     llvm.cond_br %5, ^bb0, ^bb1
    //     ^5:
    //     call @llvm.abort()
    //     ^6:
    //     // something else
    //
    // abortBlocks = { ^5 }
    // successBlocks = { ^6 }
    SmallVector<Block *> abortBlocks;
    SmallVector<Block *> successBlocks;
    collectSuccessorBlocks(potentialConditions, abortBlocks, successBlocks);

    // We now collect the aborts from the abort blocks.
    // aborts = { llvm.call @llvm.abort, ..., ..., ... }
    SmallVector<LLVM::CallOp> aborts;
    collectCallsToAbortInBlocks(abortBlocks, aborts);

    SmallVector<LLVM::CallOp> newCalls;
    // And we replace aborts with calls to unrecoverable errors.
    // At the same time we populate new calls with calls to unrecoverable error.
    //
    //     llvm.cond_br %5, ^bb0, ^bb1
    //     ^5:
    //     llvm.call @__catalyst__host__rt__unrecoverable_error()
    //     ^6:
    //     // something else
    //
    // newCalls = { llvm.call @__catalyst__host_ ..., ..., ... }
    replaceCallsWithCallToTarget(aborts, unrecoverableError, newCalls, rewriter);

    // Collect blocks with puts calls
    SmallVector<Block *> putsBlocks;
    collectPutsBlocks(potentialConditions, putsBlocks);

    // Remove puts calls
    RemoveCallsToPutsInBlocks(putsBlocks, rewriter);

    // This is a subtlety, but it is a very important one!
    // In order for the (liveness) dataflow analysis, the values need to flow from failure block
    // to some uses. Otherwise, the values are not alive in the abortBlocks.
    // So we change:
    //
    //     %2 = llvm.call @async_execute_fn()
    //     %3 = llvm.extract ...
    //     %4 = llvm.extract ...
    //     // %3, %4 <-- these are values that must be dropped ref
    //     llvm.cond_br %5, ^bb0, ^bb1
    //     ^5:
    //     llvm.call @__catalyst__host__rt__unrecoverable_error()
    //     llvm.unreachable
    //     // All values are dead at the end of basic block ^5
    //     ^6:
    //     // %3, %4 are used in either ^6 or successor blocks
    //
    // To:
    //
    //     // %3, %4 <-- these are values that must be dropped ref
    //     llvm.cond_br %5, ^bb0, ^bb1
    //     ^5:
    //     llvm.call @__catalyst__host__rt__unrecoverable_error()
    //     llvm.br ^6 // Now values %3 and %4 are alive in block ^5
    //     ^6:
    //     // %3, %4 are used in either ^6 or successor blocks
    //
    // TODO:
    // Alternatively instead of changing the failure block, you can just calculate the live values
    // of the success block and use them in the sink.
    // This would allow you to reuse liveness analysis results between each of these
    // transformations and avoid changing this operation back.
    replaceTerminatorWithUnconditionalJumpToSuccessBlock(abortBlocks, successBlocks, rewriter);

    // We annotate this call as a source
    //
    //     %2 = llvm.call @async_execute_fn() { catalyst.sourceOfRefCounts }
    AsyncUtils::annotateCallForSource(callOp, rewriter);
    // We annotate all of the unrecoverable errors as sinks.
    //     llvm.call @__catalyst__host__rt__unrecoverable_error() { catalyst.sink }
    AsyncUtils::annotateCallsForSink(newCalls, rewriter);

    // We can make a note here, that we will be interested in the flow of sources to the current
    // sink. And values that are alive in the sink, must then be deallocated.

    // And we remove catalyst.preHandleError attribute to avoid re-transforming this llvm.call
    // from:
    //    llvm.func @async_execute_fn() attributes { catalyst.preHandleError }
    // to
    //    llvm.func @async_execute_fn()
    AsyncUtils::cleanupPreHandleErrorAttr(callee, rewriter);
}

// We come to the liveness analysis, which will find out values that flow from multiple
// sources to a single sink.
struct LivenessAnalysisDropRef : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

    LogicalResult match(LLVM::CallOp op) const override;
    void rewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

LogicalResult LivenessAnalysisDropRef::match(LLVM::CallOp op) const
{
    // We match on function calls that have the sink attribute.
    //     llvm.call @__catalyst__host__rt__unrecoverable_error() { catalyst.sink }
    return AsyncUtils::isSink(op) ? success() : failure();
}

void LivenessAnalysisDropRef::rewrite(LLVM::CallOp sink, PatternRewriter &rewriter) const
{
    auto caller = AsyncUtils::getCaller(sink);

    SmallVector<LLVM::CallOp> sources;
    SmallVector<Value> tokens;
    SmallVector<Value> values;

    // We are walking the body of the function containing the sink.
    //     sources = { %0 = llvm.call @async_execute_fn(), %1 = llvm.call @async_execute_fn2() }}
    //     tokens = { llvm.extract %0[0], llvm.extract %1[0] }
    //     values = { llvm.extract %0[1], llvm.extract %1[1] }
    // values are values that should be deallocated at some point!
    // subtletly: they don't necessarily need to be deallocated at this sink.
    // but at another sink in this same function. This depends on the liveness analysis.
    caller->walk([&](LLVM::CallOp callOp) {
        bool isInteresting = AsyncUtils::callsSource(callOp);
        if (!isInteresting)
            return;

        sources.push_back(callOp);
        auto results = callOp.getResults();

        collectValuesToLookFor(results, tokens, values);
    });

    // valuesToDrop = subset of values which are live in sink.
    //
    //     // %3, %4 <-- these are values that must be dropped ref
    //     llvm.cond_br %5, ^bb0, ^bb1
    //     ^5:
    //     llvm.call @__catalyst__host__rt__unrecoverable_error() { catalyst.sink }
    //     llvm.br ^6 // Now values %3 and %4 are alive in block ^5
    //     ^6:
    //     // %3, %4 are used in either ^6 or successor blocks
    //     // %7, %8 are values that are not alive in the sink
    //
    // valuesToDrop = { %3, %4 }
    llvm::SmallPtrSet<Value, 8> valuesToDrop;
    Liveness liveness(caller);
    std::vector<Value> tokensAndValues(tokens.begin(), tokens.end());
    tokensAndValues.insert(tokensAndValues.end(), values.begin(), values.end());

    for (auto value : tokensAndValues) {
        for (auto operationWhereValueIsAlive : liveness.resolveLiveness(value)) {
            if (operationWhereValueIsAlive == sink) {
                valuesToDrop.insert(value);
            }
        }
    }

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPoint(sink);

    auto moduleOp = sink->getParentOfType<ModuleOp>();
    // We are going to have to wait for these threads
    // before we can drop them.
    // So let's ensure that the declaration is available.
    //
    //     llvm.func @mlirAsyncRuntimeAwaitValue(!llvm.ptr)
    //     llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr)
    //     llvm.func @mlirAsyncRuntimeDropRef(!llvm.ptr, i64)
    auto awaitFnDecl = AsyncUtils::lookupOrCreateAwaitTokenName(moduleOp);
    auto dropRefFnDecl = AsyncUtils::lookupOrCreateDropRef(moduleOp);

    Type llvmInt64Type = IntegerType::get(sink->getContext(), 64);
    auto one = rewriter.getIntegerAttr(llvmInt64Type, 1);
    Value c1 = rewriter.create<LLVM::ConstantOp>(sink->getLoc(), llvmInt64Type, one);

    // We just need to await for the tokens.
    // The tokens is the aggregate of all values.
    // Only when all values are available / error, then the token is as well.
    //
    //     llvm.call @mlirAsyncRuntimeAwaitToken
    //     llvm.call @__catalyst__host__rt__unrecoverable_error() { catalyst.sink }
    for (auto awaitMe : tokens) {
        auto contains = valuesToDrop.find(awaitMe) != valuesToDrop.end();
        if (contains)
            rewriter.create<LLVM::CallOp>(sink.getLoc(), awaitFnDecl, awaitMe);
    }

    // We will drop all values that were alive. Tokens and values.
    //     llvm.call @mlirAsyncRuntimeAwaitToken
    //     llvm.call @mlirAsyncRuntimeDropRef(%3, %c1) : (!llvm.ptr, i64) -> ()
    //     llvm.call @mlirAsyncRuntimeDropRef(%4, %c1) : (!llvm.ptr, i64) -> ()
    //     llvm.call @__catalyst__host__rt__unrecoverable_error() { catalyst.sink }
    for (auto dropMe : valuesToDrop) {
        SmallVector<Value> params = {dropMe, c1};
        rewriter.create<LLVM::CallOp>(sink.getLoc(), dropRefFnDecl, params);
    }

    // It is important that we do not cleanup the source, as other sinks
    // need this information. It will be cleaned up at a later stage.
    // NEVER CALL:
    //    cleanupSource(annotatedCalls, rewriter);
    AsyncUtils::cleanupSink(sink, rewriter);
}

// We now can cleanup the source
// and change back the branches to unreachable.
// Which are trivial changes.
struct CleanUpSourceTransform : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override;
};

struct BranchToUnreachableTransform : public OpRewritePattern<LLVM::BrOp> {
    using OpRewritePattern<LLVM::BrOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(LLVM::BrOp op, PatternRewriter &rewriter) const override;
};

LogicalResult CleanUpSourceTransform::matchAndRewrite(LLVM::CallOp candidate,
                                                      PatternRewriter &rewriter) const
{
    if (!AsyncUtils::callsSource(candidate))
        return failure();

    AsyncUtils::cleanupSource(candidate, rewriter);
    return success();
}

LogicalResult BranchToUnreachableTransform::matchAndRewrite(LLVM::BrOp candidate,
                                                            PatternRewriter &rewriter) const
{
    bool hasAttr = AsyncUtils::hasChangeToUnreachableAttr(candidate);
    if (!hasAttr)
        return failure();

    auto unreachable = rewriter.create<LLVM::UnreachableOp>(candidate.getLoc());
    rewriter.replaceOp(candidate, unreachable);
    return success();
}

// TODO:
// This is not over yet though.
// Because we can have the following situation.
// Imagine the following call graph.
//               A
//           /       \
//         B        async C
//         |
//       async D
//
// We can imagine a situation where
//
//    A starts running.
//    A calls C and C executes asynchronously.
//    A calls B
//    B calls D and D executes asynchronously
//
// Due to the invariants we have placed, we must await on D before returning to A.
// So, if there was an error in D, we actually never returned to A
// and we never deallocated the memory for return values from C.
// What we need to do is propagate this analysis up. (Or at least parts of it).
// We can now say that A calls a function B and B may raise an exception.
// So A must catch it and decide how to deallocate resources from A, B, and C.

void collectCallsToAbortInBlocks(SmallVector<Block *> &blocks, SmallVector<LLVM::CallOp> &calls)
{
    for (Block *block : blocks) {
        block->walk([&](LLVM::CallOp op) {
            if (AsyncUtils::callsAbort(op)) {
                LLVM::CallOp abortCall = cast<LLVM::CallOp>(op);
                calls.push_back(abortCall);
            }
        });
    }
}

void RemoveCallsToPutsInBlocks(SmallVector<Block *> &blocks, PatternRewriter &rewriter)
{
    for (Block *block : blocks) {
        block->walk([&](LLVM::CallOp op) {
            if (AsyncUtils::callsPuts(op)) {
                LLVM::CallOp putsCall = cast<LLVM::CallOp>(op);
                rewriter.eraseOp(putsCall);
            }
        });
    }
}

void replaceCallsWithCallToTarget(SmallVector<LLVM::CallOp> &oldCallOps, LLVM::LLVMFuncOp target,
                                  SmallVector<LLVM::CallOp> &newCalls, PatternRewriter &rewriter)
{
    for (auto oldCallOp : oldCallOps) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPoint(oldCallOp);
        auto newCallOp =
            rewriter.create<LLVM::CallOp>(oldCallOp.getLoc(), target, oldCallOp.getOperands());
        rewriter.replaceOp(oldCallOp, newCallOp);
        newCalls.push_back(newCallOp);
    }
}

void collectSuccessorBlocks(SmallVector<Value> &conditions, SmallVector<Block *> &aborts,
                            SmallVector<Block *> &success)
{
    for (auto condition : conditions) {
        for (Operation *user : condition.getUsers()) {
            if (isa<LLVM::CondBrOp>(user)) {
                LLVM::CondBrOp brOp = cast<LLVM::CondBrOp>(user);
                Block *trueDest = brOp.getTrueDest();
                Block *falseDest = brOp.getFalseDest();
                if (AsyncUtils::hasAbortInBlock(trueDest)) {
                    aborts.push_back(trueDest);
                    success.push_back(falseDest);
                }
                else {
                    aborts.push_back(falseDest);
                    success.push_back(trueDest);
                }
            }
        }
    }
}

void collectPutsBlocks(SmallVector<Value> &conditions, SmallVector<Block *> &puts)
{
    for (auto condition : conditions) {
        for (Operation *user : condition.getUsers()) {
            if (isa<LLVM::CondBrOp>(user)) {
                LLVM::CondBrOp brOp = cast<LLVM::CondBrOp>(user);
                Block *trueDest = brOp.getTrueDest();
                Block *falseDest = brOp.getFalseDest();
                puts.push_back(AsyncUtils::hasPutsInBlock(trueDest) ? trueDest : falseDest);
            }
        }
    }
}

void collectResultsForMlirAsyncRuntimeErrorFunctions(SmallVector<Value> &values,
                                                     SmallVector<Value> &results)
{
    for (Value value : values) {
        for (Operation *user : value.getUsers()) {
            bool isCallToIsErrorToken = AsyncUtils::callsMlirAsyncRuntimeIsTokenError(user);
            bool isCallToIsValueToken = AsyncUtils::callsMlirAsyncRuntimeIsValueError(user);
            bool isValid = isCallToIsErrorToken || isCallToIsValueToken;
            if (!isValid)
                continue;

            auto boolVal = user->getResult(0);
            results.push_back(boolVal);
        }
    }
}

void collectPotentialConditions(SmallVector<Value> &values, SmallVector<Value> &conditions)
{
    for (auto boolVal : values) {
        for (Operation *user : boolVal.getUsers()) {
            if (isa<LLVM::XOrOp>(user)) {
                auto xorResult = user->getResult(0);
                conditions.push_back(xorResult);
            }
        }
    }
}

void collectValuesToLookFor(ResultRange &results, SmallVector<Value> &tokens,
                            SmallVector<Value> &values)
{
    Value result = results.front();
    Type resultTy = result.getType();

    if (isa<LLVM::LLVMPointerType>(resultTy)) {
        tokens.push_back(result);
    }
    else if (isa<LLVM::LLVMStructType>(resultTy)) {
        for (Operation *user : result.getUsers()) {
            if (isa<LLVM::ExtractValueOp>(user)) {
                LLVM::ExtractValueOp extract = cast<LLVM::ExtractValueOp>(user);
                auto isToken = extract.getPosition()[0] == 0;
                if (isToken)
                    tokens.push_back(user->getResult(0));
                else
                    values.push_back(user->getResult(0));
            }
        }
    }
}

void collectValuesToLookFor(ResultRange &results, SmallVector<Value> &valuesToLookFor)
{
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
}

void replaceTerminatorWithUnconditionalJumpToSuccessBlock(SmallVector<Block *> abortBlocks,
                                                          SmallVector<Block *> successBlocks,
                                                          PatternRewriter &rewriter)
{
    for (auto [abort, success] : llvm::zip(abortBlocks, successBlocks)) {
        PatternRewriter::InsertionGuard insertGuard(rewriter);
        auto terminator = abort->getTerminator();
        rewriter.setInsertionPoint(terminator);
        assert(isa<LLVM::UnreachableOp>(terminator));
        auto brOp = rewriter.create<LLVM::BrOp>(terminator->getLoc(), success);
        // Make sure we clean it up later.
        AsyncUtils::annotateBrToUnreachable(brOp, rewriter);
        rewriter.replaceOp(terminator, brOp);
    }
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
    auto zeroOp = rewriter.create<LLVM::ZeroOp>(callOp.getLoc(), ptrTy);
    Block *unwindBlock = rewriter.createBlock(successBlock);

    rewriter.setInsertionPointToEnd(unwindBlock);
    bool isCleanUp = true;
    std::vector<Value> operands = {zeroOp.getResult()};
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

LLVM::InvokeOp transformCallToInvoke(LLVM::CallOp callOp, Block *successBlock, Block *failBlock,
                                     PatternRewriter &rewriter)
{
    auto calleeAttr = callOp.getCalleeAttr();
    SmallVector<Value> unwindArgs;
    auto invokeOp = rewriter.create<LLVM::InvokeOp>(callOp.getLoc(), callOp.getResultTypes(),
                                                    calleeAttr, callOp.getOperands(), successBlock,
                                                    ValueRange(), failBlock, unwindArgs);
    rewriter.replaceOp(callOp, invokeOp);
    return invokeOp;
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
        auto calleeMaybeIndirect = AsyncUtils::getCalleeSafe(call);
        if (!calleeMaybeIndirect)
            return;

        auto callee = calleeMaybeIndirect.value();
        bool tokens = AsyncUtils::isMlirAsyncRuntimeCreateToken(callee);
        bool values = AsyncUtils::isMlirAsyncRuntimeCreateValue(callee);
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

    LLVM::LLVMFuncOp setTokenError =
        AsyncUtils::lookupOrCreateMlirAsyncRuntimeSetTokenError(moduleOp);
    LLVM::LLVMFuncOp setValueError =
        AsyncUtils::lookupOrCreateMlirAsyncRuntimeSetValueError(moduleOp);
    for (auto token : tokens) {
        insertCallToMlirAsyncRuntimeErrorFunction(token, setTokenError, failBlock, rewriter);
    }
    for (auto value : values) {
        insertCallToMlirAsyncRuntimeErrorFunction(value, setValueError, failBlock, rewriter);
    }
}

void insertBranchFromFailToSuccessor(Block *fail, Block *success, PatternRewriter &rewriter)
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

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_ADDEXCEPTIONHANDLINGPASS
#define GEN_PASS_DECL_ADDEXCEPTIONHANDLINGPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct AddExceptionHandlingPass : impl::AddExceptionHandlingPassBase<AddExceptionHandlingPass> {
    using AddExceptionHandlingPassBase::AddExceptionHandlingPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();

        RewritePatternSet patterns1(context);
        patterns1.add<DetectCallsInAsyncRegionsTransform>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns1)))) {
            signalPassFailure();
        }

        if (stopAfterStep == 1) {
            return;
        }

        RewritePatternSet patterns2(context);
        patterns2.add<AddExceptionHandlingTransform>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns2)))) {
            signalPassFailure();
        }

        if (stopAfterStep == 2) {
            return;
        }

        RewritePatternSet patterns3(context);
        patterns3.add<RemoveAbortAndPutsInsertCallTransform>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns3)))) {
            signalPassFailure();
        }

        if (stopAfterStep == 3) {
            return;
        }

        RewritePatternSet patterns4(context);
        patterns4.add<LivenessAnalysisDropRef>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns4)))) {
            signalPassFailure();
        }

        if (stopAfterStep == 4) {
            return;
        }

        RewritePatternSet patterns5(context);
        patterns5.add<CleanUpSourceTransform, BranchToUnreachableTransform>(context);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns5)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createAddExceptionHandlingPass()
{
    return std::make_unique<AddExceptionHandlingPass>();
}

} // namespace catalyst
