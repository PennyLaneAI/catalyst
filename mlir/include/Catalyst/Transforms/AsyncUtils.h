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

using namespace mlir;

namespace AsyncUtils {

// Helper function for attributes
bool hasQnodeAttribute(LLVM::LLVMFuncOp funcOp);
bool hasPreHandleErrorAttr(LLVM::LLVMFuncOp funcOp);
bool isScheduledForTransformation(LLVM::CallOp callOp);
bool isSink(mlir::Operation *op);
bool isAsync(LLVM::LLVMFuncOp funcOp);
bool hasChangeToUnreachableAttr(mlir::Operation *op);
void scheduleCallToInvoke(LLVM::CallOp callOp, PatternRewriter &rewriter);
void scheduleAnalysisForErrorHandling(LLVM::LLVMFuncOp funcOp, PatternRewriter &rewriter);
void cleanupPreHandleErrorAttr(LLVM::LLVMFuncOp funcOp, PatternRewriter &rewriter);
void scheduleLivenessAnalysis(LLVM::CallOp callOp, PatternRewriter &rewriter);
void annotateCallsForSink(SmallVector<LLVM::CallOp> &calls, PatternRewriter &rewriter);
bool callsSource(LLVM::CallOp callOp);
void annotateCallForSource(LLVM::CallOp callOp, PatternRewriter &rewriter);
void cleanupSink(LLVM::CallOp op, PatternRewriter &rewriter);
void cleanupSource(LLVM::CallOp source, PatternRewriter &rewriter);
void cleanupSource(SmallVector<LLVM::CallOp> &sources, PatternRewriter &rewriter);
void annotateBrToUnreachable(LLVM::BrOp op, PatternRewriter &rewriter);

// Helper function for caller/callees
LLVM::LLVMFuncOp getCaller(LLVM::CallOp callOp);
std::optional<LLVM::LLVMFuncOp> getCalleeSafe(LLVM::CallOp callOp);

// Helper functions for matching function names
bool isFunctionNamed(LLVM::LLVMFuncOp funcOp, llvm::StringRef expectedName);
bool isAbort(LLVM::LLVMFuncOp funcOp);
bool isPuts(LLVM::LLVMFuncOp funcOp);
bool callsAbort(LLVM::CallOp callOp);
bool callsPuts(LLVM::CallOp callOp);
bool callsAbort(Operation *possibleCall);
bool isMlirAsyncRuntimeCreateValue(LLVM::LLVMFuncOp funcOp);
bool isMlirAsyncRuntimeCreateToken(LLVM::LLVMFuncOp funcOp);
bool isMlirAsyncRuntimeIsTokenError(LLVM::LLVMFuncOp funcOp);
bool isMlirAsyncRuntimeIsValueError(LLVM::LLVMFuncOp funcOp);
bool callsMlirAsyncRuntimeCreateValue(LLVM::CallOp callOp);
bool callsMlirAsyncRuntimeCreateToken(LLVM::CallOp callOp);
bool callsMlirAsyncRuntimeCreateToken(Operation *possibleCall);
bool callsMlirAsyncRuntimeIsTokenError(LLVM::CallOp callOp);
bool callsMlirAsyncRuntimeIsValueError(LLVM::CallOp callOp);
bool callsMlirAsyncRuntimeIsTokenError(Operation *possibleCall);
bool callsMlirAsyncRuntimeIsValueError(Operation *possibleCall);
bool hasAbortInBlock(Block *block);
bool hasPutsInBlock(Block *block);

// Helper function for creating function declarations
LLVM::LLVMFuncOp lookupOrCreatePersonality(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreateAbort(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreateMlirAsyncRuntimeSetValueError(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreateMlirAsyncRuntimeSetTokenError(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreateUnrecoverableError(ModuleOp moduleOp);
LLVM::LLVMFuncOp lookupOrCreateAwaitTokenName(ModuleOp);
LLVM::LLVMFuncOp lookupOrCreateAwaitValueName(ModuleOp);
LLVM::LLVMFuncOp lookupOrCreateDropRef(ModuleOp);

}; // namespace AsyncUtils
