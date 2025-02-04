// Copyright 2025 Xanadu Quantum Technologies Inc.

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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace catalyst {

LLVM::AllocaOp getStaticAlloca(Location &loc, RewriterBase &rewriter, Type ty, Value value)
{
    Block *insertionBlock = rewriter.getInsertionBlock();
    Region *parentRegion = insertionBlock->getParent();
    Block *entryBlock = &parentRegion->front();
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    if (insertionBlock == entryBlock) {
        // ... noop ...
    }
    else {
        Operation *possible_terminator = entryBlock->getTerminator();
        assert(possible_terminator && "blocks must have a terminator");
        // we need it before the terminator
        Operation *value_def = value.getDefiningOp();
        rewriter.moveOpBefore(value_def, &entryBlock->front());
        rewriter.setInsertionPoint(possible_terminator);
    }
    return rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
                                           ty, value);
}

LLVM::AllocaOp getStaticAlloca2(Location &loc, RewriterBase &rewriter, Type ty, Value value)
{
    Block *insertionBlock = rewriter.getInsertionBlock();
    Region *parentRegion = insertionBlock->getParent();
    Block *entryBlock = &parentRegion->front();
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    if (insertionBlock == entryBlock) {
        // ... noop ...
    }
    else {
        Operation *possible_terminator = entryBlock->getTerminator();
        assert(possible_terminator && "blocks must have a terminator");
        // we need it before the terminator
        Operation *value_def = value.getDefiningOp();
        rewriter.moveOpBefore(value_def, &entryBlock->front());
        rewriter.setInsertionPointAfter(value_def);
    }
    return rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
                                           ty, value);
}

mlir::memref::AllocaOp getStaticMemrefAlloca(Location &loc, RewriterBase &rewriter,
                                             MemRefType paramCountType)
{
    Block *insertionBlock = rewriter.getInsertionBlock();
    Region *parentRegion = insertionBlock->getParent();
    Block *entryBlock = &parentRegion->front();
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    if (insertionBlock == entryBlock) {
        // ... noop ...
    }
    else {
        Operation *possible_terminator = entryBlock->getTerminator();
        assert(possible_terminator && "blocks must have a terminator");
        // we need it before the terminator
        rewriter.setInsertionPoint(possible_terminator);
    }
    return rewriter.create<memref::AllocaOp>(loc, paramCountType);
}

} // namespace catalyst
