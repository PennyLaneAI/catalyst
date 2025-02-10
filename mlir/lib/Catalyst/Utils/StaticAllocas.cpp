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

LLVM::AllocaOp getStaticAlloca(Location &loc, RewriterBase &rewriter, Type ty, int value)
{
    // create an llvm.alloca operation at the beginning of the entry block of the current function.
    Block *insertionBlock = rewriter.getInsertionBlock();
    Region *parentRegion = insertionBlock->getParent();
    Block *entryBlock = &parentRegion->front();
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    // Move the value at the beginning
    rewriter.setInsertionPointAfter(&entryBlock->front());
    auto valueOp = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64IntegerAttr(value));
    return rewriter.create<LLVM::AllocaOp>(loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
                                           ty, valueOp);
}

mlir::memref::AllocaOp getStaticMemrefAlloca(Location &loc, RewriterBase &rewriter,
                                             MemRefType paramCountType)
{
    // Same as above but for memref.alloca instead of llvm.alloca
    Block *insertionBlock = rewriter.getInsertionBlock();
    Region *parentRegion = insertionBlock->getParent();
    Block *entryBlock = &parentRegion->front();
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    if (insertionBlock != entryBlock) {
        rewriter.setInsertionPoint(&entryBlock->front());
    }
    return rewriter.create<memref::AllocaOp>(loc, paramCountType);
}

} // namespace catalyst
