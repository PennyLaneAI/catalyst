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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;

namespace {

struct DetectQnodeTransform : public OpRewritePattern<LLVM::LLVMFuncOp> {
    using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

    LogicalResult match(LLVM::LLVMFuncOp op) const override;
    void rewrite(LLVM::LLVMFuncOp op, PatternRewriter &rewriter) const override;
};

LogicalResult DetectQnodeTransform::match(LLVM::LLVMFuncOp op) const { return failure(); }

void DetectQnodeTransform::rewrite(LLVM::LLVMFuncOp op, PatternRewriter &rewriter) const {}
} // namespace

namespace catalyst {

#define GEN_PASS_DEF_DETECTQNODEPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct DetectQnodePass : impl::DetectQnodePassBase<DetectQnodePass> {
    using DetectQnodePassBase::DetectQnodePassBase;

    void runOnOperation() final {}
};

std::unique_ptr<Pass> createDetectQnodePass() { return std::make_unique<DetectQnodePass>(); }

} // namespace catalyst
