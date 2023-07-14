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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

struct AnnotateFunctionTransform : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;
};

LogicalResult AnnotateFunctionTransform::match(func::FuncOp op) const {
    return failure();
}

void AnnotateFunctionTransform::rewrite(func::FuncOp op, PatternRewriter &rewriter) const
{
}

} // namespace

namespace catalyst {

#define GEN_PASS_DEF_ANNOTATEFUNCTIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct AnnotateFunctionPass : impl::AnnotateFunctionPassBase<AnnotateFunctionPass> {
    using AnnotateFunctionPassBase::AnnotateFunctionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
	RewritePatternSet patterns(context);
	patterns.add<AnnotateFunctionTransform>(context);

	if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
	    signalPassFailure();
	}
    }
};

std::unique_ptr<Pass> createAnnotateFunctionPass()
{
    return std::make_unique<AnnotateFunctionPass>();
}

} // namespace catalyst
