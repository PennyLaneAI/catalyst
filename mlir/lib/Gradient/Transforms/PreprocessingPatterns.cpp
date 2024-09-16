// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iostream"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Utils/GradientShape.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {

struct PreprocessForwardOp : public OpRewritePattern<ForwardOp> {
    using mlir::OpRewritePattern<ForwardOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ForwardOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        if (!op.getBody().empty())
            return failure();

        Block *block;
        rewriter.modifyOpInPlace(op, [&] { block = op.addEntryBlock(); });

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(block);
        auto inputs = op.getArguments();

        auto implAttr = op.getImplementationAttr();
        auto impl = op.getImplementation();
        auto implOp = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(op, implAttr);
        auto implResTy = implOp.getResultTypes();
        Location loc = op.getLoc();

        auto callOp = rewriter.create<func::CallOp>(loc, impl, implResTy, inputs);
        SmallVector<Value> outputs(callOp.getResults());

        auto F = rewriter.getIntegerAttr(rewriter.getI1Type(), 0);
        rewriter.create<catalyst::gradient::ReturnOp>(loc, outputs, F);

        return success();
    }
};

struct PreprocessReverseOp : public OpRewritePattern<ReverseOp> {
    using OpRewritePattern<ReverseOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ReverseOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        if (!op.getBody().empty())
            return failure();

        Block *block;
        rewriter.modifyOpInPlace(op, [&] { block = op.addEntryBlock(); });

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(block);
        auto inputs = op.getArguments();

        SmallVector<Value> tapeInputs;
        auto resc = op.getResc();
        // In ReverseOp, Tape comes first when we call the backward function.
        for (size_t i = 0; i < op.getTape(); i++) {
            tapeInputs.push_back(inputs[resc + i]);
        }

        for (size_t i = 0; i < resc; i++) {
            tapeInputs.push_back(inputs[i]);
        }

        auto implAttr = op.getImplementationAttr();
        auto impl = op.getImplementation();
        auto implOp = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(op, implAttr);
        auto implResTy = implOp.getResultTypes();
        Location loc = op.getLoc();

        auto callOp = rewriter.create<func::CallOp>(loc, impl, implResTy, tapeInputs);
        SmallVector<Value> outputs(callOp.getResults());

        auto F = rewriter.getIntegerAttr(rewriter.getI1Type(), 0);
        rewriter.create<catalyst::gradient::ReturnOp>(loc, outputs, F);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace gradient {

void populatePreprocessingPatterns(RewritePatternSet &patterns)
{
    patterns.add<PreprocessForwardOp>(patterns.getContext());
    patterns.add<PreprocessReverseOp>(patterns.getContext());
}

} // namespace gradient
} // namespace catalyst