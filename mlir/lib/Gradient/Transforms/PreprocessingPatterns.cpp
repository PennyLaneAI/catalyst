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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
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
        if(!op.getFunctionBody().empty())
            return failure();

        llvm::outs() << "forward\n";

        auto argc = op.getArgc();
        auto resc = op.getResc();
        SmallVector<Value> inputs;
        SmallVector<Value> differentials;
        SmallVector<Value> outputs;
        SmallVector<Value> cotangents;

        Block *block;
        rewriter.modifyOpInPlace(op, [&] { block = op.addEntryBlock(); });

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(block);
        auto params = op.getArguments();

        for (size_t i = 0; i < argc * 2; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? differentials.push_back(val) : inputs.push_back(val);
        }

        auto upperLimit = (argc * 2) + (resc * 2);
        for (size_t i = argc * 2; i < upperLimit; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? cotangents.push_back(val) : outputs.push_back(val);
        }

        auto implAttr = op.getImplementationAttr();
        auto impl = op.getImplementation();
        auto implOp = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(op, implAttr);
        auto implResTy = implOp.getResultTypes();
        Location loc = op.getLoc();

        auto callOp = rewriter.create<func::CallOp>(loc, impl, implResTy, inputs);
        SmallVector<Value> tensorOutputs(callOp.getResults());

        auto tapeCount = op.getTape();
        SmallVector<Value> tapeOutputs;
        tapeOutputs.insert(tapeOutputs.begin(), tensorOutputs.end() - tapeCount,
                           tensorOutputs.end());

        auto F = rewriter.getIntegerAttr(rewriter.getI1Type(), 0);
        rewriter.create<catalyst::gradient::ReturnOp>(loc, tapeOutputs, F);

        return success();
    }
};

struct PreprocessReverseOp : public OpRewritePattern<ReverseOp> {
    using OpRewritePattern<ReverseOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ReverseOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        if(!op.getFunctionBody().empty())
            return failure();

        llvm::outs() << "reverse\n";

        auto argc = op.getArgc();
        auto resc = op.getResc();
        SmallVector<Value> inputs;
        SmallVector<Value> differentials;
        SmallVector<Value> outputs;
        SmallVector<Value> cotangents;
        SmallVector<Value> tapeElements;

        Block *block;
        rewriter.modifyOpInPlace(op, [&] { block = op.addEntryBlock(); });

        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(block);
        auto params = op.getArguments();

        for (size_t i = 0; i < argc * 2; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? differentials.push_back(val) : inputs.push_back(val);
        }

        auto upperLimit = (argc * 2) + (resc * 2);
        for (size_t i = argc * 2; i < upperLimit; i++) {
            bool isDup = (i % 2) != 0;
            Value val = params[i];
            isDup ? cotangents.push_back(val) : outputs.push_back(val);
        }

        auto tapeCount = op.getTape();
        auto uppestLimit = upperLimit + tapeCount;
        for (size_t i = upperLimit; i < uppestLimit; i++) {
            tapeElements.push_back(params[i]);
        }

        auto implAttr = op.getImplementationAttr();
        auto impl = op.getImplementation();
        auto implOp = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(op, implAttr);
        auto implResTy = implOp.getResultTypes();
        Location loc = op.getLoc();

        SmallVector<Value> tensorInputs;
        for (auto tapeElement : tapeElements) {
            tensorInputs.push_back(tapeElement);
        }

        for (auto cotangent : cotangents) {
            tensorInputs.push_back(cotangent);
        }

        rewriter.create<func::CallOp>(loc, impl, implResTy, tensorInputs);

        auto T = rewriter.getIntegerAttr(rewriter.getI1Type(), 1);
        rewriter.create<catalyst::gradient::ReturnOp>(loc, ValueRange{}, T);
        
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
