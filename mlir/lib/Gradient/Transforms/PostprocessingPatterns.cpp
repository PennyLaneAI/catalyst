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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Utils/GradientShape.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace {

struct PostprocessForwardOp : public OpRewritePattern<ForwardOp> {
    using mlir::OpRewritePattern<ForwardOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ForwardOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        // Check if the numbers of args and returns match Enzyme's format.
        auto argc = op.getArgc();
        auto resc = op.getResc();
        auto tapeCount = op.getTape();

        // if (op.getNumArguments() == (argc + resc) * 2 && op.getNumResults() == tapeCount)
        if (op.getFunctionType().getNumInputs() == (argc + resc) * 2 &&
            op.getFunctionType().getNumResults() == tapeCount)
            return failure();

        auto argTys = op.getArgumentTypes();
        auto retTys = op.getResultTypes();
        SmallVector<Type> bufferArgs;
        SmallVector<Type> bufferRets;

        // Prepare for arg insertion.
        unsigned appendingSize = argc + 2 * resc;
        SmallVector<unsigned> argIndices(/*size=*/appendingSize,
                                         /*values=*/op.getNumArguments());
        SmallVector<Type> argTypes;
        SmallVector<DictionaryAttr> argAttrs{appendingSize};
        SmallVector<Location> argLocs{appendingSize, op.getLoc()};

        for (Type ty : argTys) {
            bufferArgs.push_back(ty);
            bufferArgs.push_back(ty);

            // create new argument to insert
            argTypes.push_back(ty);
        }

        for (size_t i = 0; i < op.getNumResults(); i++) {
            auto ty = retTys[i];
            if (i < resc) {
                bufferArgs.push_back(ty);
                bufferArgs.push_back(ty);
                argTypes.push_back(ty);
                argTypes.push_back(ty);
            }
            else {
                bufferRets.push_back(ty);
            }
        }

        auto forwardTy = rewriter.getFunctionType(bufferArgs, bufferRets);
        rewriter.modifyOpInPlace(op, [&] {
            op.insertArguments(argIndices, argTypes, argAttrs, argLocs);
            op.setFunctionType(forwardTy);
        });

        op.walk([&](ReturnOp returnOp) {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(returnOp);
            SmallVector<Value> tapeReturns;
            size_t idx = 0;
            for (Value operand : returnOp.getOperands()) {
                if (isa<MemRefType>(operand.getType()) && idx < resc) {
                    BlockArgument output = op.getArgument(idx + argc * 2);
                    rewriter.create<memref::CopyOp>(returnOp.getLoc(), operand, output);
                    idx++;
                }
                else {
                    tapeReturns.push_back(operand);
                }
            }
            rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, tapeReturns, returnOp.getEmpty());
        });
        return success();
    }
};

struct PostprocessReverseOp : public OpRewritePattern<ReverseOp> {
    using OpRewritePattern<ReverseOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ReverseOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        // Check if the numbers of args and returns match Enzyme's format.
        auto argc = op.getArgc();
        auto resc = op.getResc();
        auto tape = op.getTape();

        // if (op.getNumArguments() == (argc + resc) * 2 && op.getNumResults() == tapeCount)
        if (op.getFunctionType().getNumInputs() == (argc + resc) * 2 + tape)
            return failure();

        auto argTys = op.getArgumentTypes();
        auto retTys = op.getResultTypes();
        SmallVector<Type> bufferArgs;
        SmallVector<Type> bufferRets;

        // Prepare for arg insertion.
        unsigned appendingSize = argc + 2 * resc;
        SmallVector<unsigned> argIndices(/*size=*/appendingSize,
                                         /*values=*/0);
        SmallVector<Type> argTypes;
        SmallVector<DictionaryAttr> argAttrs{appendingSize};
        SmallVector<Location> argLocs{appendingSize, op.getLoc()};

        // For the function format of ReverseOP should follow that of ForwardOp,
        // so the returns go to the front.
        for (Type ty : retTys) {
            bufferArgs.push_back(ty);
            bufferArgs.push_back(ty);

            // create new argument to insert
            argTypes.push_back(ty);
            argTypes.push_back(ty);
        }

        // Tape is with the arguments for ReversOp/
        for (size_t i = 0; i < op.getNumArguments(); i++) {
            auto ty = argTys[i];
            if (i < resc) {
                bufferArgs.push_back(ty);
                bufferArgs.push_back(ty);
                argTypes.push_back(ty);
            }
            else {
                bufferArgs.push_back(ty);
            }
        }

        auto reverseTy = rewriter.getFunctionType(bufferArgs, bufferRets);
        rewriter.modifyOpInPlace(op, [&] {
            op.insertArguments(argIndices, argTypes, argAttrs, argLocs);
            op.setFunctionType(reverseTy);
        });

        op.walk([&](ReturnOp returnOp) {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(returnOp);
            SmallVector<Value> tapeReturns;
            size_t idx = 0;
            for (Value operand : returnOp.getOperands()) {
                if (isa<MemRefType>(operand.getType()) && idx < resc) {
                    BlockArgument output = op.getArgument(idx + argc);
                    rewriter.create<memref::CopyOp>(returnOp.getLoc(), operand, output);
                    idx++;
                }
            }
            rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, tapeReturns, returnOp.getEmpty());
        });

        return success();
    }
};

} // namespace

namespace catalyst {
namespace gradient {

void populatePostprocessingPatterns(RewritePatternSet &patterns)
{
    patterns.add<PostprocessForwardOp>(patterns.getContext());
    patterns.add<PostprocessReverseOp>(patterns.getContext());
}

} // namespace gradient
} // namespace catalyst