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

#define DEBUG_TYPE "scatter"

#include <algorithm>
#include <vector>

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace catalyst {

struct FuncOpToAsyncOPRewritePattern : public mlir::OpRewritePattern<func::FuncOp> {
    using mlir::OpRewritePattern<func::FuncOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        if (op->hasAttrOfType<UnitAttr>("qnode")) {
            FunctionType opType = op.getFunctionType();

            auto inputTypes = opType.getInputs();
            auto resultTypes = opType.getResults();

            // Convert inputs to async values
            SmallVector<Type> asyncInputTypes;
            for (Type inputType : inputTypes) {
                asyncInputTypes.push_back(async::ValueType::get(inputType));
            }

            // Convert results to async values
            SmallVector<Type> asyncResultTypes;
            for (Type resultType : resultTypes) {
                asyncResultTypes.push_back(async::ValueType::get(resultType));
            }
            FunctionType asyncFunctionType = FunctionType::get(op.getContext(), /*inputs=*/
                                                               asyncInputTypes,
                                                               /*outputs=*/asyncResultTypes);
            StringRef nameOp = op.getName();
            auto asyncFunc = rewriter.create<async::FuncOp>(op.getLoc(), nameOp, asyncFunctionType);
            asyncFunc.setPrivate();
            // Create the block of the function
            Block *funcBody = asyncFunc.addEntryBlock();

            auto funcBlockArgs = funcBody->getArguments();
            rewriter.setInsertionPointToEnd(funcBody);
            
            // Merge the two blocks and delete the first one
            Region* body = &op.getFunctionBody();
            Block* firstBlock = &body->front();
            rewriter.mergeBlocks(firstBlock, funcBody, funcBlockArgs);
            rewriter.replaceOp(op, asyncFunc);
            return success();
        }
        return failure();
    }
};

struct CallOpToAsyncOPRewritePattern : public mlir::OpRewritePattern<func::CallOp> {
    using mlir::OpRewritePattern<func::CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(func::CallOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        // Call op that have
        // Convert the args
        return success();
    }
};

void populateQnodeToAsyncPatterns(RewritePatternSet &patterns)
{
    patterns.add<catalyst::FuncOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
    // patterns.add<catalyst::CallOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
}

} // namespace catalyst
