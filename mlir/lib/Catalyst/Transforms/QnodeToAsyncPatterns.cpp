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

            // Convert results to async values
            SmallVector<Type> asyncResultTypes;
            for (Type resultType : resultTypes) {
                Type asyncType = async::ValueType::get(resultType);
                asyncResultTypes.push_back(asyncType);
            }
            FunctionType asyncFunctionType = FunctionType::get(op.getContext(), /*inputs=*/
                                                               inputTypes,
                                                               /*outputs=*/asyncResultTypes);
            StringRef nameOp = op.getName();
            auto asyncFunc = rewriter.create<async::FuncOp>(op.getLoc(), nameOp, asyncFunctionType);
            asyncFunc.setPrivate();
            // Create the block of the function
            Block *asyncFuncBody = asyncFunc.addEntryBlock();

            auto asyncFuncBlockArgs = asyncFuncBody->getArguments();
            rewriter.setInsertionPointToEnd(asyncFuncBody);

            // Merge the two blocks and delete the first one
            Region *body = &op.getFunctionBody();
            Block *originalBlock = &body->front();
            Operation *originalTerminator = originalBlock->getTerminator();
            rewriter.mergeBlocks(originalBlock, asyncFuncBody, asyncFuncBlockArgs);
            // Replace the terminator with async return
            rewriter.create<async::ReturnOp>(op.getLoc(), originalTerminator->getResultTypes(),
                                             originalTerminator->getOperands());
            rewriter.eraseOp(originalTerminator);

            // Removing the function_type attribute as we don't want that overwritten.
            op->removeAttr("function_type");
            auto oldAttrs = op->getAttrs();
            asyncFunc->setAttrs(oldAttrs);

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
        SymbolRefAttr symbol = dyn_cast_if_present<SymbolRefAttr>(op.getCallableForCallee());
        async::FuncOp funcOp =
            dyn_cast_or_null<async::FuncOp>(SymbolTable::lookupNearestSymbolFrom(op, symbol));
        // Check for Call ops that have QNode func ops
        if (funcOp->hasAttrOfType<UnitAttr>("qnode")) {
            rewriter.create<async::CallOp>(op.getLoc(), funcOp, op.getArgOperands());
            // TODO: Add the awaits
            return success();
        }
        return failure();
    }
};

void populateQnodeToAsyncPatterns(RewritePatternSet &patterns)
{
    patterns.add<catalyst::FuncOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
    // patterns.add<catalyst::CallOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
}

} // namespace catalyst
