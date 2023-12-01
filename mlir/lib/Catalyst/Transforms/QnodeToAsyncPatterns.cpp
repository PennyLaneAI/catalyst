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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Async/IR/Async.h"

using namespace mlir;

namespace catalyst {

struct FuncOpToAsyncOPRewritePattern : public mlir::OpRewritePattern<func::FuncOp> {
    using mlir::OpRewritePattern<func::FuncOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(func::FuncOp,
                                        mlir::PatternRewriter &rewriter) const override
    {
        return success();
    }
};

struct CallOpToAsyncOPRewritePattern : public mlir::OpRewritePattern<func::CallOp> {
    using mlir::OpRewritePattern<func::CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(func::CallOp,
                                        mlir::PatternRewriter &rewriter) const override
    {

        return success();
    }
};


void populateQnodeToAsyncPatterns(RewritePatternSet &patterns)
{
    patterns.add<catalyst::FuncOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
    // patterns.add<catalyst::CallOpToAsyncOPRewritePattern>(patterns.getContext(), 1);
}

} // namespace catalyst
