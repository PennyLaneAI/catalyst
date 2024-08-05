// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace catalyst;

namespace {
struct MemrefCopyToLinalgCopyRewritePattern : public mlir::OpRewritePattern<memref::CopyOp> {
    using mlir::OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(memref::CopyOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        auto srcType = cast<BaseMemRefType>(op.getSource().getType());
        auto srcMemRefType = dyn_cast<MemRefType>(srcType);
        bool layoutIsIdentity = srcMemRefType.getLayout().isIdentity();
        if (!layoutIsIdentity) {
            rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, op.getSource(), op.getTarget());
            return success();
        }
        return failure();
    }
};

} // namespace

namespace catalyst {

void populateMemrefCopyToLinalgCopyPatterns(RewritePatternSet &patterns)
{
    patterns.add<MemrefCopyToLinalgCopyRewritePattern>(patterns.getContext());
}

} // namespace catalyst
