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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace catalyst;

namespace {
struct DisableAssertionRewritePattern : public mlir::OpRewritePattern<AssertionOp> {
    using mlir::OpRewritePattern<AssertionOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AssertionOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

namespace catalyst {

void populateDisableAssertionPatterns(RewritePatternSet &patterns)
{
    patterns.add<DisableAssertionRewritePattern>(patterns.getContext());
}

} // namespace catalyst
