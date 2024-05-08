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
struct GEPOpRewritePattern : public mlir::OpRewritePattern<LLVM::GEPOp> {
    using mlir::OpRewritePattern<LLVM::GEPOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(LLVM::GEPOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        auto defOp = op.getBase().getDefiningOp();
        if (op.getInbounds() || (defOp && isa<LLVM::ZeroOp>(defOp))) {
            return failure();
        }
        rewriter.startOpModification(op);
        op.setInbounds(true);
        rewriter.finalizeOpModification(op);
        return success();
    }
};

} // namespace

namespace catalyst {

void populateGEPInboundsPatterns(RewritePatternSet &patterns)
{
    patterns.add<GEPOpRewritePattern>(patterns.getContext());
}

} // namespace catalyst
