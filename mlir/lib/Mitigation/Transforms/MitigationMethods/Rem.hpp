#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Mitigation/IR/MitigationOps.h"

using namespace mlir;

namespace catalyst {
namespace mitigation {

struct RemLowering : public OpRewritePattern<mitigation::RemOp> {
    using OpRewritePattern<mitigation::RemOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mitigation::RemOp op, PatternRewriter &rewriter) const override;
};

// Add patterns from this file into a pattern set.
void populateRemLoweringPatterns(RewritePatternSet &patterns);

} // namespace mitigation
} // namespace catalyst