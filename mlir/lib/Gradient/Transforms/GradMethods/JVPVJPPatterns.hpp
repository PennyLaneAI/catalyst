#pragma once

#include "mlir/IR/PatternMatch.h"
#include "Gradient/IR/GradientOps.h"

namespace catalyst {
namespace gradient {

struct JVPLoweringPattern : public mlir::OpRewritePattern<JVPOp> {
    using mlir::OpRewritePattern<JVPOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(JVPOp op, mlir::PatternRewriter &rewriter) const override;
};

struct VJPLoweringPattern : public mlir::OpRewritePattern<VJPOp> {
    using mlir::OpRewritePattern<VJPOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VJPOp op, mlir::PatternRewriter &rewriter) const override;
};


} // gradient
} // catalyst
