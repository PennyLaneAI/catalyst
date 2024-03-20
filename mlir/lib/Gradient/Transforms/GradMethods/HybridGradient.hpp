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

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Gradient/IR/GradientOps.h"

namespace catalyst {
namespace gradient {

/// A pattern responsible for common transformations required when differentiating hybrid circuits
/// with Enzyme.

// grad lowering
struct HybridGradientLowering : public mlir::OpRewritePattern<GradOp> {
    using OpRewritePattern<GradOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(GradOp op, mlir::PatternRewriter &rewriter) const override;
};

// value_and_grad lowering
struct HybridValueAndGradientLowering : public mlir::OpRewritePattern<ValueAndGradOp> {
    using OpRewritePattern<ValueAndGradOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(ValueAndGradOp op,
                                        mlir::PatternRewriter &rewriter) const override;
};

} // namespace gradient
} // namespace catalyst
