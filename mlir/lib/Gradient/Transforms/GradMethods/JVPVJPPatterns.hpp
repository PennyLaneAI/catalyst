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

#pragma once

#include "Gradient/IR/GradientOps.h"
#include "mlir/IR/PatternMatch.h"

namespace catalyst {
namespace gradient {

struct JVPLoweringPattern : public mlir::OpRewritePattern<JVPOp> {
    using mlir::OpRewritePattern<JVPOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(JVPOp op, mlir::PatternRewriter& rewriter) const override;
};

struct VJPLoweringPattern : public mlir::OpRewritePattern<VJPOp> {
    using mlir::OpRewritePattern<VJPOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VJPOp op, mlir::PatternRewriter& rewriter) const override;
};

} // namespace gradient
} // namespace catalyst
