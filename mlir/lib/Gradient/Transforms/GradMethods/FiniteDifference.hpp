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

#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Gradient/IR/GradientOps.h"

using namespace mlir;

namespace catalyst {
namespace gradient {

struct FiniteDiffLowering : public OpRewritePattern<GradOp> {
    using OpRewritePattern<GradOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(GradOp op, PatternRewriter &rewriter) const override;

  private:
    static void computeFiniteDiff(PatternRewriter &rewriter, Location loc, func::FuncOp gradFn,
                                  func::FuncOp callee, const std::vector<size_t> &diffArgIndices,
                                  double hValue);
};

} // namespace gradient
} // namespace catalyst
