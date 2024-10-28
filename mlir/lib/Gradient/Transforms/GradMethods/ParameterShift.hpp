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

#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Gradient/IR/GradientOps.h"

constexpr double PI = llvm::numbers::pi;

using namespace mlir;

namespace catalyst {
namespace gradient {

struct ParameterShiftLowering : public OpRewritePattern<func::FuncOp> {
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;

    LogicalResult match(func::FuncOp op) const override;
    void rewrite(func::FuncOp op, PatternRewriter &rewriter) const override;

  private:
    static std::pair<int64_t, int64_t> analyzeFunction(func::FuncOp callee);
    static func::FuncOp genShiftFunction(PatternRewriter &rewriter, Location loc,
                                         func::FuncOp callee, const int64_t numShifts,
                                         const int64_t loopDepth);
    static func::FuncOp genQGradFunction(PatternRewriter &rewriter, Location loc,
                                         func::FuncOp callee, func::FuncOp shiftedFn,
                                         const int64_t numShifts, const int64_t loopDepth);
};

} // namespace gradient
} // namespace catalyst
