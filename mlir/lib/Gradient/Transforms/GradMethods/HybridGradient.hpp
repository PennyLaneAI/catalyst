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
struct HybridGradientLowering : public mlir::OpRewritePattern<GradOp> {
    using OpRewritePattern<GradOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(GradOp op, mlir::PatternRewriter &rewriter) const override;

  private:
    /// Recursively process all the QNodes of the `callee` being differentiated. The resulting
    /// BackpropOps will be called with `backpropArgs`.
    static mlir::FailureOr<mlir::func::FuncOp>
    cloneCallee(mlir::PatternRewriter &rewriter, GradOp gradOp, mlir::func::FuncOp callee,
                mlir::SmallVectorImpl<Value> &backpropArgs);

    /// Generate a version of the QNode that accepts the parameter buffer. This is so Enzyme will
    /// see that the gate parameters flow into the custom quantum function.
    static mlir::func::FuncOp genQNodeQuantumOnly(mlir::PatternRewriter &rewriter,
                                                  mlir::Location loc, mlir::func::FuncOp qnode);

    /// Generate a function that computes a Jacobian row-by-row using one or more BackpropOps.
    static mlir::func::FuncOp genFullGradFunction(mlir::PatternRewriter &rewriter,
                                                  mlir::Location loc, GradOp gradOp,
                                                  mlir::func::FuncOp callee, FunctionType fnType);
};

} // namespace gradient
} // namespace catalyst
