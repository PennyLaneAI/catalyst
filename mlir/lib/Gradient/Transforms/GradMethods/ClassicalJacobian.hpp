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

using namespace mlir;

namespace catalyst {
namespace gradient {

func::FuncOp genParamCountFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee);

/// Generate a version of the QNode that writes gate parameters to a buffer before calling a
/// modified QNode that explicitly accepts preprocessed gate parameters.
func::FuncOp genSplitPreprocessed(PatternRewriter &rewriter, Location loc, func::FuncOp qnode,
                                  func::FuncOp qnodeWithParams);

func::FuncOp genArgMapFunction(PatternRewriter &rewriter, Location loc, func::FuncOp callee);

} // namespace gradient
} // namespace catalyst
