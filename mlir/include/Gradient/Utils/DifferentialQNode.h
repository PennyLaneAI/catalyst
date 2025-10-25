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

#include "llvm/ADT/StringRef.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace catalyst {
namespace gradient {

/// Check if the given func op represents a QNode.
bool isQNode(mlir::func::FuncOp funcOp);

/// Get the differentiation method specified on the QNode, or an empty StringRef if `funcOp` is
/// either not a QNode or does not have a differentiation method.
llvm::StringRef getQNodeDiffMethod(mlir::func::FuncOp funcOp);

/// Mark this `funcOp` representing a QNode as requiring the generation and registry of a custom
/// gradient to compute the derivatives of its quantum functions.
void setRequiresCustomGradient(mlir::func::FuncOp funcOp, mlir::FlatSymbolRefAttr pureQuantumFunc);

/// Check if this `funcOp` requires the generation and registration of a custom gradient.
bool requiresCustomGradient(mlir::func::FuncOp funcOp);

/// Register a custom quantum gradient for the given QNode.
void registerCustomGradient(mlir::func::FuncOp qnode, mlir::FlatSymbolRefAttr qgradFn);

} // namespace gradient
} // namespace catalyst
