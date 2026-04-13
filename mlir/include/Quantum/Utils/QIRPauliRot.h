// Copyright 2026 Xanadu Quantum Technologies Inc.

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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace catalyst {
namespace quantum {

// Create a Pauli product string ptr holding the Pauli word.
mlir::Value getPauliProductPtr(mlir::Location loc, mlir::OpBuilder &rewriter, mlir::ModuleOp mod,
                               mlir::ArrayAttr pauliProduct);

// Helper for creating a call to `__catalyst__qis__PauliRot`.
void createPauliRotCall(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
                        mlir::Operation *opForDecl, mlir::Value pauliWordPtr,
                        mlir::Value thetaValue, mlir::Value modifiersPtr, mlir::Value cond,
                        mlir::ValueRange inQubits);

} // namespace quantum
} // namespace catalyst
