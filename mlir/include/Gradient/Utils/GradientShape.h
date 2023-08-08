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

#include <optional>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace catalyst {
namespace gradient {

bool isDifferentiable(mlir::Type type);

std::vector<mlir::Type> computeResultTypes(mlir::func::FuncOp callee,
                                           const std::vector<size_t> &diffArgIndices);

std::vector<mlir::Type> computeQGradTypes(mlir::func::FuncOp callee);

std::vector<mlir::Type> computeBackpropTypes(mlir::func::FuncOp callee,
                                             const std::vector<size_t> &diffArgIndices);

std::vector<size_t> computeDiffArgIndices(std::optional<mlir::DenseIntElementsAttr> indices);

std::vector<mlir::Value> computeDiffArgs(mlir::ValueRange args,
                                         std::optional<mlir::DenseIntElementsAttr> indices);

} // namespace gradient
} // namespace catalyst
