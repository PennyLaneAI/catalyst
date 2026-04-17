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

#include <optional>

#include "mlir/IR/Value.h"

namespace catalyst {

/// Recursively resolve a Value to a numeric constant by walking through
/// common MLIR op chains (arith.constant, arith.index_cast, arith.addi/subi/muli,
/// arith.addf, tensor.extract, tensor.from_elements,
/// stablehlo.constant/convert/broadcast_in_dim/add/subtract, etc.).
///
/// Returns std::nullopt if the value cannot be statically resolved.
std::optional<double> resolveConstant(mlir::Value val);

/// Convenience wrapper that resolves to int64_t by truncating the result.
/// Returns std::nullopt if the value cannot be resolved.
std::optional<int64_t> resolveConstantInt(mlir::Value val);

} // namespace catalyst
