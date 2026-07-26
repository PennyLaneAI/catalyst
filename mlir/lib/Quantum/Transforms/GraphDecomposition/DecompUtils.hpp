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

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace catalyst {
namespace quantum {

/**
 * @brief Shared helpers for the graph-decomposition and decompose-lowering passes.
 *
 * A decomposition rule is a `func.func` carrying the `target_gate` attribute, naming the
 * operator it decomposes. These utilities centralize the attribute names and the queries
 * that identify such rules so the passes agree on a single definition.
 */
namespace DecompUtils {

/// Attribute marking a `func.func` as a decomposition rule for the named target gate.
inline constexpr llvm::StringRef target_gate_attr_name = "target_gate";

/// Attribute holding the target gate set a circuit function should decompose into.
inline constexpr llvm::StringRef decomp_gateset_attr_name = "decomp_gateset";

/// True if @p func is a decomposition rule (i.e. carries the `target_gate` attribute).
bool isDecompositionFunction(mlir::func::FuncOp func);

/// The target gate a decomposition rule decomposes, or an empty ref if @p func is not a rule.
llvm::StringRef getTargetGateName(mlir::func::FuncOp func);

/// The `num_wires` attribute of @p func, or 0 if absent.
uint64_t getNumWires(mlir::func::FuncOp func);

/// True if @p op is nested inside a decomposition rule function.
bool isInDecompRule(mlir::Operation *op);

} // namespace DecompUtils
} // namespace quantum
} // namespace catalyst
