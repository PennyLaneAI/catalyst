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

#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h" // for DecomposableGate interface
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "Catalyst/Analysis/ResourceInterfaces.h"

//===----------------------------------------------------------------------===//
// QRef interface declarations.
//===----------------------------------------------------------------------===//

namespace catalyst {
namespace qref {

/// The arguments of a DecomposableGate, in operand order.
///
/// Operand order is the only positional order of an operator's arguments: the frontend lowering
/// (frontend/catalyst/from_plxpr/operand_layout.py) and the decomposition rules both lay out their
/// operands in it. Names are used only for identity, so they are sorted only when printed into a
/// graphOpId (see `defaultGetGraphOpId`). Do not use a hash or sorted map for these.
using DynamicShape = llvm::SmallVector<std::pair<std::string, llvm::SmallVector<mlir::Type>>>;
using WireLens = llvm::SmallVector<std::pair<std::string, size_t>>;

std::string defaultGetGraphOpId(mlir::Operation *op);

} // namespace qref
} // namespace catalyst

#include "QRef/IR/QRefInterfaces.h.inc"
