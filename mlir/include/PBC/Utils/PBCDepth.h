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
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"

namespace mlir {
class Block;
}

namespace catalyst::pbc {

// Compute the exact PBC depth of the resolved, predicated quantum dependency graph.
// Preserve the qubit-only PBCLayer metric: classical conditions and memory operations
// do not introduce synchronization or contribute additional layers.
// Calls compose per-qubit max-plus transfers. Large loops with invariant quantum
// graphs use exponentiation; bounded loops may resolve iteration-dependent wires.
// Measurement-controlled correction blocks with an identity alternative reserve
// their conditional instruction slots, just like explicitly conditional PBC ops.
// This is circuit depth, not the duration of an individual measurement trajectory.
// Returns nullopt for unresolved quantum indices, branch choices or iteration counts,
// unsupported IR, integer overflow, exhausted analysis budget, or no reachable PBC.
std::optional<int64_t>
computeQubitDisjointDepthTotal(mlir::Block *block, llvm::ArrayRef<mlir::Attribute> arguments = {});

} // namespace catalyst::pbc
