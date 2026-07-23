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
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

//===----------------------------------------------------------------------===//
// Catalyst dialect declarations.
//===----------------------------------------------------------------------===//

#include "Catalyst/IR/CatalystOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Catalyst discardable attributes.
//===----------------------------------------------------------------------===//

namespace catalyst {

// Resource-estimation hint on an `scf.if`: the probability (in [0, 1]) that the
// condition is true, i.e. that the "then" branch is taken. Used to compute
// expected (probability-weighted) resource counts.
inline constexpr llvm::StringRef EstimatedProbabilityAttrName = "catalyst.estimated_probability";

// Resource-estimation hint on an `scf.index_switch`: an array of probabilities
// (each in [0, 1]), one per case region in case order. The default region's
// probability is the remaining mass (1 - sum). The entries must sum to at most
// one.
inline constexpr llvm::StringRef EstimatedProbabilitiesAttrName =
    "catalyst.estimated_probabilities";

} // namespace catalyst

//===----------------------------------------------------------------------===//
// Catalyst type declarations.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Catalyst/IR/CatalystOpsTypes.h.inc"
