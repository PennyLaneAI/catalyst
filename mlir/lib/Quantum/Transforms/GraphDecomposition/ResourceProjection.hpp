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

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceResult.h"

#include "GraphPrep.hpp"

namespace catalyst {
namespace quantum {

/**
 * @brief Project the resources of each function onto the decomposition chosen for the module,
 * without applying it.
 *
 * As in the program after `graph-decomposition`, each decomposable op becomes a call to the
 * function of its chosen rule, which holds the rule's basis gates and calls the functions of the
 * rules of its other operators. The other operations are kept as they are.
 *
 * @param analysis The resource analysis of `module`, with detailed operations.
 * @return The resources of each function and rule function, or failure if the decomposition graph
 * cannot be built.
 */
mlir::FailureOr<llvm::StringMap<ResourceResult>>
projectDecomposition(mlir::ModuleOp module, const ResourceAnalysis &analysis,
                     const GraphPrep::PrepOptions &options, bool verbose = false);

} // namespace quantum
} // namespace catalyst
