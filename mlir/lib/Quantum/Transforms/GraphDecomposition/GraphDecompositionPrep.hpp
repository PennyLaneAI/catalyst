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

#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "DGBuilder.hpp"

namespace catalyst {
namespace quantum {

namespace GraphDecompositionPrep {

/**
 * @brief Options shared by passes that build a decomposition graph.
 *
 * Necessary options for the preparation of a decomposition graph.
 */
struct PrepOptions {
    llvm::ArrayRef<std::string> targetGateSetOption;
    llvm::ArrayRef<std::string> fixedDecompsOption;
    llvm::ArrayRef<std::string> altDecompsOption;
    llvm::StringRef bytecodeRulesFile;
    llvm::StringRef libQPDPath;
    llvm::StringRef libpythonPath;
};

/**
 * @brief Prepare a decomposition graph from a module.
 *
 * Strip `quantum.ctrl` / `quantum.adjoint` regions
 * if needed, then gathers rules for the target gateset and builds a
 * DecompositionGraph.
 *
 * Returns nullptr on failure.
 */
std::unique_ptr<DecompGraph::Solver::DecompositionGraph> prepareGraph(mlir::ModuleOp module,
                                                                      const PrepOptions &opts);

} // namespace GraphDecompositionPrep
} // namespace quantum
} // namespace catalyst
