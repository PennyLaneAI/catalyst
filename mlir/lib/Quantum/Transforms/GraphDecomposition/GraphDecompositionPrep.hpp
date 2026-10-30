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
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"

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
 * @brief Gather the op IDs used by `funcName` into `opIDs`.
 *
 * Flattening does not merge loop bodies with a dynamic trip count, which the analysis only records
 * as `varFunctionCalls`, so those bodies are visited explicitly.
 */
void collectOpIDs(const catalyst::ResourceAnalysis &analysis, llvm::StringRef funcName,
                  llvm::StringSet<> &visitedFuncs, llvm::StringSet<> &opIDs);

/**
 * @brief Find the quantum.node function in the module.
 *
 * Returns the first `func.func` with the `quantum.node` attribute, or null if none exists.
 */
mlir::func::FuncOp findQNodeFunc(mlir::ModuleOp module);

/**
 * @brief Prepare a decomposition graph from a module.
 *
 * Gathers rules for the target gateset in addition to the passed in opIDs and builds a
 * DecompositionGraph.
 *
 * Returns nullptr on failure.
 */
std::unique_ptr<DecompGraph::Solver::DecompositionGraph>
prepareGraph(mlir::ModuleOp module, llvm::ArrayRef<std::string> opIDs, const PrepOptions &opts);

} // namespace GraphDecompositionPrep
} // namespace quantum
} // namespace catalyst
