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

#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace quantum {

struct OperatorNode {
    CustomOp op;
    StringRef name;
    float weight;
};

struct RuleNode {
    StringRef name;
    mlir::OwningOpRef<func::FuncOp> funcOp;
    DictionaryAttr resource;
};

struct GraphDecompositionSolver {

    /**
     * @brief Solve the graph decomposition problem given the operators, resources, and target
     * gateset.
     *
     * This is a placeholder for the actual graph decomposition solver implemented in gdecomp_cpp.
     *
     * @param operators The list of operator nodes representing the operations in the graph.
     * @param rules The list of rule nodes representing the decomposition rules and their resources.
     * @param gateset The target gateset for decomposition, represented as a DictionaryAttr.
     * @return A list of RuleNodes representing the selected decomposition rules to apply for the
     * graph decomposition.
     *
     */
    static std::vector<RuleNode> Solve(const std::vector<OperatorNode> &operators,
                                       const std::vector<RuleNode> &rules,
                                       const std::vector<llvm::StringRef> &gateset)
    {
        return {};
    }
};

} // namespace quantum
} // namespace catalyst
