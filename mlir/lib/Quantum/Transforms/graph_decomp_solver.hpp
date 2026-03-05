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
#include <string>

#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
namespace quantum {

struct OperatorNode {
    std::string gateName;
    bool adjoint = false;
    size_t numQubits = 0;
    size_t numCtrlQubits = 0;
    size_t numParams = 0;
};

struct RuleNode {
    StringRef name;
    func::FuncOp funcOp;
    DictionaryAttr resource;
};

struct GraphDecompositionSolver {
    
    /** 
     * @brief Solve the graph decomposition problem given the operators, resources, and target gateset.
     * 
     */
    static std::vector<RuleNode> Solve(const std::vector<OperatorNode> &operators, const std::vector<RuleNode> &rules, const llvm::StringSet<llvm::MallocAllocator> &gateset)
    {
        return {};
    }
};

} // namespace quantum
} // namespace catalyst