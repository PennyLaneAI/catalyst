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

/**
 * @file DecompositionSolver.cpp
 */

#include <algorithm>

#include "DecompositionSolver.hpp"

namespace DecompGraph::Solver {

Core::ChosenDecompRule DecompositionSolver::solveOperator(const Core::OperatorNode &op)
{
    // Check if the operator has already been solved
    if (const auto it = solvedMap.find(op); it != solvedMap.end()) {
        return it->second;
    }

    if (visited.find(op) != visited.end()) {
        RT_FAIL("Cycle detected in the decomposition graph");
    }


    // RAII guard for the visited set to check/solve the graph recursively
    struct VisitGuard {
        std::unordered_set<Core::OperatorNode, Core::OperatorNodeHash> &visited_;
        std::vector<Core::OperatorNode> &solvingStack_;
        const Core::OperatorNode &currentNode_;

        explicit VisitGuard(
            std::unordered_set<Core::OperatorNode, Core::OperatorNodeHash> &visited,
            std::vector<Core::OperatorNode> &solvingStack,
            const Core::OperatorNode &node) : visited_(visited), solvingStack_(solvingStack), currentNode_(node)
        {
            visited_.insert(currentNode_); // add to visited in case of exceptions
            solvingStack_.push_back(currentNode_); // push to stack in case of exceptions
        }
        ~VisitGuard() { 
            visited_.erase(currentNode_);
            if (!solvingStack_.empty()) {
                solvingStack_.pop_back();
            }
        }

        VisitGuard(const VisitGuard &) = delete;
        VisitGuard &operator=(const VisitGuard &) = delete;
    } visitGuard(visited, solvingStack, op);

    auto chosen = graph.isTargetGate(op) ? basisRule(op) : bestRule(op);
    if (!chosen.ruleName.empty()) {
        solvedMap.emplace(op, chosen);
    }
    return chosen;
}

Core::GraphResult DecompositionSolver::solve()
{
    Core::GraphResult result;

    result.solvedRoots = graph.getRoots();

    for (const auto &root : result.solvedRoots) {
        const auto chosen_rule = solveOperator(root);
        if (chosen_rule.ruleName.empty()) {
            RT_FAIL("Failed to solve the root operator");
        }
    }

    for (const auto& [op, entry] : solvedMap) {
        result.optimizedMap.emplace(op, entry);
    }

    return result;
}

} // namespace DecompGraph::Solver
