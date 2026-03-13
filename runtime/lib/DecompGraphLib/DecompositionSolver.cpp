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

    visited.insert(op);
    solvingStack.push_back(op);

    try {
        Core::ChosenDecompRule chosen_rule;

        if (graph.isTargetGate(op)) {
            chosen_rule = basisRule(op);
        }
        else {
            chosen_rule = bestRule(op);
        }

        solvedMap.emplace(op, chosen_rule);
        visited.erase(op);
        solvingStack.pop_back();
        return chosen_rule;
    }
    catch (...) {
        // Otherwise, we clean up the visited set and solving stack on exceptions
        // to avoid false cycle detections in future calls.
        visited.erase(op);
        solvingStack.pop_back();
        return {}; // or throw ?
    }
}

void DecompositionSolver::collectClosure(const Core::OperatorNode &op, Core::GraphResult &result)
{
    if (result.optimizedMap.find(op) != result.optimizedMap.end()) {
        return; // already in closure
    }

    const auto chosen_rule = solvedMap.find(op);
    if (chosen_rule == solvedMap.end()) {
        return; // FIXME: this should not happen!
    }

    result.optimizedMap.emplace(op, chosen_rule->second);
    if (chosen_rule->second.isBasis) {
        return; // basis case, stop recursion
    }

    for (const auto &input : chosen_rule->second.chosenInputs) {
        collectClosure(input.op, result);
    }
}

Core::GraphResult DecompositionSolver::solve()
{
    Core::GraphResult result;

    result.solvedRoots = graph.getRoots();

    for (const auto &root : result.solvedRoots) {
        (void)solveOperator(root);
    }

    for (const auto &root : result.solvedRoots) {
        collectClosure(root, result);
    }

    return result;
}

} // namespace DecompGraph::Solver