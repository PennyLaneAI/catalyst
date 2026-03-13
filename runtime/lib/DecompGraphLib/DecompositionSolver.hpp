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
 * @file DecompositionSolver.hpp
 */

#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Exception.hpp"

#include "DecompositionGraph.hpp"
#include "QuantumNodes.hpp"

namespace DecompGraph::Solver {

class DecompositionSolver {
  private:
    const DecompositionGraph &graph;

    std::unordered_map<Core::OperatorNode, Core::ChosenDecompRule, Core::OperatorNodeHash>
        solvedMap{};
    std::unordered_set<Core::OperatorNode, Core::OperatorNodeHash> visited{};
    std::vector<Core::OperatorNode> solvingStack{};

    Core::ChosenDecompRule basisRule(const Core::OperatorNode &op)
    {
        RT_ASSERT(graph.isTargetGate(op) && "Operator is not a target gate in the gateset");
        Core::ChosenDecompRule solution;
        solution.op = op;
        solution.isBasis = true;
        solution.totalCost = graph.getGateset().getCost(op);
        solution.basisCounts.emplace(op, 1);
        return solution;
    }

    Core::ChosenDecompRule evalRule(const Core::RuleNode &rule)
    {
        Core::ChosenDecompRule solution;
        solution.chosenRuleName = rule.name;
        solution.isBasis = false;
        solution.chosenInputs = rule.inputs;
        solution.op = rule.output;

        double total_cost = 0.0;
        for (const auto &input : rule.inputs) {
            const auto child = solveOperator(input.op);
            total_cost += child.totalCost * static_cast<double>(input.multiplicity);
            for (const auto &[basis_op, count] : child.basisCounts) {
                solution.basisCounts[basis_op] += count * input.multiplicity;
            }
        }

        solution.totalCost = total_cost;
        return solution;
    }

    Core::ChosenDecompRule bestRule(const Core::OperatorNode &op)
    {
        const auto &all_rules = graph.getAllRulesFor(op);
        if (all_rules.empty()) {
            RT_FAIL("No decomposition rule found for operator");
        }

        std::optional<Core::ChosenDecompRule> best_rule;

        for (const auto &rule : all_rules) {
            auto candidate = evalRule(rule);
            if (!best_rule.has_value() || candidate.totalCost < best_rule->totalCost) {
                best_rule = std::move(candidate);
            }
        }

        if (!best_rule.has_value()) {
            RT_FAIL("No valid decomposition rule found for operator");
        }

        return best_rule.value();
    }

  public:
    explicit DecompositionSolver(const DecompositionGraph &_graph) : graph(_graph) {}

    [[nodiscard]] Core::GraphResult solve();

    // helper methods
    // TODO(Ali): move them to private/protected after testing
    Core::ChosenDecompRule solveOperator(const Core::OperatorNode &op);
    void collectClosure(const Core::OperatorNode &op, Core::GraphResult &result);
};

} // namespace DecompGraph::Solver