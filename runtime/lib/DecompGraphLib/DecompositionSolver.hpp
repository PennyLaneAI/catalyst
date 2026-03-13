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

    /**
     * @brief Solves for the given operator node and returns the chosen decomposition rule.
     */
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

    /**
     * @brief Evaluates the given decomposition rule and returns the resulting chosen
     * decomposition rule.
     */
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

    /**
     * @brief Finds the best decomposition rule for the given operator node by evaluating
     * all applicable rules and selecting the one with the lowest total cost.
     */
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
    /**
     * @brief Constructs a DecompositionSolver with the given decomposition graph.
     *
     * @param _graph The decomposition graph to be solved.
     */
    explicit DecompositionSolver(const DecompositionGraph &_graph) : graph(_graph) {}

    /**
     * @brief Solves the graph decomposition problem for the given decomposition graph
     * and returns the result.
     *
     * This method initiates the recursive solving process starting from the root operators
     * in the decomposition graph. It uses memoization to store already solved operators
     * and their chosen decomposition rules to avoid redundant computations. The result includes
     * the mapping from operator nodes to their chosen decomposition rules, as well as the list
     * of solved root nodes.
     *
     * @return Core::GraphResult The result of the graph decomposition, including the optimized
     * mapping from operator nodes to their chosen decomposition rules and the list of solved
     * root nodes.
     */
    [[nodiscard]] Core::GraphResult solve();

    // helper methods
    // TODO(Ali): move them to private/protected after testing

    /**
     * @brief Solves for the given operator node and returns the chosen decomposition rule.
     *
     * @param op The operator node to solve for.
     * @return Core::ChosenDecompRule The chosen decomposition rule for the given operator node
     * as determined by the graph solver. This includes whether the operator is a basis gate,
     * the name of the chosen rule, the chosen input operators for the rule,
     * the total cost of the decomposition, and the counts of basis gates
     * used in the decomposition.
     *
     */
    Core::ChosenDecompRule solveOperator(const Core::OperatorNode &op);

    /**
     * @brief Collects the closure of the given operator node in the result.
     *
     * This method recursively collects the chosen decomposition rules for
     * the given operator node and all of its descendant operator nodes in
     * the decomposition graph, and populates the optimizedMap in the result
     * with these mappings.
     *
     * @param op The operator node for which to collect the closure.
     * @param result The GraphResult object to populate with the optimized
     * mapping from operator nodes to their chosen decomposition rules.
     */
    void collectClosure(const Core::OperatorNode &op, Core::GraphResult &result);
};

} // namespace DecompGraph::Solver
