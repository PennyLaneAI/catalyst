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
 * @file DGSolver.hpp
 *
 * @brief This file defines the DecompositionSolver class, which implements the graph decomp
 * algorithm to find optimal decompositions of quantum operators based on a given decomp graph.
 * The solver uses a recursive approach with memoization to efficiently explore the decomposition
 * rules and find the best decomposition for each operator node in the graph. The result includes
 * the mapping from operator nodes to their chosen decomposition rules, as well as the list of
 * solved root nodes in the graph.
 */

#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DGBuilder.hpp"
#include "DGTypes.hpp"
#include "DGUtils.hpp"

namespace DecompGraph::Solver {

class DecompositionSolver {
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
    Core::GraphResult solve();

  private:
    const DecompositionGraph &graph;

    std::unordered_map<Core::OperatorNode, Core::ChosenDecompRule, Core::OperatorNodeHash>
        solvedMap{};
    std::unordered_set<Core::OperatorNode, Core::OperatorNodeHash> visited{};
    std::vector<Core::OperatorNode> solvingStack{};

    /**
     * @brief basisRule constructs a ChosenDecompRule for a target gate operator,
     * which is a valid decomposition rule that represents the operator itself as
     * a basis gate in the target gateset.
     *
     * @param op The operator node to solve for.
     * @return Core::ChosenDecompRule
     */
    Core::ChosenDecompRule basisRule(const Core::OperatorNode &op);

    /**
     * @brief Evaluates the given decomposition rule and returns the resulting chosen
     * decomposition rule.
     *
     * This method recursively solves for the input operators of the given rule,
     * calculates the total cost of the decomposition by summing the costs of the input
     * operators according to the target gateset, and aggregates the basis gate counts
     * from the input operators. If any of the input operators cannot be solved
     * (i.e., they do not have a valid decomposition rule), this method returns
     * an invalid ChosenDecompRule with an empty rule name.
     *
     * @param rule The decomposition rule to evaluate.
     * @return Core::ChosenDecompRule The resulting chosen decomposition rule after evaluating
     * the given rule.
     */
    Core::ChosenDecompRule evalRule(const Core::RuleNode &rule);

    /**
     * @brief Finds the best decomposition rule for the given operator node by evaluating
     * all applicable rules and selecting the one with the lowest total cost.
     *
     * This method retrieves all decomposition rules that can decompose the given operator node
     * from the decomposition graph, evaluates each rule using the evalRule method, and keeps
     * track of the best valid rule (i.e., the one with the lowest total cost
     * that can successfully decompose the operator). If no valid rules are found, this method
     * returns an invalid ChosenDecompRule with an empty rule name.
     *
     * @param op The operator node to find the best decomposition rule for.
     * @return Core::ChosenDecompRule The best chosen decomposition rule for the given operator
     * node, or an invalid ChosenDecompRule if no valid rules are found.
     */
    Core::ChosenDecompRule bestRule(const Core::OperatorNode &op);

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
     * @brief Returns an invalid ChosenDecompRule for the given operator node, which indicates
     * that no valid decomposition rule could be found for the operator. This is used as a
     * sentinel value to indicate failure in finding a valid decomposition rule during the solving
     * process.
     *
     * @param op The operator node for which to return an invalid ChosenDecompRule.
     * @return Core::ChosenDecompRule An invalid ChosenDecompRule.
     */
    [[nodiscard]] inline Core::ChosenDecompRule invalidRule(const Core::OperatorNode &op) {
        return {op, false, "", {}, 0.0, {}};
    }

    /**
     * @brief Checks if the given ChosenDecompRule is invalid, which is determined by whether the
     * rule name is empty. An invalid ChosenDecompRule indicates that no valid decomposition rule
     * could be found for the operator during the solving process.
     */
    [[nodiscard]] inline bool isInvalidRule(const Core::ChosenDecompRule &rule) const {
        return rule.ruleName.empty();
    }
};

} // namespace DecompGraph::Solver
