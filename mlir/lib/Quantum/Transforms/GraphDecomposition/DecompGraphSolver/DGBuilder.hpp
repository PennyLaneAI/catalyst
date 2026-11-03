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
 * @file DGBuilder.hpp
 *
 * @brief This file implements the DecompositionGraph class, which holds the decomposition problem
 * defined by the provided operators, gateset, and rules. Operators are the graph's nodes and each
 * rule is a hyperedge from its input operators to its output operator; the graph is stored as a
 * map from each operator to the rules that decompose it, which is all the solver needs to traverse
 * it.
 */

#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include "DGTypes.hpp"
#include "DGUtils.hpp"

namespace DecompGraph::Solver {

class DecompositionGraph {
  private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    // Debug printer, declared in DGUtils.hpp beside `showSolution`; a friend because it reports
    // the graph's internals.
    friend void Core::showGraph(const DecompositionGraph &graph, std::ostream &os);

  public:
    using RuleId = std::size_t;

    /**
     * @brief Constructs the decomposition graph from the given operators, gateset, and rules.
     *
     * The constructor registers every operator (the given ones and those appearing in rules) and
     * indexes the rules by the operator they decompose, so the solver can look up an operator's
     * rules directly.
     *
     * @param operators The list of operators that are used in the decomposition rules and gateset.
     * @param gateset The target gateset that we want to decompose into, along with their
     * associated costs.
     * @param rules The list of decomposition rules that define how operators can be decomposed
     * into other operators.
     */
    DecompositionGraph(std::vector<Core::OperatorNode> operators, Core::WeightedGateset gateset,
                       std::vector<Core::RuleNode> rules);
    ~DecompositionGraph();

    // copy and move constructors and assignment operators
    DecompositionGraph(const DecompositionGraph &other);
    DecompositionGraph(DecompositionGraph &&other) noexcept;
    DecompositionGraph &operator=(const DecompositionGraph &other);
    DecompositionGraph &operator=(DecompositionGraph &&other) noexcept;

    /**
     * @brief Returns the list of root operators in the graph, which are the operators
     * that we want to decompose into the target gateset.
     *
     * These are typically the operators that appear as outputs in the decomposition rules
     * and are the starting points for the decomposition process. The solver will attempt to find
     * optimal decompositions for these root operators based on the provided rules and gateset.
     *
     * @return The list of root operators in the graph.
     */
    [[nodiscard]] const std::vector<Core::OperatorNode> &getRootOps() const noexcept;

    /**
     * @brief Returns the target gateset for the graph decomposition problem, which includes
     * the operators that we want to decompose into and their associated costs.
     *
     * @return The target gateset for the graph decomposition problem.
     */
    [[nodiscard]] const Core::WeightedGateset &getGateset() const noexcept;

    /**
     * @brief Returns the list of decomposition rules for the graph decomposition problem,
     * which define how operators can be decomposed into other operators. Each rule includes
     * the output operator it produces and the input operators it requires, along with their
     * multiplicities.
     *
     * @return The list of decomposition rules for the graph decomposition problem.
     */
    [[nodiscard]] const std::vector<Core::RuleNode> &getRules() const noexcept;

    /**
     * @brief Returns the number of decomposition rules in the graph.
     */
    std::size_t getNumRules() const;

    /**
     * @brief Returns the number of unique operators in the graph.
     */
    std::size_t getNumOperators() const;

    // helper methods
    // TODO(Ali): move them to private/protected after testing

    /**
     * @brief Returns the decomposition rule corresponding to the given rule ID.
     */
    const Core::RuleNode &getRule(RuleId id) const;

    /**
     * @brief Returns all decomposition rules that can decompose the given operator node.
     */
    const std::vector<Core::RuleNode> &getAllRulesFor(const Core::OperatorNode &op) const;

    /**
     * @brief Checks if the given operator node is a target gate in the gateset.
     */
    bool isTargetGate(const Core::OperatorNode &op) const;

    /**
     * @brief Checks if the given operator node exists in the graph,
     * either as a root operator or as an operator appearing in the decomposition rules.
     */
    bool hasOperator(const Core::OperatorNode &op) const;
};

} // namespace DecompGraph::Solver
