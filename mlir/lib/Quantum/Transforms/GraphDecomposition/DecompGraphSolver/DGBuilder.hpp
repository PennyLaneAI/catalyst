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
 * @brief This file implements the DecompositionGraph class, which constructs a graph
 * representation of the decomposition problem based on the provided operators, gateset,
 * and rules. The graph is built using the Boost Graph Library, where vertices represent
 * operators and rules, and edges represent the relationships between them according to
 * the decomposition rules.
 */

#pragma once

#include <memory>
#include <vector>

#include "DGTypes.hpp"

namespace DecompGraph::Solver {

class DecompositionGraph {
  private:
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    using RuleId = std::size_t;

    /**
     * @brief Constructs the decomposition graph from the given operators, gateset, and rules.
     *
     * The constructor initializes the graph by registering all operators and rules, creating
     * vertices for them, and adding edges according to the decomposition rules. The graph is
     * built in such a way that it can be efficiently traversed by the decomposition solver
     * to find optimal decompositions of the target gates.
     *
     * @param operators The list of operators that are used in the decomposition rules and gateset.
     * @param gateset The target gateset that we want to decompose into, along with their
     * associated costs.
     * @param rules The list of decomposition rules that define how operators can be decomposed
     * into other operators.
     * @param fixedDecomps The mapping from operators to their fixed decomposition rules, which are
     * rules that cannot be changed or overridden by the solver. Default is an empty map.
     * @param altDecomps The mapping from operators to their alternative decomposition rules, which
     * are rules that can be used in place of the default rule. Default is an empty map.
     */
    DecompositionGraph(std::vector<Core::OperatorNode> operators, Core::WeightedGateset gateset,
                       std::vector<Core::RuleNode> rules, Core::FixedDecomps fixedDecomps = {},
                       Core::AltDecomps altDecomps = {});
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
     * @brief Returns the mapping from operators to their fixed decomposition rules,
     * which are rules that cannot be changed or overridden by the solver.
     */
    [[nodiscard]] const Core::FixedDecomps &getFixedDecomps() const noexcept;

    /**
     * @brief Returns the mapping from operators to their alternative decomposition rules,
     * which are rules that can be used in place of the default rule.
     */
    [[nodiscard]] const Core::AltDecomps &getAltDecomps() const noexcept;

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

    /**
     * @brief Prints the graph structure for debugging purposes.
     *
     * This method can be used to visualize the graph structure, including the operators, rules,
     * and their relationships. It can help in understanding how the graph is constructed and how
     * the decomposition rules are connected to the operators. The exact format of the output can
     * be designed to be human-readable and informative for debugging.
     */
    void showGraph() const;
};

} // namespace DecompGraph::Solver
