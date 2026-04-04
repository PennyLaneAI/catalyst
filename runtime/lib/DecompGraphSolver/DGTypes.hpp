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
 * @file DGTypes.hpp
 *
 * @brief This file defines the core data structures for representing operators and rules
 * in the decomposition framework.
 */
#pragma once

#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace DecompGraph::Core {

////////////////////////
// Operators and Gateset
////////////////////////

/**
 * @brief This represents the operator nodes in the graph decomposition problem.
 *
 * The OperatorNode struct captures the essential information about an operator, including
 * its name, the number of wires it acts on, and the number of parameters it has.
 * This information is crucial for the graph decomposition solver to determine how operators
 * can be combined and decomposed to achieve the desired target gateset while optimizing for
 * resource usage.
 */
struct OperatorNode {
    std::string name;
    std::size_t numWires{0};
    std::size_t numParams{0};
    bool adjoint{false};

    bool operator==(const OperatorNode &other) const
    {
        return name == other.name && adjoint == other.adjoint;
    }
    bool operator!=(const OperatorNode &other) const { return !(*this == other); }
};

/**
 * @brief A hash function for OperatorNode to be used in unordered containers.
 *
 * This struct provides a custom hash function for OperatorNode, allowing it to be used as
 * a key in unordered maps or sets. The hash is computed based on the name, number of wires,
 * and number of parameters.
 */
struct OperatorNodeHash {
    std::size_t operator()(const OperatorNode &node) const
    {
        std::size_t h1 = std::hash<std::string>{}(node.name);
        // TODO: commented out when we capture numWires and numParams for the target gateset
        // std::size_t h2 = std::hash<std::size_t>{}(node.numWires);
        // std::size_t h3 = std::hash<std::size_t>{}(node.numParams);
        std::size_t h4 = std::hash<bool>{}(node.adjoint);

        // Combine the hash values
        return h1 ^ (h4 << 3);
    }
};

/**
 * @brief This represents the weighted target gateset for the graph decomposition problem.
 */
struct WeightedGateset {
    std::unordered_map<OperatorNode, double, OperatorNodeHash> ops;

    [[nodiscard]] bool contains(const OperatorNode &op) const { return ops.find(op) != ops.end(); }
    [[nodiscard]] double getCost(const OperatorNode &op) const
    {
        auto it = ops.find(op);
        if (it != ops.end()) {
            return it->second;
        }

        return std::numeric_limits<double>::infinity();
    }
};

///////////////////////////
// Rules and Decompositions
///////////////////////////

/**
 * @brief This represents a term in decomposition rules,
 * which includes an operator and its multiplicity.
 */
struct RuleTerm {
    OperatorNode op;
    std::size_t multiplicity{1};
};

/**
 * @brief This represents the origin of a decomposition rule.
 *
 * This enum is used to categorize decomposition rules based on their source or type:
 * - Default: The default rule for decomposing an operator as defined in the decomposition graph.
 * - Fixed: A fixed rule that cannot be changed or overridden by the solver.
 * - Alternative: An alternative rule that can be used in place of the default rule.
 */
enum class RuleOrigin : uint8_t { Default = 0, Fixed = 1, Alternative = 2 };

/**
 * @brief This represents the decomposition rules in the graph decomposition problem.
 *
 * The RuleNode struct captures the essential information about a decomposition rule, including
 * its name, the output operator it produces, and the input operators it requires. This
 * information is crucial for the graph decomposition solver to determine how to apply
 * decomposition rules to break down complex operators into simpler ones that are part of
 * the target gateset.
 *
 * TODO:
 * - We can add a field for work_wires_required if we want to consider the number of ancillary
 * wires needed for the decomposition, which can be an important factor in resource optimization.
 * - We can also consider adding a field for the decomposition function or a pointer to it,
 * which can be used to actually perform the decomposition after the graph solver selects
 * the rules.
 */
struct RuleNode {
    std::string name;
    OperatorNode output;
    std::vector<RuleTerm> inputs;
    RuleOrigin origin{RuleOrigin::Default};

    bool operator==(const RuleNode &other) const
    {
        return name == other.name && output == other.output && origin == other.origin;
    }
};

/**
 * @brief This represents the mapping from operators to their fixed decomposition rules,
 * which are rules that cannot be changed or overridden by the solver.
 */
using FixedDecomps = std::unordered_map<OperatorNode, RuleNode, OperatorNodeHash>;

/**
 * @brief This represents the mapping from operators to their alternative decomposition rules,
 * which are rules that can be used in place of the default rule.
 */
using AltDecomps = std::unordered_map<OperatorNode, std::vector<RuleNode>, OperatorNodeHash>;

/**
 * @brief This represents the chosen decomposition rule for an operator in
 * the solution of the graph decomposition problem.
 */
struct ChosenDecompRule {
    OperatorNode op;
    bool isBasis{false};
    std::string ruleName;
    std::vector<RuleTerm> inputs;
    double totalCost{0.0};
    std::unordered_map<OperatorNode, std::size_t, OperatorNodeHash> basisCounts;
};

/**
 * @brief This represents the result of the graph decomposition, which includes the mapping
 * from operator nodes to their chosen decomposition rules, as well as the list of solved root nodes
 * in the graph.
 */
struct GraphResult {
    std::unordered_map<OperatorNode, ChosenDecompRule, OperatorNodeHash> optimizedMap;
    std::vector<OperatorNode> solvedRoots;
};

} // namespace DecompGraph::Core
