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

#include <cstdint>
#include <limits>
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
 *
 * Optionally, an operator may carry a set of static named arguments (e.g.,
 * `{"pauli_word": "X"}` for `PauliRot`) that further specialize the operator.
 * When non-empty, static arguments participate in equality
 * comparisons so that rules guarded on a specific value only
 * match operator queries that carry the same value.
 *
 * TODO: Fix the equality with wildcards for numWires and numParams
 * when adding support for operators with dynamic numbers of wires/params.
 */
struct OperatorNode {
    std::string name;
    int64_t numWires{-1};  // -1 = any number of wires
    int64_t numParams{-1}; // -1 = any number of params
    bool adjoint{false};
    std::size_t numControlWires{0};

    // Optional static arguments for operators that require additional data.
    std::unordered_map<std::string, std::string> staticNamedArgs{};

    bool operator==(const OperatorNode &other) const
    {
        // For equality, we consider numWires and numParams conditionally equal
        // if they are not set to -1 (which indicates a wildcard that can match any value).
        const bool default_wires =
            (numWires == -1 || other.numWires == -1 || numWires == other.numWires);
        const bool default_params =
            (numParams == -1 || other.numParams == -1 || numParams == other.numParams);

        // Static arguments are optional: if either side has no static args, they
        // are treated as matching (wildcard). When both sides provide entries, the maps must
        // be equal element-wise for the operators to be considered equivalent.
        const bool static_args_match = staticNamedArgs.empty() || other.staticNamedArgs.empty() ||
                                       staticNamedArgs == other.staticNamedArgs;

        return name == other.name && default_wires && default_params && adjoint == other.adjoint &&
               static_args_match && numControlWires == other.numControlWires;
    }
    bool operator!=(const OperatorNode &other) const { return !(*this == other); }
};

/**
 * @brief A hash function for OperatorNode to be used in unordered containers.
 *
 * This struct provides a custom hash function for OperatorNode, allowing it to be used as
 * a key in unordered maps or sets. The hash is computed based on the name of the operator.
 *
 * Note: The hash function only considers the name of the operator for hashing, which means that
 * different OperatorNode instances with the same name but different numWires, numParams, or
 * adjoint values will have the same hash. This is intentional to allow for wildcard matching
 * based on the name, but it also means that care must be taken when using OperatorNode
 * instances (such as MultiRZ) that may have the same name but different parameters,
 * as they will be treated as the same key in hash-based containers.
 * The number of wires and parameters can be encoded in the name if needed to distinguish them
 * in the hash when converting MLIR operations to OperatorNodes.
 */
struct OperatorNodeHash {
    std::size_t operator()(const OperatorNode &node) const
    {
        return std::hash<std::string>{}(node.name);
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
 * - ControlGenerated: A rule synthesized by controlling a base decomposition rule.
 */
enum class RuleOrigin : uint8_t { Default = 0, Fixed, Alternative, ControlGenerated };

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

    bool isEmpty() const { return inputs.empty(); }
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
 * @brief This returns a copy of the given operator wrapped in `numControlWires`
 * additional controls (Controlled(op)). Control wires accumulate, so applying it
 * repeatedly yields a multi-controlled operator.
 */
inline OperatorNode makeControlled(OperatorNode op, std::size_t numControlWires = 1)
{
    op.numControlWires += numControlWires;
    return op;
}

/**
 * @brief Constructs the Controlled decomposition rule from a base rule.
 *
 * Given a rule `output -> {inputs}`, produces `Controlled(output) -> {Controlled(input), ...}`
 * where every operator gains `numControlWires` control wires: controlling a decomposition means
 * applying the same controls to each gate it produces.
 *
 * The `numControlWires` count is encoded in the rule name so distinct control counts over
 * the same base rule stay unique. The result is tagged with `RuleOrigin::ControlGenerated`
 * so later stages can lower it by controlling each gate.
 *
 * @note: PennyLane counts `PauliX` flips for zero `control_values`.
 * Those flips and `control_values` are not supported yet; the cost reflects only the
 * cost of controlling each produced gate.
 */
inline RuleNode makeControlledRule(const RuleNode &base, std::size_t numControlWires)
{
    RuleNode ctrl;
    ctrl.name = base.name + "_controlled_" + std::to_string(numControlWires);
    ctrl.output = makeControlled(base.output, numControlWires);
    ctrl.origin = RuleOrigin::ControlGenerated;
    ctrl.inputs.reserve(base.inputs.size());
    for (const auto &term : base.inputs) {
        ctrl.inputs.push_back({makeControlled(term.op, numControlWires), term.multiplicity});
    }
    return ctrl;
}

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
    RuleOrigin origin{RuleOrigin::Default};
};

/**
 * @brief This represents the result of the graph decomposition, which includes the mapping
 * from operator nodes to their chosen decomposition rules.
 */
using GraphResult =
    std::unordered_map<Core::OperatorNode, Core::ChosenDecompRule, Core::OperatorNodeHash>;

} // namespace DecompGraph::Core
