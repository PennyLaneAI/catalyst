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

#include <cstddef>
#include <cstdint>
#include <functional>
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
    std::string id;
    bool adjoint{false};
    std::size_t numControlWires{0};

    // optional params, primarily for debug use
    std::string name{""};
    int numWires{-1};
    int numParams{-1};
    std::unordered_map<std::string, std::string> staticNamedArgs{};

    bool operator==(const OperatorNode &other) const { return id == other.id; }
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
        return std::hash<std::string>{}(node.id);
    }
};

/**
 * @brief This represents the weighted target gateset for the graph decomposition problem.
 */
struct WeightedGateset {
    // TODO: using ID here mandates that gatesets specify all legal IDs, rather than generic class
    // like "PauliRot". This should be updated to work on generic names
    std::unordered_map<OperatorNode, double, OperatorNodeHash> ops;

    [[nodiscard]] bool contains(const OperatorNode &op) const { return ops.find(op) != ops.end(); }

    [[nodiscard]] double getCost(const OperatorNode &op) const
    {
        auto it = ops.find(op);
        return it != ops.end() ? it->second : std::numeric_limits<double>::infinity();
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
 * - Default: The default rule for decomposing an operator as defined in the decomposition
 * graph.
 * - Fixed: A fixed rule that cannot be changed or overridden by the solver.
 * - Alternative: An alternative rule that can be used in place of the default rule.
 * - AdjointGenerated: A rule synthesized by adjointing a base decomposition rule.
 * - ControlGenerated: A rule synthesized by controlling a base decomposition rule.
 */
enum class RuleOrigin : uint8_t {
    Default = 0,
    Fixed = 1,
    Alternative = 2,
    AdjointGenerated = 3,
    ControlGenerated = 4
};

/**
 * @brief This represents the decomposition rules in the graph decomposition problem.
 *
 * The RuleNode struct captures the essential information about a decomposition rule, including
 * its name, the output operator it produces, and the input operators it requires. This
 * information is crucial for the graph decomposition solver to determine how to apply
 * decomposition rules to break down complex operators into simpler ones that are part of
 * the target gateset.
 *
 * @todo
 * - We can add a field for work_wires_required if we want to consider the number of ancillary
 * wires needed for the decomposition, which can be an important factor in resource
 * optimization.
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

namespace modifiers {

/**
 * @brief The modifiers parsed out of an operator id.
 *
 * Modifiers are serialized in one canonical form, control-outermost then adjoint:
 * `[C(<k>, ][Adjoint(]<core>[)][)]`. Because adjoint and control commute, serializing them in a
 * fixed order means any order of application yields the same id (e.g. `C(Adjoint(op))` and
 * `Adjoint(C(op))` are the same node), and control wires accumulate instead of nesting.
 */
struct Modifiers {
    std::size_t numControlWires{0};
    bool adjoint{false};
    std::string core; // the base id with all modifiers stripped
};

inline Modifiers parseModifiers(const std::string &id)
{
    Modifiers m;
    std::string s = id;

    // Strip the outermost "C(<k>, ...)" control wrapper (op ids never start with "C(", so this is
    // unambiguous). The control count is parsed by hand because the build has exceptions disabled.
    if (s.rfind("C(", 0) == 0 && !s.empty() && s.back() == ')') {
        std::size_t sep = s.find(", ", 2);
        if (sep != std::string::npos) {
            std::string kStr = s.substr(2, sep - 2);
            std::size_t k = 0;
            bool ok = !kStr.empty();
            for (char c : kStr) {
                if (c < '0' || c > '9') {
                    ok = false;
                    break;
                }
                k = k * 10 + static_cast<std::size_t>(c - '0');
            }
            if (ok) {
                m.numControlWires = k;
                s = s.substr(sep + 2, s.size() - (sep + 2) - 1);
            }
        }
    }

    // Strip the "Adjoint( ... )" wrapper.
    static constexpr char kAdj[] = "Adjoint(";
    constexpr std::size_t kAdjLen = sizeof(kAdj) - 1;
    if (s.size() > kAdjLen && s.compare(0, kAdjLen, kAdj) == 0 && s.back() == ')') {
        m.adjoint = true;
        s = s.substr(kAdjLen, s.size() - kAdjLen - 1);
    }

    m.core = s;
    return m;
}

inline std::string buildId(const Modifiers &m)
{
    std::string s = m.core;
    if (m.adjoint) {
        s = "Adjoint(" + s + ")";
    }
    if (m.numControlWires > 0) {
        s = "C(" + std::to_string(m.numControlWires) + ", " + s + ")";
    }
    return s;
}

} // namespace modifiers

/**
 * @brief This returns a copy of the operator with the adjoint modifier toggled.
 *
 * Identity is the opaque `id` string (equality/hashing are id-only), so the modifier is folded into
 * the id, re-serialized in canonical control-outermost form. The `adjoint` bool and
 * `numControlWires` count are kept in lockstep with the id. Applying twice cancels:
 * `makeAdjoint(makeAdjoint(op)) == op`.
 */
inline OperatorNode makeAdjoint(OperatorNode op)
{
    modifiers::Modifiers m = modifiers::parseModifiers(op.id);
    m.adjoint = !m.adjoint;
    op.id = modifiers::buildId(m);
    op.adjoint = m.adjoint;
    op.numControlWires = m.numControlWires;
    return op;
}

/**
 * @brief This returns a copy of the operator with `numControlWires` additional control wires.
 *
 * Controls accumulate (`C(1, C(1, op)) == C(2, op)`) and the id is re-serialized in canonical
 * control-outermost form, so control commutes with adjoint (`C(Adjoint(op)) == Adjoint(C(op))`).
 * The `numControlWires` count and `adjoint` bool are kept in lockstep with the id.
 */
inline OperatorNode makeControlled(OperatorNode op, std::size_t numControlWires = 1)
{
    modifiers::Modifiers m = modifiers::parseModifiers(op.id);
    m.numControlWires += numControlWires;
    op.id = modifiers::buildId(m);
    op.numControlWires = m.numControlWires;
    op.adjoint = m.adjoint;
    return op;
}

/**
 * @brief Constructs the Adjoint decomposition of a base rule.
 *
 * Given a rule `output -> {inputs}`, produces `Adjoint(output) -> {Adjoint(input), ...}`
 * with the same multiplicities: the adjoint of a decomposition is obtained by adjointing
 * every produced gate (and reversing their order, which does not affect resource/cost counting).
 */
inline RuleNode makeAdjointRule(const RuleNode &base)
{
    RuleNode adj;
    adj.name = base.name + "_adjoint";
    adj.output = makeAdjoint(base.output);
    adj.origin = RuleOrigin::AdjointGenerated;
    adj.inputs.reserve(base.inputs.size());
    for (const auto &term : base.inputs) {
        adj.inputs.push_back({makeAdjoint(term.op), term.multiplicity});
    }
    return adj;
}

/**
 * @brief This returns a copy of the given operator with all controls removed.
 */
inline OperatorNode withoutControls(OperatorNode op)
{
    modifiers::Modifiers m = modifiers::parseModifiers(op.id);
    m.numControlWires = 0;
    op.id = modifiers::buildId(m);
    op.numControlWires = 0;
    op.adjoint = m.adjoint;
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

    // TODO: revisit this after testing..
    RuleOrigin origin{RuleOrigin::Default};
};

/**
 * @brief This represents the result of the graph decomposition, which includes the mapping
 * from operator nodes to their chosen decomposition rules.
 */
using GraphResult =
    std::unordered_map<Core::OperatorNode, Core::ChosenDecompRule, Core::OperatorNodeHash>;

} // namespace DecompGraph::Core
