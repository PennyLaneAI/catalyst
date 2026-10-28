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
 * @file DGBuilder.cpp
 */

#include "DGBuilder.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/detail/adjacency_list.hpp"
#include "boost/graph/graph_selectors.hpp"
#include "boost/graph/graph_traits.hpp"

#include "DGTypes.hpp"
#include "DGUtils.hpp"

using namespace DecompGraph::Core;

namespace DecompGraph::Solver {

struct DecompositionGraph::Impl {
    using RuleId = DecompositionGraph::RuleId;
    using OperatorId = std::size_t;

    struct OperatorVertex {
        OperatorId op_id;
    };

    struct RuleVertex {
        RuleId rule_id;
    };

    enum class VertexType : std::uint8_t { Operator = 0, Rule = 1 };

    struct GraphVertex {
        VertexType type;
        std::variant<OperatorVertex, RuleVertex> payload;
    };

    struct GraphWeightedEdge {}; // placeholder used in the boost graph as a type

    using BbGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                                          GraphVertex, GraphWeightedEdge>;
    using Vertex = boost::graph_traits<BbGraph>::vertex_descriptor;

    BbGraph graph;
    std::vector<OperatorNode> operators;
    WeightedGateset gateset;
    std::vector<RuleNode> rules;

    std::unordered_map<OperatorNode, OperatorId, OperatorNodeHash> opToId;
    std::vector<OperatorNode> idToOp;
    std::unordered_map<OperatorId, Vertex> opIdToVertex;

    std::unordered_map<RuleId, Vertex> ruleIdToVertex;
    std::unordered_map<OperatorNode, std::vector<RuleNode>, OperatorNodeHash> opToRules;

    OperatorId registerOp(const OperatorNode &op) {
        const auto it = opToId.find(op);
        if (it != opToId.end()) {
            return it->second;
        }

        OperatorId newId = opToId.size();
        opToId.emplace(op, newId);
        idToOp.push_back(op);
        const auto vertex =
            boost::add_vertex(GraphVertex{VertexType::Operator, OperatorVertex{newId}}, graph);
        opIdToVertex.emplace(newId, vertex);
        return newId;
    }

    Impl(std::vector<OperatorNode> _operators, WeightedGateset _gateset,
         std::vector<RuleNode> _rules)
        : operators(std::move(_operators)), gateset(std::move(_gateset)), rules(std::move(_rules)) {
        materializeRules();
    }

    void materializeRules() {
        std::unordered_map<OperatorNode, std::vector<RuleNode>, OperatorNodeHash> baseByOutput;
        baseByOutput.reserve(rules.size());
        for (auto &rule : rules) {
            baseByOutput[rule.output].push_back(rule);
        }

        std::vector<RuleNode> effectiveRules;
        effectiveRules.reserve(rules.size());
        std::unordered_set<OperatorNode, OperatorNodeHash> seenOutputs;

        auto appendRulesForOutput = [&](const OperatorNode &op) {
            if (!seenOutputs.insert(op).second) {
                return;
            }

            const auto baseIt = baseByOutput.find(op);
            if (baseIt != baseByOutput.end()) {
                for (const auto &rule : baseIt->second) {
                    effectiveRules.push_back(rule);
                }
            }
        };

        for (const auto &rule : rules) {
            appendRulesForOutput(rule.output);
        }

        rules = std::move(effectiveRules);
    }

    void buildGraph() {
        // Register all operators
        for (const auto &op : operators) {
            registerOp(op);
        }

        // Register all rules
        for (RuleId ruleId = 0; ruleId < rules.size(); ruleId++) {
            const auto &rule = rules[ruleId];
            const auto id = registerOp(rule.output);
            const auto output_vertex = opIdToVertex[id];

            // Create a vertex for the rule and connect it to its output operator vertex
            const auto rule_vertex =
                boost::add_vertex(GraphVertex{VertexType::Rule, RuleVertex{ruleId}}, graph);
            ruleIdToVertex.emplace(ruleId, rule_vertex);
            opToRules[rule.output].push_back(rule);

            // Connect rule vertex to output operator vertex
            boost::add_edge(rule_vertex, output_vertex, GraphWeightedEdge{}, graph);

            // Empty rules (with no inputs) are effectively just target gates
            // and don't need to be connected to input operator vertices
            if (rule.isEmpty()) {
                continue;
            }

            // Connect rule vertex to input operator vertices
            for (const auto &input : rule.inputs) {
                const auto input_id = registerOp(input.op);
                const auto input_vertex = opIdToVertex[input_id];
                boost::add_edge(input_vertex, rule_vertex, GraphWeightedEdge{}, graph);
            }
        }
    }
};

DecompositionGraph::DecompositionGraph(std::vector<OperatorNode> operators, WeightedGateset gateset,
                                       std::vector<RuleNode> rules)
    : impl(std::make_unique<Impl>(std::move(operators), std::move(gateset), std::move(rules))) {
    impl->buildGraph();
}

DecompositionGraph::~DecompositionGraph() = default;

DecompositionGraph::DecompositionGraph(const DecompositionGraph &other)
    : impl(std::make_unique<Impl>(*other.impl)) {}

DecompositionGraph::DecompositionGraph(DecompositionGraph &&other) noexcept = default;

DecompositionGraph &DecompositionGraph::operator=(const DecompositionGraph &other) {
    if (this != &other) {
        impl = std::make_unique<Impl>(*other.impl);
    }
    return *this;
}

DecompositionGraph &DecompositionGraph::operator=(DecompositionGraph &&other) noexcept = default;

[[nodiscard]] const std::vector<OperatorNode> &DecompositionGraph::getRootOps() const noexcept {
    return impl->operators;
}

[[nodiscard]] const WeightedGateset &DecompositionGraph::getGateset() const noexcept {
    return impl->gateset;
}

[[nodiscard]] const std::vector<RuleNode> &DecompositionGraph::getRules() const noexcept {
    return impl->rules;
}

std::size_t DecompositionGraph::getNumRules() const { return impl->rules.size(); }

std::size_t DecompositionGraph::getNumOperators() const { return impl->operators.size(); }

const RuleNode &DecompositionGraph::getRule(RuleId id) const { return impl->rules[id]; }

const std::vector<RuleNode> &DecompositionGraph::getAllRulesFor(const OperatorNode &op) const {
    static const std::vector<RuleNode> empty;
    const auto it = impl->opToRules.find(op);
    if (it != impl->opToRules.end()) {
        return it->second;
    }
    return empty;
}

bool DecompositionGraph::isTargetGate(const OperatorNode &op) const {
    return impl->gateset.contains(op);
}

bool DecompositionGraph::hasOperator(const OperatorNode &op) const {
    return impl->opToId.find(op) != impl->opToId.end();
}

} // namespace DecompGraph::Solver

namespace DecompGraph::Core {

void showGraph(const Solver::DecompositionGraph &graph, std::ostream &os) {
    const auto *impl = graph.impl.get();
    os << "Decomposition Graph:\n";
    // Show all operators by their names
    os << "Operators:\n";
    for (const auto &[op, id] : impl->opToId) {
        os << "  ID " << id << ": " << print_op(op) << "\n";
    }

    // Show all rules by their names and their input/output operators
    os << "Rules:\n";
    for (const auto &[ruleId, _] : impl->ruleIdToVertex) {
        const auto &rule = impl->rules[ruleId];
        os << "  Rule ID " << ruleId << ": " << rule.name;
        os << "\n";
        os << "    Output: " << print_op(rule.output) << "\n";
        os << "    Inputs:\n";
        for (const auto &input : rule.inputs) {
            os << "      - " << print_op(input.op) << " (multiplicity: " << input.multiplicity
               << ")\n";
        }
    }

    // Show target gateset
    os << "Target Gateset:\n";
    for (const auto &[name, cost] : impl->gateset.ops) {
        os << "  " << name << " with cost " << cost << "\n";
    }
}

} // namespace DecompGraph::Core
