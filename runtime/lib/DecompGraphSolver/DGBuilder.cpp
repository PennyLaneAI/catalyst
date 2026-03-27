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

#include <boost/graph/adjacency_list.hpp>

#include <iostream>
#include <variant>

#include "DGBuilder.hpp"
#include "DGUtils.hpp"

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
    std::vector<Core::OperatorNode> operators;
    Core::WeightedGateset gateset;
    std::vector<Core::RuleNode> rules;
    Core::FixedDecomps fixedDecomps;
    Core::AltDecomps altDecomps;

    std::unordered_map<Core::OperatorNode, OperatorId, Core::OperatorNodeHash> opToId;
    std::vector<Core::OperatorNode> idToOp;
    std::unordered_map<OperatorId, Vertex> opIdToVertex;

    std::unordered_map<RuleId, Vertex> ruleIdToVertex;
    std::unordered_map<Core::OperatorNode, std::vector<Core::RuleNode>, Core::OperatorNodeHash>
        opToRules;

    OperatorId registerOp(const Core::OperatorNode &op)
    {
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

    Impl(std::vector<Core::OperatorNode> _operators, Core::WeightedGateset _gateset,
         std::vector<Core::RuleNode> _rules, Core::FixedDecomps _fixedDecomps = {},
         Core::AltDecomps _altDecomps = {})
        : operators(std::move(_operators)), gateset(std::move(_gateset)), rules(std::move(_rules)),
          fixedDecomps(std::move(_fixedDecomps)), altDecomps(std::move(_altDecomps))
    {
    }

    void buildGraph()
    {
        // Register all operators
        for (const auto &op : operators) {
            registerOp(op);
        }

        // Register all target gates
        for (const auto &[op, _] : gateset.ops) {
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

            // Connect rule vertex to input operator vertices
            for (const auto &input : rule.inputs) {
                const auto input_id = registerOp(input.op);
                const auto input_vertex = opIdToVertex[input_id];
                boost::add_edge(input_vertex, rule_vertex, GraphWeightedEdge{}, graph);
            }
        }
    }
};

DecompositionGraph::DecompositionGraph(std::vector<Core::OperatorNode> operators,
                                       Core::WeightedGateset gateset,
                                       std::vector<Core::RuleNode> rules,
                                       Core::FixedDecomps fixedDecomps, Core::AltDecomps altDecomps)
    : impl(std::make_unique<Impl>(std::move(operators), std::move(gateset), std::move(rules),
                                  std::move(fixedDecomps), std::move(altDecomps)))
{
    impl->buildGraph();
}

DecompositionGraph::~DecompositionGraph() = default;

DecompositionGraph::DecompositionGraph(const DecompositionGraph &other)
    : impl(std::make_unique<Impl>(*other.impl))
{
}

DecompositionGraph::DecompositionGraph(DecompositionGraph &&other) noexcept = default;

DecompositionGraph &DecompositionGraph::operator=(const DecompositionGraph &other)
{
    if (this != &other) {
        impl = std::make_unique<Impl>(*other.impl);
    }
    return *this;
}

DecompositionGraph &DecompositionGraph::operator=(DecompositionGraph &&other) noexcept = default;

[[nodiscard]] const std::vector<Core::OperatorNode> &DecompositionGraph::getRoots() const noexcept
{
    return impl->operators;
}

[[nodiscard]] const Core::WeightedGateset &DecompositionGraph::getGateset() const noexcept
{
    return impl->gateset;
}

[[nodiscard]] const std::vector<Core::RuleNode> &DecompositionGraph::getRules() const noexcept
{
    return impl->rules;
}

[[nodiscard]] const Core::FixedDecomps &DecompositionGraph::getFixedDecomps() const noexcept
{
    return impl->fixedDecomps;
}

[[nodiscard]] const Core::AltDecomps &DecompositionGraph::getAltDecomps() const noexcept
{
    return impl->altDecomps;
}

std::size_t DecompositionGraph::getNumRules() const { return impl->rules.size(); }

std::size_t DecompositionGraph::getNumOperators() const { return impl->operators.size(); }

const Core::RuleNode &DecompositionGraph::getRule(RuleId id) const { return impl->rules[id]; }

const std::vector<Core::RuleNode> &
DecompositionGraph::getAllRulesFor(const Core::OperatorNode &op) const
{
    static const std::vector<Core::RuleNode> empty;
    const auto it = impl->opToRules.find(op);
    if (it != impl->opToRules.end()) {
        return it->second;
    }
    return empty;
}

bool DecompositionGraph::isTargetGate(const Core::OperatorNode &op) const
{
    return impl->gateset.contains(op);
}

bool DecompositionGraph::hasOperator(const Core::OperatorNode &op) const
{
    return impl->opToId.find(op) != impl->opToId.end();
}

void DecompositionGraph::showGraph() const
{
    std::cerr << "Decomposition Graph:\n";
    // Show all operators by their names
    std::cerr << "Operators:\n";
    for (const auto &[op, id] : impl->opToId) {
        std::cerr << "  ID " << id << ": " << Core::print_op(op) << "\n";
    }

    // Show all rules by their names and their input/output operators
    std::cerr << "Rules:\n";
    for (const auto &[ruleId, vertex] : impl->ruleIdToVertex) {
        const auto &rule = impl->rules[ruleId];
        std::cerr << "  Rule ID " << ruleId << ": " << rule.name << "\n";
        std::cerr << "    Output: " << Core::print_op(rule.output) << "\n";
        std::cerr << "    Inputs:\n";
        for (const auto &input : rule.inputs) {
            std::cerr << "      - " << Core::print_op(input.op)
                      << " (multiplicity: " << input.multiplicity << ")\n";
        }
    }

    // Show target gateset
    std::cerr << "Target Gateset:\n";
    for (const auto &[op, cost] : impl->gateset.ops) {
        std::cerr << "  " << Core::print_op(op) << " with cost " << cost << "\n";
    }
}

} // namespace DecompGraph::Solver
