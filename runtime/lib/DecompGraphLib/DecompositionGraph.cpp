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
 * @file DecompositionGraph.cpp
 */

#include <boost/graph/adjacency_list.hpp>

#include <iostream>
#include <variant>

#include "DecompositionGraph.hpp"

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

    struct GraphWeightedEdge {
        // EdgeType type;
        // double weight = 0.0; // default 0.0 for StartToBasisOp & RuleToOperator
    };

    using BbGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                                          GraphVertex, GraphWeightedEdge>;
    using Vertex = boost::graph_traits<BbGraph>::vertex_descriptor;

    BbGraph graph;
    std::vector<Core::OperatorNode> operators;
    Core::WeightedGateset gateset;
    std::vector<Core::RuleNode> rules;

    std::unordered_map<Core::OperatorNode, OperatorId, Core::OperatorNodeHash> opToId;
    std::vector<Core::OperatorNode> idToOp;
    std::unordered_map<OperatorId, Vertex> opIdToVertex;

    std::unordered_map<RuleId, Vertex> ruleIdToVertex;
    std::unordered_map<Core::OperatorNode, std::vector<Core::RuleNode>, Core::OperatorNodeHash>
        rulesForop;

    OperatorId registerOp(const Core::OperatorNode &op)
    {
        const auto it = opToId.find(op);
        if (it != opToId.end()) {
            return it->second;
        } // else {

        OperatorId newId = opToId.size();
        opToId.emplace(op, newId);
        idToOp.push_back(op);
        // return newId;
        // opToId[op] = newId;
        const auto vertex =
            boost::add_vertex(GraphVertex{VertexType::Operator, OperatorVertex{newId}}, graph);
        opIdToVertex.emplace(newId, vertex);
        return newId;
    }

    Impl(std::vector<Core::OperatorNode> _operators, Core::WeightedGateset _gateset,
         std::vector<Core::RuleNode> _rules)
        : operators(std::move(_operators)), gateset(std::move(_gateset)), rules(std::move(_rules))
    {
    }

    void buildGraph()
    {
        // Register all operators and create vertices
        for (const auto &op : operators) {
            registerOp(op);
        }

        // Register all target gates and create vertices
        for (const auto &[op, _] : gateset.ops) {
            registerOp(op);
        }

        // Register all rules and create vertices
        for (RuleId ruleId = 0; ruleId < rules.size(); ruleId++) {
            const auto &rule = rules[ruleId];
            registerOp(rule.output);
            for (const auto &inout : rule.inputs) {
                registerOp(inout.op);
            }

            const auto ruleVertex =
                boost::add_vertex(GraphVertex{VertexType::Rule, RuleVertex{ruleId}}, graph);
            // const auto ruleVertex = boost::add_vertex(GraphVertex{VertexType::Rule,
            // RuleVertex{ruleId}}, graph);
            ruleIdToVertex.emplace(ruleId, ruleVertex);
            rulesForop[rule.output].push_back(rule);
        }

        for (RuleId ruleId = 0; ruleId < rules.size(); ruleId++) {
            const auto &rule = rules[ruleId];
            const auto rule_vertex = ruleIdToVertex[ruleId];
            const auto output_vertex = opIdToVertex[opToId[rule.output]];
            boost::add_edge(rule_vertex, output_vertex, GraphWeightedEdge{}, graph);

            for (const auto &input : rule.inputs) {
                const auto input_vertex = opIdToVertex[opToId[input.op]];
                boost::add_edge(input_vertex, rule_vertex, GraphWeightedEdge{}, graph);
            }
        }
    }
};

DecompositionGraph::DecompositionGraph(std::vector<Core::OperatorNode> operators,
                                       Core::WeightedGateset gateset,
                                       std::vector<Core::RuleNode> rules)
    : impl(std::make_unique<Impl>(std::move(operators), std::move(gateset), std::move(rules)))
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

std::size_t DecompositionGraph::getNumRules() const { return impl->rules.size(); }

std::size_t DecompositionGraph::getNumOperators() const { return impl->operators.size(); }

const Core::RuleNode &DecompositionGraph::getRule(RuleId id) const { return impl->rules[id]; }

const std::vector<Core::RuleNode> &
DecompositionGraph::getAllRulesFor(const Core::OperatorNode &op) const
{
    static const std::vector<Core::RuleNode> empty;
    const auto it = impl->rulesForop.find(op);
    if (it != impl->rulesForop.end()) {
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

} // namespace DecompGraph::Solver
