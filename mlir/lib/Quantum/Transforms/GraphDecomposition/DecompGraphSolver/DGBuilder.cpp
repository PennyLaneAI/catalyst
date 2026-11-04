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
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DGTypes.hpp"
#include "DGUtils.hpp"

using namespace DecompGraph::Core;

namespace DecompGraph::Solver {

struct DecompositionGraph::Impl {
    using OperatorId = std::size_t;

    std::vector<OperatorNode> operators;
    WeightedGateset gateset;
    std::vector<RuleNode> rules;

    std::unordered_map<OperatorNode, OperatorId, OperatorNodeHash> opToId;
    std::unordered_map<OperatorNode, std::vector<RuleNode>, OperatorNodeHash> opToRules;

    void registerOp(const OperatorNode &op) { opToId.emplace(op, opToId.size()); }

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
        for (const auto &op : operators) {
            registerOp(op);
        }

        for (const auto &rule : rules) {
            registerOp(rule.output);
            opToRules[rule.output].push_back(rule);
            for (const auto &input : rule.inputs) {
                registerOp(input.op);
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
    for (std::size_t ruleId = 0; ruleId < impl->rules.size(); ++ruleId) {
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
