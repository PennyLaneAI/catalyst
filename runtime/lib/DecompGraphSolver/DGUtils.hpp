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
 * @file DGUtils.hpp
 *
 * @brief This file defines utility functions and custom exceptions
 * for the graph decomposition framework.
 */
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DGTypes.hpp"

namespace DecompGraph::Core {

static inline auto print_op(const OperatorNode &op) -> std::string
{
    std::ostringstream oss;
    oss << op.name;
    oss << "[w:" << op.numWires << "]";
    oss << "[p:" << op.numParams << "]";
    if (op.adjoint) {
        oss << "[adj]";
    }
    return oss.str();
}

static inline auto cycle_message(const std::vector<OperatorNode> &cycle) -> std::string
{
    std::ostringstream oss;
    oss << "Cyclic decomposition detected: ";
    for (std::size_t i = 0; i < cycle.size(); i++) {
        if (i != 0U) {
            oss << " -> ";
        }
        oss << print_op(cycle[i]);
    }
    return oss.str();
}

static inline auto graph_failed_message(const OperatorNode &op,
                                        const std::vector<std::string> &rule_errors) -> std::string
{
    std::ostringstream oss;
    oss << "Decomposition rule not found for operator '" << print_op(op) << "'";
    if (!rule_errors.empty()) {
        oss << ". Tried rules:";
        for (const auto &error : rule_errors) {
            oss << "\n  - " << error;
        }
    }
    return oss.str();
}

class GraphError : public std::runtime_error {
  public:
    explicit GraphError(std::string message) : std::runtime_error(std::move(message)) {}
};

class MissingRuleForOperatorError : public GraphError {
  public:
    explicit MissingRuleForOperatorError(OperatorNode op)
        : GraphError("Missing rule for operator: " + print_op(op))
    {
    }
};

class CyclicDecompositionError : public GraphError {
  public:
    explicit CyclicDecompositionError(std::vector<OperatorNode> cycle)
        : GraphError(cycle_message(cycle))
    {
    }
};

class GraphSolverFailedError : public GraphError {
  public:
    GraphSolverFailedError(OperatorNode op, std::vector<std::string> rule_errors)
        : GraphError(graph_failed_message(op, rule_errors))
    {
    }
};

class RuleInvalidOverrideError : public GraphError {
  public:
    RuleInvalidOverrideError(const std::string &kind, const OperatorNode &op, const RuleNode &rule)
        : GraphError("Invalid " + kind + " override for operator '" + print_op(op) +
                     "' with rule '" + rule.name + "' for rule.output '" + print_op(rule.output) +
                     "'")
    {
    }
};

} // namespace DecompGraph::Core
