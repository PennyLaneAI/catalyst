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
 * @file Utils.hpp
 * @brief This file defines utility functions and error classes for the decomposition framework.
 */
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DecompositionTypes.hpp"

namespace DecompGraph::Utils {

auto print_op(const Core::OperatorNode &op) -> std::string
{
    std::ostringstream oss;
    oss << op.name;
    if (op.numWires != 0U) {
        oss << " [w:" << op.numWires << "]";
    }
    if (op.numParams != 0U) {
        oss << " [p:" << op.numParams << "]";
    }
    if (op.adjoint) {
        oss << " [adj]";
    }
    return oss.str();
}

auto cycle_message(const std::vector<Core::OperatorNode> &cycle) -> std::string
{
    std::ostringstream oss;
    oss << "Cyclic decomposition detected: ";
    for (auto i = 0; i < cycle.size(); i++) {
        if (i != 0U) {
            oss << " -> ";
        }
        oss << print_op(cycle[i]);
    }
    return oss.str();
}

auto graph_failed_message(const Core::OperatorNode &op, const std::vector<std::string> &rule_errors)
    -> std::string
{
    std::ostringstream oss;
    oss << "Graph is failed for operator '" << print_op(op) << "'";
    if (!rule_errors.empty()) {
        oss << ". Tried rules:";
        for (const auto &error : rule_errors) {
            oss << "\n  - " << error;
        }
    }
    return oss.str();
}

class GraphSolveError : public std::runtime_error {
  public:
    explicit GraphSolveError(std::string message) : std::runtime_error(std::move(message)) {}
};

class MissingRuleForOperatorError : public GraphSolveError {
  public:
    explicit MissingRuleForOperatorError(Core::OperatorNode op)
        : GraphSolveError("Missing rule for operator: " + print_op(op))
    {
    }
};

class CyclicDecompositionError : public GraphSolveError {
  public:
    explicit CyclicDecompositionError(std::vector<Core::OperatorNode> cycle)
        : GraphSolveError(cycle_message(cycle))
    {
    }
};

class GraphSolverFailedError : public GraphSolveError {
  public:
    GraphSolverFailedError(Core::OperatorNode op, std::vector<std::string> rule_errors)
        : GraphSolveError(graph_failed_message(op, rule_errors))
    {
    }
};
} // namespace DecompGraph::Utils
