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

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "DGTypes.hpp"

namespace DecompGraph::Core {

static inline auto print_op(const OperatorNode &op) -> std::string {
    // id override
    if (!op.id.empty()) {
        return "id: " + op.id;
    }

    std::ostringstream oss;
    oss << op.name;
    oss << "[w:" << op.numWires << "]";
    oss << "[p:" << op.numParams << "]";
    if (!op.staticNamedArgs.empty()) {
        std::vector<std::string> keys;
        keys.reserve(op.staticNamedArgs.size());
        for (const auto &[k, _] : op.staticNamedArgs) {
            keys.push_back(k);
        }
        std::sort(keys.begin(), keys.end());
        for (const auto &k : keys) {
            oss << "[" << k << ":" << op.staticNamedArgs.at(k) << "]";
        }
    }
    return oss.str();
}

static inline auto graph_failed_message(const OperatorNode &op,
                                        const std::vector<std::string> &rule_errors,
                                        const std::vector<OperatorNode> &unsolvable = {})
    -> std::string {
    std::ostringstream oss;
    oss << "Decomposition rule not found for operator '" << print_op(op) << "'";
    // Keep the tried rules right next to the failed operator, so both stay together (and survive a
    // truncated snippet of the trace); the longer required-gates list follows.
    if (!rule_errors.empty()) {
        oss << ".\nTried rules for '" << print_op(op) << "':";
        for (const auto &error : rule_errors) {
            oss << "\n  - " << error;
        }
    }
    if (!unsolvable.empty()) {
        oss << "\nThe following required operators could not reach the target gateset:";
        constexpr size_t maxToShow = 25;
        size_t shown = 0;
        for (const auto &u : unsolvable) {
            if (shown++ == maxToShow) {
                oss << "\n  * ... and " << (unsolvable.size() - maxToShow) << " more";
                break;
            }
            oss << "\n  * " << print_op(u);
        }
        oss << "\nAdd one of these (or gates they can decompose into) to the target gateset.";
    }
    return oss.str();
}

/**
 * Both `GraphResult` and `ChosenDecompRule::basisCounts` are unordered maps,
 * so every level is sorted by its printed operator label before being written.
 * Without that the dump comes out in a different order from run to run,
 * which makes it hard to check for lit tests.
 */
static inline void showSolution(const Core::GraphResult &result) {
    std::vector<std::pair<std::string, const Core::ChosenDecompRule *>> entries;
    entries.reserve(result.size());
    for (const auto &[op, rule] : result) {
        entries.emplace_back(print_op(op), &rule);
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::cerr << "Decomposition Solution:\n";
    for (const auto &[opLabel, rule] : entries) {
        std::cerr << "  Operator: " << opLabel << "\n";
        std::cerr << "    Chosen Rule: " << rule->ruleName << (rule->isBasis ? " [basis]" : "")
                  << "\n";
        std::cerr << "    Total Cost: " << rule->totalCost << "\n";
        std::cerr << "    Basis Counts:\n";

        std::vector<std::pair<std::string, size_t>> basisCounts;
        basisCounts.reserve(rule->basisCounts.size());
        for (const auto &[basis_op, count] : rule->basisCounts) {
            basisCounts.emplace_back(print_op(basis_op), count);
        }
        std::sort(basisCounts.begin(), basisCounts.end());
        for (const auto &[basisLabel, count] : basisCounts) {
            std::cerr << "      - " << basisLabel << ": " << count << "\n";
        }
    }
}

class GraphError : public std::runtime_error {
  public:
    explicit GraphError(std::string message) : std::runtime_error(std::move(message)) {}
};

class GraphSolverFailedError : public GraphError {
  public:
    GraphSolverFailedError(OperatorNode op, std::vector<std::string> rule_errors,
                           std::vector<OperatorNode> unsolvable = {})
        : GraphError(graph_failed_message(op, rule_errors, unsolvable)) {}
};

class RuleInvalidOverrideError : public GraphError {
  public:
    RuleInvalidOverrideError(const std::string &kind, const OperatorNode &op, const RuleNode &rule)
        : GraphError("Invalid " + kind + " override for operator '" + print_op(op) +
                     "' with rule '" + rule.name + "' for rule.output '" + print_op(rule.output) +
                     "'") {}
};

} // namespace DecompGraph::Core
