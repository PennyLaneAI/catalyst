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
 * @file DGSolver.cpp
 */

#include "DGSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DGTypes.hpp"
#include "DGUtils.hpp"

using namespace DecompGraph::Core;

namespace DecompGraph::Solver {

namespace {

/**
 * @brief The best rule offered so far for an operator that has not yet been fully explored.
 */
struct RuleCandidate {
    ChosenDecompRule chosenRule;
    bool isEmptyRule{false};
    std::size_t ruleIdx{0}; // index in getAllRulesFor(op)
};

/**
 * @brief Comparison function for competing rules.
 *
 * Prefers cost first, emptiness second, ordering last.
 */
bool isBetter(double cost, bool isEmptyRule, std::size_t ruleOrder, const RuleCandidate &current) {
    if (cost != current.chosenRule.totalCost) {
        return cost < current.chosenRule.totalCost;
    }
    if (isEmptyRule != current.isEmptyRule) {
        return isEmptyRule;
    }
    return ruleOrder < current.ruleIdx;
}

/**
 * @brief A rule whose inputs have not been fully explored.
 *
 * A rule cannot be applied unless all of its input operators are reachable from the gateset.
 */
struct PendingRule {
    const RuleNode *rule;
    std::size_t ruleOrder;
    std::size_t remainingInputs;
};

struct QueueEntry {
    double cost;
    OperatorNode op;
};

/**
 * @brief Queue entry comparison function, implementing a min-heap on the cost of entries.
 *
 * Ties are broken by operator ID.
 */
struct QueueEntryGreater {
    bool operator()(const QueueEntry &lhs, const QueueEntry &rhs) const {
        if (lhs.cost != rhs.cost) {
            return lhs.cost > rhs.cost;
        }
        return lhs.op.id > rhs.op.id;
    }
};

} // namespace

std::vector<OperatorNode>
DecompositionSolver::collectReachableOps(const std::vector<OperatorNode> &roots) const {
    std::vector<OperatorNode> reachableOps;
    std::unordered_set<OperatorNode, OperatorNodeHash> visited;
    std::vector<OperatorNode> worklist(roots.rbegin(), roots.rend());

    while (!worklist.empty()) {
        OperatorNode op = std::move(worklist.back());
        worklist.pop_back();
        if (!visited.insert(op).second) {
            continue;
        }
        reachableOps.push_back(op);
        if (graph.isTargetGate(op)) {
            continue;
        }
        for (const auto &rule : graph.getAllRulesFor(op)) {
            for (const auto &input : rule.inputs) {
                if (input.multiplicity != 0 && visited.find(input.op) == visited.end()) {
                    worklist.push_back(input.op);
                }
            }
        }
    }
    return reachableOps;
}

ChosenDecompRule DecompositionSolver::basisRule(const OperatorNode &op) const {
    ChosenDecompRule solution;
    solution.op = op;
    solution.isBasis = true;
    solution.totalCost = graph.getGateset().getCost(op);
    solution.basisCounts.emplace(op, 1);
    solution.ruleName = "BasisRule";
    return solution;
}

ChosenDecompRule DecompositionSolver::evalRule(const RuleNode &rule) const {
    ChosenDecompRule solution;
    solution.ruleName = rule.name;
    solution.isBasis = false;
    solution.inputs = rule.inputs;
    solution.op = rule.output;

    double total_cost = 0.0;
    for (const auto &input : rule.inputs) {
        if (input.multiplicity == 0) {
            continue;
        }
        const ChosenDecompRule &child = solvedMap.at(input.op);
        total_cost += child.totalCost * static_cast<double>(input.multiplicity);
        for (const auto &[basis_op, count] : child.basisCounts) {
            solution.basisCounts[basis_op] += count * input.multiplicity;
        }
    }

    solution.totalCost = total_cost;
    return solution;
}

/**
 * @brief Return an optimal solution to the decomposition graph.
 *
 * The solution is computed via a modified Dijkstra's algorithm.
 */
GraphResult DecompositionSolver::solve() {
    // Only reuse a result after every root has been solved successfully.
    if (solveSucceeded) {
        return solvedMap;
    }

    // A failed attempt may have finalized some operators. Start fresh on retry.
    solvedMap.clear();

    // validate gateset costs
    for (const auto &[name, cost] : graph.getGateset().ops) {
        if (std::isnan(cost) || cost < 0.0) {
            throw GraphError("Invalid cost " + std::to_string(cost) + " for gate '" + name +
                             "' in the target gateset: costs must be non-negative.");
        }
    }

    const std::vector<OperatorNode> reachableOps = collectReachableOps(graph.getRootOps());

    std::unordered_map<OperatorNode, RuleCandidate, OperatorNodeHash> opsToRuleCandidates;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> opQueue;

    auto offerRule = [&](ChosenDecompRule chosen, bool isEmptyRule, std::size_t ruleOrder) {
        if (solvedMap.find(chosen.op) != solvedMap.end()) {
            return;
        }
        const auto it = opsToRuleCandidates.find(chosen.op);
        if (it != opsToRuleCandidates.end() &&
            !isBetter(chosen.totalCost, isEmptyRule, ruleOrder, it->second)) {
            return;
        }
        opQueue.push({chosen.totalCost, chosen.op});
        OperatorNode op = chosen.op;
        opsToRuleCandidates.insert_or_assign(
            std::move(op), RuleCandidate{std::move(chosen), isEmptyRule, ruleOrder});
    };

    /*
     * opsToConsumingRules maps ops to the rules that consume them. Rules are stored here as indices
     * in the pendingRules vector. Pending rules keeps a counter of the unresolved op costs, and
     * once the counter reaches 0 the rule can be explored. This is how we implement the fact that a
     * rule cannot be explored unless all of its inputs have been explored.
     */
    std::unordered_map<OperatorNode, std::vector<std::size_t>, OperatorNodeHash>
        opsToConsumingRules;
    std::vector<PendingRule> pendingRules;

    for (const auto &op : reachableOps) {
        if (graph.isTargetGate(op)) {
            offerRule(basisRule(op), false, 0);
            continue;
        }

        const auto &rules = graph.getAllRulesFor(op);
        for (std::size_t order = 0; order < rules.size(); ++order) {
            const RuleNode &rule = rules[order];
            std::size_t numInputs = 0;
            for (const auto &input : rule.inputs) {
                if (input.multiplicity != 0) {
                    opsToConsumingRules[input.op].push_back(pendingRules.size());
                    ++numInputs;
                }
            }
            if (numInputs == 0) {
                offerRule(evalRule(rule), rule.isEmpty(), order);
            } else {
                pendingRules.push_back({&rule, order, numInputs});
            }
        }
    }

    while (!opQueue.empty()) {
        const QueueEntry currentOp = opQueue.top();
        opQueue.pop();
        if (solvedMap.find(currentOp.op) != solvedMap.end()) {
            continue;
        }
        auto ruleCandidate = opsToRuleCandidates.find(currentOp.op);
        if (currentOp.cost != ruleCandidate->second.chosenRule.totalCost) {
            continue; // superseded by a cheaper candidate
        }
        solvedMap.emplace(currentOp.op, std::move(ruleCandidate->second.chosenRule));
        opsToRuleCandidates.erase(ruleCandidate);

        const auto consumingRules = opsToConsumingRules.find(currentOp.op);
        if (consumingRules == opsToConsumingRules.end()) {
            continue;
        }
        for (const std::size_t ruleIndex : consumingRules->second) {
            PendingRule &pendingRule = pendingRules[ruleIndex];
            if (--pendingRule.remainingInputs == 0) {
                offerRule(evalRule(*pendingRule.rule), false, pendingRule.ruleOrder);
            }
        }
    }

    for (const auto &root : graph.getRootOps()) {
        if (solvedMap.find(root) == solvedMap.end()) {
            throwUnsolved(root);
        }
    }

    solveSucceeded = true;
    return solvedMap;
}

void DecompositionSolver::throwUnsolved(const OperatorNode &root) const {
    // Debugging output:
    showGraph(graph);
    showSolution(solvedMap);

    // Prepare error msg:
    std::vector<std::string> rules_error;
    for (const auto &rule : graph.getAllRulesFor(root)) {
        rules_error.push_back(rule.name);
    }

    // List of ops where the chain could not reach the gateset:
    std::vector<OperatorNode> unsolvable;
    for (const auto &op : collectReachableOps({root})) {
        if (op != root && solvedMap.find(op) == solvedMap.end()) {
            unsolvable.push_back(op);
        }
    }
    std::sort(
        unsolvable.begin(), unsolvable.end(),
        [](const OperatorNode &a, const OperatorNode &b) { return print_op(a) < print_op(b); });

    throw GraphSolverFailedError(root, rules_error, unsolvable);
}

} // namespace DecompGraph::Solver
