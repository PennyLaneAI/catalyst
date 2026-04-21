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
#include <optional>
#include <unordered_set>
#include <vector>

#include "DGTypes.hpp"

using namespace DecompGraph::Core;

namespace DecompGraph::Solver {

ChosenDecompRule DecompositionSolver::basisRule(const OperatorNode &op)
{
    if (!graph.isTargetGate(op)) {
        return invalidRule(op); // not a target gate, so no valid basis rule
    }

    ChosenDecompRule solution;
    solution.op = op;
    solution.isBasis = true;
    solution.totalCost = graph.getGateset().getCost(op);
    solution.basisCounts.emplace(op, 1);
    solution.ruleName = "BasisRule";
    return solution;
}

ChosenDecompRule DecompositionSolver::evalRule(const RuleNode &rule)
{
    ChosenDecompRule solution;
    solution.ruleName = rule.name;
    solution.isBasis = false;
    solution.inputs = rule.inputs;
    solution.op = rule.output;

    double total_cost = 0.0;
    for (const auto &input : rule.inputs) {
        ChosenDecompRule child;
        child = solveOperator(input.op);
        if (isInvalidRule(child)) {
            // if any input cannot be solved, this rule is invalid
            return invalidRule(solution.op);
        }

        total_cost += child.totalCost * static_cast<double>(input.multiplicity);
        for (const auto &[basis_op, count] : child.basisCounts) {
            solution.basisCounts[basis_op] += count * input.multiplicity;
        }
    }

    if (rule.inputs.empty() && total_cost == 0.0) {
        return invalidRule(solution.op); // invalid rule
    }
    solution.totalCost = total_cost;
    return solution;
}

ChosenDecompRule DecompositionSolver::bestRule(const OperatorNode &op)
{
    const auto &all_rules = graph.getAllRulesFor(op);
    if (all_rules.empty()) {
        return invalidRule(op); // no valid rules
    }

    std::optional<ChosenDecompRule> best_rule;

    for (const auto &rule : all_rules) {
        auto candidate = evalRule(rule);
        if (!isInvalidRule(candidate) &&
            (!best_rule.has_value() || candidate.totalCost < best_rule->totalCost)) {
            best_rule = std::move(candidate);
        }
    }

    if (!best_rule.has_value()) {
        return invalidRule(op); // no valid rules
    }

    return best_rule.value();
}

ChosenDecompRule DecompositionSolver::solveOperator(const OperatorNode &op)
{
    // Check if the operator has already been solved
    if (const auto it = solvedMap.find(op); it != solvedMap.end()) {
        return it->second;
    }

    if (visited.find(op) != visited.end()) {
        return invalidRule(op); // cycle detected, return invalid rule to prevent infinite recursion
    }

    // RAII guard for the visited set to check/solve the graph recursively
    struct VisitGuard {
        std::unordered_set<OperatorNode, OperatorNodeHash> &visited_;
        std::vector<OperatorNode> &solvingStack_;
        const OperatorNode &currentNode_;

        explicit VisitGuard(std::unordered_set<OperatorNode, OperatorNodeHash> &visited,
                            std::vector<OperatorNode> &solvingStack, const OperatorNode &node)
            : visited_(visited), solvingStack_(solvingStack), currentNode_(node)
        {
            visited_.insert(currentNode_);         // add to visited in case of exceptions
            solvingStack_.push_back(currentNode_); // push to stack in case of exceptions
        }
        ~VisitGuard()
        {
            visited_.erase(currentNode_);
            if (!solvingStack_.empty()) {
                solvingStack_.pop_back();
            }
        }

        VisitGuard(const VisitGuard &) = delete;
        VisitGuard &operator=(const VisitGuard &) = delete;
    } visitGuard(visited, solvingStack, op);

    auto chosen = graph.isTargetGate(op) ? basisRule(op) : bestRule(op);

    if (!isInvalidRule(chosen)) {
        solvedMap.emplace(op, chosen);
    }
    return chosen;
}

GraphResult DecompositionSolver::solve()
{
    // Return cached solution if already solved
    if (!solvedMap.empty()) {
        return solvedMap;
    }

    for (const auto &root : graph.getRootOps()) {
        const auto chosen_rule = solveOperator(root);
        if (isInvalidRule(chosen_rule)) {
            // Debugging output:
            graph.showGraph();
            showSolution(solvedMap);

            // Prepare error msg:
            std::vector<std::string> rules_error;
            for (const auto &rule : graph.getAllRulesFor(root)) {
                rules_error.push_back(rule.name);
            }

            throw GraphSolverFailedError(root,
                                         rules_error); // all rules failed for this root operator
        }
    }

    return solvedMap;
}

} // namespace DecompGraph::Solver
