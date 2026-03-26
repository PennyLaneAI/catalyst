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

#include <algorithm>

#include <iostream>

#include "DGSolver.hpp"
#include "DGTypes.hpp"

namespace DecompGraph::Solver {

Core::ChosenDecompRule DecompositionSolver::basisRule(const Core::OperatorNode &op)
{
    if (!graph.isTargetGate(op)) {
        throw Core::GraphSolveError("Operator is not a target gate in the gateset");
    }

    Core::ChosenDecompRule solution;
    solution.op = op;
    solution.isBasis = true;
    solution.totalCost = graph.getGateset().getCost(op);
    solution.basisCounts.emplace(op, 1);
    solution.ruleName = "BasisRule";
    return solution;
}

Core::ChosenDecompRule DecompositionSolver::evalRule(const Core::RuleNode &rule)
{
    Core::ChosenDecompRule solution;
    solution.ruleName = rule.name;
    solution.isBasis = false;
    solution.inputs = rule.inputs;
    solution.op = rule.output;

    double total_cost = 0.0;
    for (const auto &input : rule.inputs) {
        const auto child = solveOperator(input.op);
        if (child.ruleName.empty()) {
            return {solution.op, false, "", {}, 0.0, {}}; // invalid rule
        }
        total_cost += child.totalCost * static_cast<double>(input.multiplicity);
        for (const auto &[basis_op, count] : child.basisCounts) {
            solution.basisCounts[basis_op] += count * input.multiplicity;
        }
    }

    if (total_cost == 0.0) {
        return {solution.op, false, "", {}, 0.0, {}}; // invalid rule
    }
    solution.totalCost = total_cost;
    return solution;
}

Core::ChosenDecompRule DecompositionSolver::bestRule(const Core::OperatorNode &op)
{
    const auto &all_rules = graph.getAllRulesFor(op);
    if (all_rules.empty()) {
        return {op, false, "", {}, 0.0, {}}; // no valid rules
    }

    std::optional<Core::ChosenDecompRule> best_rule;

    for (const auto &rule : all_rules) {
        auto candidate = evalRule(rule);
        if (!candidate.ruleName.empty() &&
            (!best_rule.has_value() || candidate.totalCost < best_rule->totalCost)) {
            best_rule = std::move(candidate);
        }
    }

    if (!best_rule.has_value()) {
        return {op, false, "", {}, 0.0, {}}; // no valid rules
    }

    return best_rule.value();
}

Core::ChosenDecompRule DecompositionSolver::solveOperator(const Core::OperatorNode &op)
{
    // Check if the operator has already been solved
    if (const auto it = solvedMap.find(op); it != solvedMap.end()) {
        return it->second;
    }

    if (visited.find(op) != visited.end()) {
        throw Core::CyclicDecompositionError(solvingStack);
    }

    // RAII guard for the visited set to check/solve the graph recursively
    struct VisitGuard {
        std::unordered_set<Core::OperatorNode, Core::OperatorNodeHash> &visited_;
        std::vector<Core::OperatorNode> &solvingStack_;
        const Core::OperatorNode &currentNode_;

        explicit VisitGuard(std::unordered_set<Core::OperatorNode, Core::OperatorNodeHash> &visited,
                            std::vector<Core::OperatorNode> &solvingStack,
                            const Core::OperatorNode &node)
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
    if (!chosen.ruleName.empty()) {
        solvedMap.emplace(op, chosen);
    }
    return chosen;
}

Core::GraphResult DecompositionSolver::solve()
{
    Core::GraphResult result;

    result.solvedRoots = graph.getRoots();

    for (const auto &root : result.solvedRoots) {
        std::cerr << "Solving for root operator: " << root.name << "\n";
        const auto chosen_rule = solveOperator(root);
        std::cerr << "Chosen rule for operator " << root.name << ": " << chosen_rule.ruleName
                  << " with cost " << chosen_rule.totalCost << "\n";
        if (chosen_rule.ruleName.empty()) {
            throw Core::GraphSolverFailedError(root, {}); // no valid rules
        }
    }

    for (const auto &[op, entry] : solvedMap) {
        result.optimizedMap.emplace(op, entry);
    }

    return result;
}

DecompositionSolver::SolutionType DecompositionSolver::getSolvedMap()
{
    graph.showGraph(); // Debug: show the graph structure

    if (solvedMap.empty()) {
        solve();
    }
    return solvedMap;
}

} // namespace DecompGraph::Solver
