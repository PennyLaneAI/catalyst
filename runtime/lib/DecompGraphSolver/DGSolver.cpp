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
        graph.showGraph(); // Debug: show the graph structure
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
        Core::ChosenDecompRule child;
        try {
            child = solveOperator(input.op);
        }
        catch (const Core::CyclicDecompositionError &) {
            // A cycle in the decomposition graph invalidates this rule,
            // but should not prevent trying sibling decomposition rules.
            return invalidRule(solution.op);
        }
        catch (const Core::GraphSolveError &) {
            // Any error in solving the input operator invalidates this rule,
            // but should not prevent trying sibling decomposition rules.
            return invalidRule(solution.op);
        }

        if (child.ruleName.empty()) {
            return invalidRule(solution.op); // invalid rule
        }
        total_cost += child.totalCost * static_cast<double>(input.multiplicity);
        for (const auto &[basis_op, count] : child.basisCounts) {
            solution.basisCounts[basis_op] += count * input.multiplicity;
        }
    }

    if (total_cost == 0.0) {
        return invalidRule(solution.op); // invalid rule
    }
    solution.totalCost = total_cost;
    return solution;
}

Core::ChosenDecompRule DecompositionSolver::bestRule(const Core::OperatorNode &op)
{
    const auto &all_rules = graph.getAllRulesFor(op);
    if (all_rules.empty()) {
        return invalidRule(op); // no valid rules
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
        return invalidRule(op); // no valid rules
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
        graph.showGraph(); // Debug: show the graph structure
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

    // Debug: print the chosen rule for the operator
    std::cerr << "Chosen rule for operator " << op.name << ": " << chosen.ruleName << " with cost "
              << chosen.totalCost << "\n"; // FIXME: remove after debugging

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
        const auto chosen_rule = solveOperator(root);
        if (chosen_rule.ruleName.empty()) {
            graph.showGraph();                            // Debug: show the graph structure
            throw Core::GraphSolverFailedError(root, {}); // all rules failed for this root operator
        }
    }

    for (const auto &[op, entry] : solvedMap) {
        result.optimizedMap.emplace(op, entry);
    }

    return result;
}

DecompositionSolver::SolutionType DecompositionSolver::getSolvedMap()
{
    if (solvedMap.empty()) {
        solve();
    }
    return solvedMap;
}

} // namespace DecompGraph::Solver
