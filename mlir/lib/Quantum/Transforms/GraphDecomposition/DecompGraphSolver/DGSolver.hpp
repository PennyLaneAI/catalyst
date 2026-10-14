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
 * @file DGSolver.hpp
 *
 * @brief This file defines the DecompositionSolver class, which finds the cheapest decomposition
 * of every operator in a decomposition graph down to a weighted target gateset.
 *
 * The decomposition graph is a directed hypergraph: operators are nodes, and each rule is a
 * hyperedge from all of its input operators (each with a multiplicity) to its single output
 * operator. A rule can only be used once every one of its inputs can be decomposed, and its cost is
 * the multiplicity-weighted sum of its inputs' costs. Finding the cheapest decomposition is the
 * shortest-hyperpath problem (equivalently, minimum-weight derivations in a weighted context-free
 * grammar), which the solver handles with Knuth's generalization of Dijkstra's algorithm:
 *
 *   D. E. Knuth, "A generalization of Dijkstra's algorithm", Information Processing Letters 6(1),
 *   1977.
 *
 */

#pragma once

#include <vector>

#include "DGBuilder.hpp"
#include "DGTypes.hpp"

namespace DecompGraph::Solver {

/**
 * @brief Solves a decomposition graph for the cheapest rule of every reachable operator.
 *
 * Algorithm:
 *   1. Collect every operator reachable from the root operators by following rule inputs. Target
 *      gates are leaves: their own rules are never considered.
 *   2. Seed a priority queue with every reachable target gate (at its gateset cost) and every
 *      empty rule (at cost 0).
 *   3. Repeatedly explore the cheapest unexplored operator. A rule becomes usable once all of its
 *      inputs are explored, and offers its output the multiplicity-weighted sum of their costs.
 *   4. Every root operator must end up explored; otherwise GraphSolverFailedError is thrown.
 *
 * Each operator's result is its unconditional optimum, so the solution does not depend on the order
 * of the root operators or of the rules. Cyclic rules need no special handling: a rule waits until
 * its inputs are finalized, and by then its output is either still open or already final at an
 * equal or lower cost. Following each operator's chosen rule can never loop back to that operator,
 * since a chosen rule's inputs are always finalized before its output. The solver runs in
 * O((V + E) log V) for V reachable operators and E rule input terms.
 *
 * Limitations:
 *   - Gateset costs must be non-negative (and not NaN); solve() throws GraphError otherwise. The
 *     algorithm relies on a rule never costing less than any input it waits on, which a negative
 *     cost would break.
 *   - Rule inputs with multiplicity 0 are ignored: they contribute no cost and need not be
 *     solvable, since the rule does not produce them.
 *   - Costs are additive and context-free: an operator has one cost wherever it appears, and every
 *     copy is costed independently. Anything whose cost depends on where an operator appears (e.g.
 *     the number of work wires available to it, which combine as a maximum rather than a sum) is
 * not modelled. Supporting it requires folding that context into the operator's identity (e.g. one
 *     node per (operator, available work wires) pair), which keeps this algorithm exact at the cost
 *     of more nodes. Costing shared structure once rather than per copy would make the problem
 *     NP-hard (a Steiner problem on the hypergraph) and cannot be expressed here.
 *   - Rules have no cost of their own; a rule's cost is exactly the cost of its inputs.
 *   - Ties in total cost are broken deterministically, preferring an empty rule, then the rule that
 *     comes first in `DecompositionGraph::getAllRulesFor`. A tied rule is only considered if it
 *     becomes usable before its output is finalized, which can fail to happen when the tied rule
 *     depends on an operator of exactly the same cost as its output (e.g. through zero-cost
 *     inputs). The chosen cost is optimal either way; only which of the tied rules wins can differ.
 *   - Target gates are always kept as basis gates, even when a rule could replace them more
 * cheaply.
 *   - The whole reachable graph is solved, not just the part needed by the roots, so that the
 *     result covers every reachable solvable operator.
 */
class DecompositionSolver {
  public:
    /**
     * @brief Constructs a DecompositionSolver with the given decomposition graph.
     *
     * @param _graph The decomposition graph to be solved.
     */
    explicit DecompositionSolver(const DecompositionGraph &_graph) : graph(_graph) {}

    /**
     * @brief Solves the decomposition graph and returns the chosen rule for every reachable
     * operator that can reach the target gateset.
     *
     * The result is cached, so later calls return it without solving again.
     *
     * @throws GraphError if a gateset cost is negative or NaN.
     * @throws GraphSolverFailedError if a root operator cannot reach the target gateset.
     *
     * @return Core::GraphResult The mapping from each solved operator to its chosen rule.
     */
    Core::GraphResult solve();

  private:
    const DecompositionGraph &graph;

    Core::GraphResult solvedMap{};

    /**
     * @brief Returns every operator reachable from the given operators by following rule inputs,
     * in discovery order. Target gates are not expanded, and inputs with multiplicity 0 are not
     * followed.
     */
    std::vector<Core::OperatorNode>
    collectReachableOps(const std::vector<Core::OperatorNode> &roots) const;

    /**
     * @brief Returns the basis solution for a target gate: the operator itself, at its gateset
     * cost.
     */
    Core::ChosenDecompRule basisRule(const Core::OperatorNode &op) const;

    /**
     * @brief Returns the solution given by applying the rule. Every input with a non-zero
     * multiplicity must already be in solvedMap.
     */
    Core::ChosenDecompRule evalRule(const Core::RuleNode &rule) const;

    /**
     * @brief Throws GraphSolverFailedError for a root operator that could not be solved, listing
     * its rules and the operators reachable from it that could not reach the gateset.
     */
    [[noreturn]] void throwUnsolved(const Core::OperatorNode &root) const;
};

} // namespace DecompGraph::Solver
