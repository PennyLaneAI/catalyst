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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "DecompositionGraph.hpp"
#include "DecompositionSolver.hpp"
#include "QuantumNodes.hpp"

using namespace Catch::Matchers;
using namespace DecompGraph::Core;
using namespace DecompGraph::Solver;

TEST_CASE("Test DecompositionGraph construction", "[DecompGraph::Solver]")
{
    const auto h = OperatorNode{"H", 1, 0, false};
    const auto rz = OperatorNode{"RZ", 1, 1, false};
    const auto rx = OperatorNode{"RX", 1, 1, false};

    const WeightedGateset gateset{{{rz, 1.0}, {rx, 2.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
    };

    const DecompositionGraph graph({h}, gateset, rules);

    REQUIRE(graph.getRoots().size() == 1);
    REQUIRE(graph.getRoots()[0] == h);
    REQUIRE(graph.getGateset().ops.size() == 2);
    REQUIRE(graph.getGateset().contains(rz));
    REQUIRE(graph.getGateset().contains(rx));
    REQUIRE_FALSE(graph.getGateset().contains(h));
    REQUIRE(graph.getRules().size() == 1);
    REQUIRE(graph.getRules()[0].name == "h_to_rz_rx_rz");
    REQUIRE(graph.getNumRules() == 1);
    REQUIRE(graph.getNumOperators() == 1);
    REQUIRE(graph.getRule(0).name == "h_to_rz_rx_rz");
    REQUIRE(graph.getAllRulesFor(h).size() == 1);
    REQUIRE(graph.getAllRulesFor(h)[0].name == "h_to_rz_rx_rz");
    REQUIRE(graph.isTargetGate(rz));
    REQUIRE(graph.isTargetGate(rx));
    REQUIRE_FALSE(graph.isTargetGate(h));
    REQUIRE(graph.hasOperator(h));
    REQUIRE(graph.hasOperator(rz));
}

TEST_CASE("Test DecompositionGraph copy and move semantics", "[DecompGraph::Solver]")
{
    const auto h = OperatorNode{"H", 1, 0, false};
    const auto rz = OperatorNode{"RZ", 1, 1, false};
    const auto rx = OperatorNode{"RX", 1, 1, false};

    const WeightedGateset gateset{{{rz, 1.0}, {rx, 2.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
    };

    DecompositionGraph graph({h}, gateset, rules);

    // Test copy constructor
    DecompositionGraph copyConstructedGraph(graph);
    REQUIRE(copyConstructedGraph.getRoots() == graph.getRoots());
    REQUIRE(copyConstructedGraph.getGateset().ops == graph.getGateset().ops);
    REQUIRE(copyConstructedGraph.getRules().size() == graph.getRules().size());
    REQUIRE(copyConstructedGraph.getRules()[0].name == graph.getRules()[0].name);

    // Test copy assignment operator
    DecompositionGraph copyAssignedGraph = graph;
    REQUIRE(copyAssignedGraph.getRoots() == graph.getRoots());
    REQUIRE(copyAssignedGraph.getGateset().ops == graph.getGateset().ops);
    REQUIRE(copyAssignedGraph.getRules().size() == graph.getRules().size());
    REQUIRE(copyAssignedGraph.getRules()[0].name == graph.getRules()[0].name);

    // Test move constructor
    DecompositionGraph moveConstructedGraph(std::move(graph));
    REQUIRE(moveConstructedGraph.getRoots() == copyConstructedGraph.getRoots());
    REQUIRE(moveConstructedGraph.getGateset().ops == copyConstructedGraph.getGateset().ops);
    REQUIRE(moveConstructedGraph.getRules().size() == copyConstructedGraph.getRules().size());
    REQUIRE(moveConstructedGraph.getRules()[0].name == copyConstructedGraph.getRules()[0].name);

    // Test move assignment operator
    DecompositionGraph moveAssignedGraph = std::move(copyAssignedGraph);
    REQUIRE(moveAssignedGraph.getRoots() == copyConstructedGraph.getRoots());
    REQUIRE(moveAssignedGraph.getGateset().ops == copyConstructedGraph.getGateset().ops);
    REQUIRE(moveAssignedGraph.getRules().size() == copyConstructedGraph.getRules().size());
    REQUIRE(moveAssignedGraph.getRules()[0].name == copyConstructedGraph.getRules()[0].name);
}

TEST_CASE("Test DecompositionGraph lookup and counting", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H", 1, 0, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode rx{"RX", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};

    const WeightedGateset gateset{{{rz, 1.0}, {ry, 2.0}, {rx, 3.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
        {"h_to_ry_rx_ry", h, {{ry, 2}, {rx, 1}}},
    };

    const DecompositionGraph graph({h}, gateset, rules);
    REQUIRE(graph.getNumRules() == 2);
    REQUIRE(graph.getNumOperators() == 1);

    REQUIRE(graph.getAllRulesFor(h).size() == 2);
    REQUIRE(graph.getAllRulesFor(h)[0].name == "h_to_rz_rx_rz");
    REQUIRE(graph.getAllRulesFor(h)[1].name == "h_to_ry_rx_ry");

    // compute the cost of each rule
    for (const auto &rule : graph.getAllRulesFor(h)) {
        double totalCost = 0.0;
        for (const auto &input : rule.inputs) {
            totalCost += graph.getGateset().ops.at(input.op) * input.multiplicity;
        }
        if (rule.name == "h_to_rz_rx_rz") {
            REQUIRE(totalCost == 1.0 * 2 + 3.0 * 1);
        }
        else if (rule.name == "h_to_ry_rx_ry") {
            REQUIRE(totalCost == 2.0 * 2 + 3.0 * 1);
        }
    }
}

TEST_CASE("Test the graph construction with realistic ops and multiple rules from PennyLane",
          "[DecompGraph::Solver]")
{
    const OperatorNode h{"H", 1, 0, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode rx{"RX", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};
    const OperatorNode cnot{"CNOT", 2, 0, false};
    const OperatorNode swap{"SWAP", 2, 0, false};
    const OperatorNode customBellOp{"BellOp", 2, 0, false};

    const WeightedGateset gateset{{{rz, 1.0}, {rx, 3.0}, {cnot, 5.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
        {"h_to_ry_rx_ry", h, {{ry, 2}, {rx, 1}}},
        {"swap_to_cnot", swap, {{cnot, 3}}},
        {"bell_to_cnot_h", customBellOp, {{cnot, 1}, {h, 1}}},
        {"bell_to_swap_h_cnot", customBellOp, {{swap, 1}, {h, 1}, {cnot, 1}}},
    };

    const DecompositionGraph graph({customBellOp}, gateset, rules);
    REQUIRE(graph.getNumRules() == 5);
    REQUIRE(graph.getNumOperators() == 1);
    REQUIRE(graph.getAllRulesFor(customBellOp).size() == 2);
}

TEST_CASE("Test DecompositionSolver with one single operator", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H", 1, 0, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode rx{"RX", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};

    const WeightedGateset gateset{{{rz, 1.0}, {ry, 2.0}, {rx, 3.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
        {"h_to_ry_rx_ry", h, {{ry, 2}, {rx, 1}}},
    };

    const DecompositionGraph graph({h}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    REQUIRE(result.solvedRoots.size() == 1);
    REQUIRE(result.solvedRoots[0] == h);
    REQUIRE(result.optimizedMap.size() == 4);
    const auto &chosen_rule = result.optimizedMap.at(h);
    REQUIRE_FALSE(chosen_rule.isBasis);
    REQUIRE(chosen_rule.ruleName == "h_to_rz_rx_rz");
    REQUIRE(chosen_rule.inputs.size() == 2);
    REQUIRE(chosen_rule.inputs[0].op == rz);
    REQUIRE(chosen_rule.inputs[0].multiplicity == 2);
    REQUIRE(chosen_rule.inputs[1].op == rx);
    REQUIRE(chosen_rule.inputs[1].multiplicity == 1);
    REQUIRE(chosen_rule.totalCost == 1.0 * 2 + 3.0 * 1);
    REQUIRE(chosen_rule.basisCounts.size() == 2);
    REQUIRE(chosen_rule.basisCounts.at(rz) == 2);
    REQUIRE(chosen_rule.basisCounts.at(rx) == 1);

    const auto &rz_rule = result.optimizedMap.at(rz);
    REQUIRE(rz_rule.isBasis);
    const auto &rx_rule = result.optimizedMap.at(rx);
    REQUIRE(rx_rule.isBasis);
    const auto &ry_rule = result.optimizedMap.at(ry);
    REQUIRE(ry_rule.isBasis);
}

TEST_CASE("Test the graph solver with intermediate ops and multiple rules", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H", 1, 0, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode rx{"RX", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};
    const OperatorNode cnot{"CNOT", 2, 0, false};
    const OperatorNode swap{"SWAP", 2, 0, false};
    const OperatorNode customBellOp{"BellOp", 2, 0, false};

    const WeightedGateset gateset{{{rz, 1.0}, {rx, 3.0}, {cnot, 5.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
        {"h_to_ry_rx_ry", h, {{ry, 2}, {rx, 1}}},
        {"swap_to_cnot", swap, {{cnot, 3}}},
        {"bell_to_cnot_h", customBellOp, {{cnot, 1}, {h, 1}}},
        {"bell_to_swap_h_cnot", customBellOp, {{swap, 1}, {h, 1}, {cnot, 1}}},
    };

    const DecompositionGraph graph({customBellOp}, gateset, rules);
    DecompositionSolver solver(graph);

    const auto result = solver.solve();
    REQUIRE(result.solvedRoots.size() == 1);
    REQUIRE(result.solvedRoots[0] == customBellOp);
    REQUIRE(result.optimizedMap.size() == 6);
    const auto &chosen_rule = result.optimizedMap.at(customBellOp);
    REQUIRE_FALSE(chosen_rule.isBasis);
    REQUIRE(chosen_rule.ruleName == "bell_to_cnot_h");

    const auto &chosen_rule_h = result.optimizedMap.at(h);
    REQUIRE_FALSE(chosen_rule_h.isBasis);
    REQUIRE(chosen_rule_h.ruleName == "h_to_rz_rx_rz");
}
