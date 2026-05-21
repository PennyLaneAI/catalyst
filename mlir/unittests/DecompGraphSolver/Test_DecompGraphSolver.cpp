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

#include <algorithm>
#include <iostream>

#include "DGBuilder.hpp"
#include "DGSolver.hpp"
#include "DGTypes.hpp"
#include "DGUtils.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

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

    REQUIRE(graph.getRootOps().size() == 1);
    REQUIRE(graph.getRootOps()[0] == h);
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

TEST_CASE("Test DecompositionSolver solve method with incomplete gates in Gateset",
          "[DecompGraph::Solver]")
{
    const auto h = OperatorNode{"H", 1, 0, false};
    const auto h_gateset = OperatorNode{"H"};
    const WeightedGateset gateset{{{h_gateset, 1.0}}};
    const std::vector<RuleNode> rules{
        {"h_to_h", h, {{h, 1}}},
    };

    const DecompositionGraph graph({h}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    REQUIRE(result.at(h).isBasis);
    REQUIRE(result.at(h_gateset).isBasis);

    const std::vector<RuleNode> rules_with_h_gateset{
        {"h_to_h", h_gateset, {{h_gateset, 1}}},
    };

    const DecompositionGraph graph_with_h_gateset({h}, gateset, rules_with_h_gateset);
    DecompositionSolver solver_with_h_gateset(graph_with_h_gateset);
    const auto result_with_h_gateset = solver_with_h_gateset.solve();
    REQUIRE(result_with_h_gateset.at(h).isBasis);
}

TEST_CASE("Do not solve for target gates", "[DecompGraph::Solver]")
{
    const auto h = OperatorNode{"H", 1, 0, false};
    const auto rz = OperatorNode{"RZ", 1, 1, false};

    const WeightedGateset gateset{{{h, 2.0}, {rz, 1.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz", h, {{rz, 1}}},
    };

    const DecompositionGraph graph({h, rz}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto solutions = solver.solve();
    REQUIRE(solutions.size() == 2);
    REQUIRE(solutions.at(h).isBasis);
    REQUIRE(solutions.at(h).totalCost == 2.0);
    REQUIRE(solutions.at(rz).isBasis);
    REQUIRE(solutions.at(rz).totalCost == 1.0);
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
    REQUIRE(copyConstructedGraph.getRootOps() == graph.getRootOps());
    REQUIRE(copyConstructedGraph.getGateset().ops == graph.getGateset().ops);
    REQUIRE(copyConstructedGraph.getRules().size() == graph.getRules().size());
    REQUIRE(copyConstructedGraph.getRules()[0].name == graph.getRules()[0].name);

    // Test copy assignment operator
    DecompositionGraph copyAssignedGraph = graph;
    REQUIRE(copyAssignedGraph.getRootOps() == graph.getRootOps());
    REQUIRE(copyAssignedGraph.getGateset().ops == graph.getGateset().ops);
    REQUIRE(copyAssignedGraph.getRules().size() == graph.getRules().size());
    REQUIRE(copyAssignedGraph.getRules()[0].name == graph.getRules()[0].name);

    // Test move constructor
    DecompositionGraph moveConstructedGraph(std::move(graph));
    REQUIRE(moveConstructedGraph.getRootOps() == copyConstructedGraph.getRootOps());
    REQUIRE(moveConstructedGraph.getGateset().ops == copyConstructedGraph.getGateset().ops);
    REQUIRE(moveConstructedGraph.getRules().size() == copyConstructedGraph.getRules().size());
    REQUIRE(moveConstructedGraph.getRules()[0].name == copyConstructedGraph.getRules()[0].name);

    // Test move assignment operator
    DecompositionGraph moveAssignedGraph = std::move(copyAssignedGraph);
    REQUIRE(moveAssignedGraph.getRootOps() == copyConstructedGraph.getRootOps());
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
    REQUIRE(result.size() == 4);
    const auto &chosen_rule = result.at(h);
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

    const auto &rz_rule = result.at(rz);
    REQUIRE(rz_rule.isBasis);
    const auto &rx_rule = result.at(rx);
    REQUIRE(rx_rule.isBasis);
    const auto &ry_rule = result.at(ry);
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
    REQUIRE(result.size() == 6);
    const auto &chosen_rule = result.at(customBellOp);
    REQUIRE_FALSE(chosen_rule.isBasis);
    REQUIRE(chosen_rule.ruleName == "bell_to_cnot_h");

    const auto &chosen_rule_h = result.at(h);
    REQUIRE_FALSE(chosen_rule_h.isBasis);
    REQUIRE(chosen_rule_h.ruleName == "h_to_rz_rx_rz");

    std::vector<std::string> expected_rule_names{"bell_to_cnot_h", "h_to_rz_rx_rz", "swap_to_cnot"};
    for (const auto &[op, entry] : result) {
        if (op == customBellOp) {
            REQUIRE(std::find(expected_rule_names.begin(), expected_rule_names.end(),
                              entry.ruleName) != expected_rule_names.end());
        }
        else if (op == h) {
            REQUIRE(std::find(expected_rule_names.begin(), expected_rule_names.end(),
                              entry.ruleName) != expected_rule_names.end());
        }
        else if (op == swap) {
            REQUIRE(std::find(expected_rule_names.begin(), expected_rule_names.end(),
                              entry.ruleName) != expected_rule_names.end());
        }
        else if (op == rz || op == rx) {
            REQUIRE(entry.isBasis);
            REQUIRE(entry.ruleName == "BasisRule");
        }
        else if (op == ry) {
            REQUIRE(entry.isBasis);
            REQUIRE(entry.ruleName == "BasisRule");
        }
    }
}

TEST_CASE("Test GraphSolveError for unsolvable operator", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H", 1, 0, false};
    const OperatorNode rz{"RZ", 1, 1, false};

    const WeightedGateset gateset{{{rz, 1.0}}};

    const std::vector<RuleNode> rules{
        {"rz_to_rz", rz, {{rz, 1}}},
    };

    const DecompositionGraph graph({h}, gateset, rules);
    DecompositionSolver solver(graph);

    REQUIRE_THROWS_AS(solver.solve(), GraphSolverFailedError);
}

TEST_CASE("Test GraphSolveError for cyclic decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H", 1, 0, false};

    const WeightedGateset gateset{};

    const std::vector<RuleNode> rules{
        {"h_to_h", h, {{h, 1}}},
    };

    const DecompositionGraph graph({h}, gateset, rules);
    DecompositionSolver solver(graph);

    REQUIRE_THROWS_AS(solver.solve(), GraphSolverFailedError);
}

TEST_CASE("Test PauliX -> GlobalPhase(1), RX(1) decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode x{"X"};
    const OperatorNode globalPhase{"GlobalPhase"};
    const OperatorNode rx{"RX"};

    const WeightedGateset gateset{{{globalPhase, 1.0}, {rx, 1.0}}};

    const std::vector<RuleNode> rules{
        {"x_to_globalPhase_rx", x, {{globalPhase, 1}, {rx, 1}}},
        {"globalPhase_rx", globalPhase, {{rx, 1}}},
    };

    const DecompositionGraph graph({x}, gateset, rules);
    DecompositionSolver solver(graph);

    const auto result = solver.solve();
    REQUIRE(result.size() == 3);
    const auto &chosen_rule = result.at(x);
    REQUIRE_FALSE(chosen_rule.isBasis);
    REQUIRE(chosen_rule.ruleName == "x_to_globalPhase_rx");
    REQUIRE(chosen_rule.inputs.size() == 2);
    REQUIRE(chosen_rule.inputs[0].op == globalPhase);
    REQUIRE(chosen_rule.inputs[0].multiplicity == 1);
    REQUIRE(chosen_rule.inputs[1].op == rx);
    REQUIRE(chosen_rule.inputs[1].multiplicity == 1);
    REQUIRE(chosen_rule.totalCost == 1.0 * 1 + 1.0 * 1);
}

TEST_CASE("Test cyclic decomposition with multiple rules for the same operator",
          "[DecompGraph::Solver]")
{
    const OperatorNode hadamard{"Hadamard"};
    const OperatorNode globalPhase{"GlobalPhase"};
    const OperatorNode rx{"RX"};
    const OperatorNode rz{"RZ"};
    const OperatorNode ry{"RY"};
    const OperatorNode changeOpBasis{"ChangeOpBasis"};
    const OperatorNode pauliRot{"PauliRot"};
    const OperatorNode rot{"Rot"};

    const std::vector<RuleNode> rules{
        {"__builtin__ry_to_rz_cliff", ry, {{changeOpBasis, 1}}},
        {"__builtin__ry_to_rx_cliff", ry, {{changeOpBasis, 1}}},
        {"__builtin__ry_to_ppr", ry, {{pauliRot, 1}}},
        {"__builtin__ry_to_rz_rx", ry, {{rx, 1}, {rz, 2}}},
        {"__builtin__ry_to_rot", ry, {{rot, 1}}},
        {"__builtin__hadamard_to_rz_rx", hadamard, {{globalPhase, 1}, {rx, 1}, {rz, 2}}},
        {"__builtin__hadamard_to_rz_ry", hadamard, {{globalPhase, 1}, {ry, 1}, {rz, 1}}},
    };

    const WeightedGateset gateset{{{globalPhase, 1.0}, {rx, 1.0}, {rz, 1.0}}};
    const DecompositionGraph graph({hadamard}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto solutions = solver.solve();
    const auto &h_solution = solutions.at(hadamard);
    REQUIRE_FALSE(h_solution.isBasis);
    REQUIRE(h_solution.ruleName == "__builtin__hadamard_to_rz_rx");
    REQUIRE(h_solution.totalCost == 1.0 * 1 + 1.0 * 1 + 1.0 * 2);

    const WeightedGateset gateset2{{{globalPhase, 1.0}, {rx, 1.0}, {rz, 2.0}, {ry, 1.0}}};
    const DecompositionGraph graph2({hadamard}, gateset2, rules);
    DecompositionSolver solver2(graph2);
    const auto solutions2 = solver2.solve();
    const auto &h_solution2 = solutions2.at(hadamard);
    REQUIRE(h_solution2.ruleName == "__builtin__hadamard_to_rz_ry");
}

TEST_CASE("Test GraphBuilder with fixed decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H"};
    const OperatorNode rz{"RZ"};
    const OperatorNode rx{"RX"};

    const WeightedGateset gateset{{{rz, 1.0}, {rx, 3.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
        {"h_to_rx_rz_rx", h, {{rx, 1}, {rz, 2}}},
    };

    const FixedDecomps fixedDecomps{
        {h, {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}}},
    };

    DecompositionGraph graph({h}, gateset, rules, fixedDecomps);
    REQUIRE(graph.getAllRulesFor(h).size() == 1);
    REQUIRE(graph.getAllRulesFor(h)[0].name == "h_to_rz_rx_rz");
}

TEST_CASE("Test GraphBuilder with alternative decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H"};
    const OperatorNode rz{"RZ"};
    const OperatorNode rx{"RX"};

    const WeightedGateset gateset{{{rz, 1.0}, {rx, 3.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
    };

    const AltDecomps altDecomps{
        {h, {{"h_to_rx_rz_rx", h, {{rx, 1}, {rz, 2}}}, {"h_to_h", h, {{h, 1}}}}},
    };

    DecompositionGraph graph({h}, gateset, rules, {}, altDecomps);
    REQUIRE(graph.getAllRulesFor(h).size() == 3);
}

TEST_CASE("Test GraphSolver with fixed decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H"};
    const OperatorNode rz{"RZ"};
    const OperatorNode rx{"RX"};

    const WeightedGateset gateset{{{rz, 3.0}, {rx, 1.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
        {"h_to_rx_rz_rx", h, {{rx, 1}, {rz, 2}}},
    };

    const FixedDecomps fixedDecomps{
        {h, {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}}},
    };

    DecompositionGraph graph({h}, gateset, rules, fixedDecomps);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    REQUIRE(result.size() == 3);
    const auto &chosen_rule = result.at(h);
    REQUIRE_FALSE(chosen_rule.isBasis);
    REQUIRE(chosen_rule.ruleName == "h_to_rz_rx_rz");
}

TEST_CASE("Test GraphSolver with alternative decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H"};
    const OperatorNode rz{"RZ"};
    const OperatorNode rx{"RX"};

    const WeightedGateset gateset{{{rz, 1.0}, {rx, 3.0}}};

    const std::vector<RuleNode> rules{
        {"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}},
    };

    const AltDecomps altDecomps{
        {h, {{"h_to_rx_rz_rx", h, {{rx, 1}, {rz, 2}}}}},
    };

    DecompositionGraph graph({h}, gateset, rules, {}, altDecomps);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    REQUIRE(result.size() == 3);
    const auto &chosen_rule = result.at(h);
    REQUIRE_FALSE(chosen_rule.isBasis);
    REQUIRE(chosen_rule.ruleName == "h_to_rz_rx_rz");
}

TEST_CASE("Test GraphSolver with MultiRZ decompositions", "[DecompGraph::Solver]")
{
    const OperatorNode multiRZ3{"MultiRZ3"};
    const OperatorNode multiRZ5{"MultiRZ5"};
    const OperatorNode rz{"RZ"};

    const WeightedGateset gateset{{{rz, 1.0}}};

    const std::vector<RuleNode> rules{
        {"multiRZ3_to_rz", multiRZ3, {{rz, 3}}},
        {"multiRZ5_to_rz", multiRZ5, {{rz, 5}}},
    };

    const DecompositionGraph graph({multiRZ3, multiRZ5}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    REQUIRE(result.size() == 3);
    const auto &chosen_rule_multiRZ3 = result.at(multiRZ3);
    REQUIRE_FALSE(chosen_rule_multiRZ3.isBasis);
    REQUIRE(chosen_rule_multiRZ3.ruleName == "multiRZ3_to_rz");
    REQUIRE(chosen_rule_multiRZ3.totalCost == 1.0 * 3);
    const auto &chosen_rule_multiRZ5 = result.at(multiRZ5);
    REQUIRE_FALSE(chosen_rule_multiRZ5.isBasis);
    REQUIRE(chosen_rule_multiRZ5.ruleName == "multiRZ5_to_rz");
    REQUIRE(chosen_rule_multiRZ5.totalCost == 1.0 * 5);
}

TEST_CASE("Test GraphSolver with empty decomposition rules", "[DecompGraph::Solver]")
{
    const OperatorNode hadamard{"Hadamard"};
    const OperatorNode globalPhase{"GlobalPhase"};

    const WeightedGateset gateset{{{globalPhase, 1.0}}};

    const std::vector<RuleNode> rules{
        {"hadamard_to_globalPhase", hadamard, {}},
    };

    const DecompositionGraph graph({hadamard}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    REQUIRE(result.size() == 1);
    const auto &chosen_rule = result.at(hadamard);
    REQUIRE_FALSE(chosen_rule.isBasis);
    REQUIRE(chosen_rule.ruleName == "hadamard_to_globalPhase");
    REQUIRE(chosen_rule.inputs.empty());
    REQUIRE(chosen_rule.totalCost == 0.0);
}
