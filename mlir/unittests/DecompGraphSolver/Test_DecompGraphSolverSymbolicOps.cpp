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

TEST_CASE("Test makeAdjoint and cancels on double application", "[DecompGraph::Core]")
{
    const OperatorNode h{"H", 1, 0, false};
    const OperatorNode adjH = makeAdjoint(h);

    REQUIRE(adjH.name == "H");
    REQUIRE(adjH.adjoint);
    REQUIRE(adjH != h);

    // cancel_adjoint: Adjoint(Adjoint(H)) == H
    REQUIRE(makeAdjoint(adjH) == h);
    REQUIRE_FALSE(makeAdjoint(adjH).adjoint);
}

TEST_CASE("Test makeAdjointRule", "[DecompGraph::Core]")
{
    const OperatorNode rot{"Rot", 1, 3, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};
    const RuleNode base{"rot_decomp", rot, {{rz, 2}, {ry, 1}}};

    const RuleNode adj = makeAdjointRule(base);

    REQUIRE(adj.name == "rot_decomp_adjoint");
    REQUIRE(adj.origin == RuleOrigin::AdjointGenerated);
    REQUIRE(adj.output == makeAdjoint(rot));
    REQUIRE(adj.output.adjoint);
    REQUIRE(adj.inputs.size() == 2);
    REQUIRE(adj.inputs[0].op == makeAdjoint(rz));
    REQUIRE(adj.inputs[0].op.adjoint);
    REQUIRE(adj.inputs[0].multiplicity == 2);
    REQUIRE(adj.inputs[1].op == makeAdjoint(ry));
    REQUIRE(adj.inputs[1].multiplicity == 1);
}

TEST_CASE("Test DecompositionGraph adjoint rules from base rules",
          "[DecompGraph::Solver]")
{
    const OperatorNode rot{"Rot", 1, 3, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};

    const WeightedGateset gateset{{{rz, 1.0}, {ry, 1.0}}};
    const std::vector<RuleNode> rules{{"rot_decomp", rot, {{rz, 2}, {ry, 1}}}};

    // Adjoint(Rot) is a root, so the builder should synthesize its adjoint decomposition
    const DecompositionGraph graph({makeAdjoint(rot)}, gateset, rules);

    REQUIRE(graph.getNumRules() == 2);
    REQUIRE(graph.hasOperator(makeAdjoint(rot)));

    const auto &adjRules = graph.getAllRulesFor(makeAdjoint(rot));
    REQUIRE(adjRules.size() == 1);
    REQUIRE(adjRules[0].name == "rot_decomp_adjoint");
    REQUIRE(adjRules[0].origin == RuleOrigin::AdjointGenerated);
    REQUIRE(adjRules[0].output == makeAdjoint(rot));
    REQUIRE(adjRules[0].inputs[0].op == makeAdjoint(rz));
    REQUIRE(adjRules[0].inputs[1].op == makeAdjoint(ry));
}

TEST_CASE("Test DecompositionGraph does not synthesize adjoint rules for empty or adjoint rules",
          "[DecompGraph::Solver]")
{
    const OperatorNode h{"H", 1, 0, false};
    const OperatorNode adjH = makeAdjoint(h);

    const WeightedGateset gateset{{{h, 1.0}}};
    const std::vector<RuleNode> rules{
        {"h_is_basis", h, {}},         // empty rule
        {"self_adjoint_H", adjH, {{h, 1}}}, // adjoint output, must not be mirrored!!
    };

    const DecompositionGraph graph({h}, gateset, rules);

    REQUIRE(graph.getNumRules() == 2);
    REQUIRE(graph.getAllRulesFor(adjH).size() == 1);
    REQUIRE(graph.getAllRulesFor(adjH)[0].name == "self_adjoint_H");
}

TEST_CASE("Test Adjoint: self_adjoint (Adjoint(H) -> H)", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H", 1, 0, false};
    const OperatorNode adjH = makeAdjoint(h);

    const WeightedGateset gateset{{{h, 1.0}}};
    const std::vector<RuleNode> rules{{"self_adjoint_H", adjH, {{h, 1}}}};

    const DecompositionGraph graph({adjH}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();

    REQUIRE(result.find(adjH) != result.end());
    const auto &chosen = result.at(adjH);
    REQUIRE_FALSE(chosen.isBasis);
    REQUIRE(chosen.ruleName == "self_adjoint_H");
    REQUIRE(chosen.origin == RuleOrigin::Default);
    REQUIRE(chosen.totalCost == 1.0);
    REQUIRE(chosen.basisCounts.at(h) == 1);

    REQUIRE(graph.getAllRulesFor(adjH).size() == 1);
}

TEST_CASE("Test Adjoint: adjoint_rotation (Adjoint(RX) -> RX)", "[DecompGraph::Solver]")
{
    const OperatorNode rx{"RX", 1, 1, false};
    const OperatorNode adjRX = makeAdjoint(rx);

    const WeightedGateset gateset{{{rx, 1.0}}};
    const std::vector<RuleNode> rules{{"adjoint_rotation_RX", adjRX, {{rx, 1}}}};

    const DecompositionGraph graph({adjRX}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();

    const auto &chosen = result.at(adjRX);
    REQUIRE(chosen.ruleName == "adjoint_rotation_RX");
    REQUIRE(chosen.totalCost == 1.0);
    REQUIRE(chosen.basisCounts.at(rx) == 1);
}

TEST_CASE("Test Adjoint: multiple rules and the solver should pick the cheapest", "[DecompGraph::Solver]")
{
    const OperatorNode rot{"Rot", 1, 3, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};
    const OperatorNode e{"E", 1, 0, false};

    const std::vector<RuleNode> commonRules{
        {"rot_decomp", rot, {{rz, 2}, {ry, 1}}},
        {"adjoint_rotation_RZ", makeAdjoint(rz), {{rz, 1}}},
        {"adjoint_rotation_RY", makeAdjoint(ry), {{ry, 1}}},
    };

    SECTION("rot_decomp_adjoint is cheaper")
    {
        const WeightedGateset gateset{{{rz, 1.0}, {ry, 1.0}, {e, 10.0}}};
        std::vector<RuleNode> rules = commonRules;
        rules.push_back({"_adjoint_rot", makeAdjoint(rot), {{e, 1}}}); // cost 10

        const DecompositionGraph graph({makeAdjoint(rot)}, gateset, rules);

        // Both an explicit adjoint rule and the synthesized one exist for Adjoint(Rot).
        REQUIRE(graph.getAllRulesFor(makeAdjoint(rot)).size() == 2);

        DecompositionSolver solver(graph);
        const auto result = solver.solve();
        const auto &chosen = result.at(makeAdjoint(rot));
        REQUIRE(chosen.ruleName == "rot_decomp_adjoint");
        REQUIRE(chosen.origin == RuleOrigin::AdjointGenerated);
        REQUIRE(chosen.totalCost == 3.0);
        REQUIRE(chosen.basisCounts.at(rz) == 2);
        REQUIRE(chosen.basisCounts.at(ry) == 1);
    }

    SECTION("_adjoint_rot is cheaper")
    {
        const WeightedGateset gateset{{{rz, 1.0}, {ry, 1.0}}};
        std::vector<RuleNode> rules = commonRules;
        rules.push_back({"_adjoint_rot", makeAdjoint(rot), {{rz, 1}}}); // cost 1

        const DecompositionGraph graph({makeAdjoint(rot)}, gateset, rules);
        DecompositionSolver solver(graph);
        const auto result = solver.solve();
        const auto &chosen = result.at(makeAdjoint(rot));
        REQUIRE(chosen.ruleName == "_adjoint_rot");
        REQUIRE(chosen.origin == RuleOrigin::Default);
        REQUIRE(chosen.totalCost == 1.0);
    }
}

TEST_CASE("Test Adjoint: adjoint pushed through a decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode myOp{"MyOp", 2, 0, false};
    const OperatorNode a{"A", 1, 0, false};
    const OperatorNode b{"B", 1, 0, false};

    const WeightedGateset gateset{{{a, 1.0}, {b, 1.0}}};
    const std::vector<RuleNode> rules{
        {"myop_decomp", myOp, {{a, 1}, {b, 1}}},
        // Define self_adjoint rules so the adjointed produced gates can resolve:
        {"self_adjoint_A", makeAdjoint(a), {{a, 1}}},
        {"self_adjoint_B", makeAdjoint(b), {{b, 1}}},
    };

    const DecompositionGraph graph({makeAdjoint(myOp)}, gateset, rules);

    REQUIRE(graph.getAllRulesFor(makeAdjoint(myOp)).size() == 1);

    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    const auto &chosen = result.at(makeAdjoint(myOp));
    REQUIRE(chosen.ruleName == "myop_decomp_adjoint");
    REQUIRE(chosen.origin == RuleOrigin::AdjointGenerated);
    REQUIRE(chosen.totalCost == 2.0);
    REQUIRE(chosen.basisCounts.at(a) == 1);
    REQUIRE(chosen.basisCounts.at(b) == 1);
}
