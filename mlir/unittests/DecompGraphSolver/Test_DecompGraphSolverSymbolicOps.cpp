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
    const OperatorNode h{"H[][1]{}"};
    const OperatorNode adjH = makeAdjoint(h);

    REQUIRE(adjH.id == "Adjoint(H[][1]{})");
    REQUIRE(adjH.adjoint);
    REQUIRE(adjH != h);

    // cancel_adjoint: Adjoint(Adjoint(H)) == H
    REQUIRE(makeAdjoint(adjH) == h);
    REQUIRE_FALSE(makeAdjoint(adjH).adjoint);
}

TEST_CASE("Test makeAdjointRule", "[DecompGraph::Core]")
{
    const OperatorNode rot{"Rot[f64,f64,f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};
    const OperatorNode ry{"RY[f64][1]{}"};
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

TEST_CASE("Test DecompositionGraph adjoint rules from base rules", "[DecompGraph::Solver]")
{
    const OperatorNode rot{"Rot[f64,f64,f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};
    const OperatorNode ry{"RY[f64][1]{}"};

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
    const OperatorNode h{"H[][1]{}"};
    const OperatorNode adjH = makeAdjoint(h);

    const WeightedGateset gateset{{{h, 1.0}}};
    const std::vector<RuleNode> rules{
        {"h_is_basis", h, {}},              // empty rule
        {"self_adjoint_H", adjH, {{h, 1}}}, // adjoint output, must not be mirrored!!
    };

    const DecompositionGraph graph({h}, gateset, rules);

    REQUIRE(graph.getNumRules() == 2);
    REQUIRE(graph.getAllRulesFor(adjH).size() == 1);
    REQUIRE(graph.getAllRulesFor(adjH)[0].name == "self_adjoint_H");
}

TEST_CASE("Test Adjoint: self_adjoint (Adjoint(H) -> H)", "[DecompGraph::Solver]")
{
    const OperatorNode h{"H[][1]{}"};
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
    const OperatorNode rx{"RX[f64][1]{}"};
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

TEST_CASE("Test Adjoint: multiple rules and the solver should pick the cheapest",
          "[DecompGraph::Solver]")
{
    const OperatorNode rot{"Rot[f64,f64,f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};
    const OperatorNode ry{"RY[f64][1]{}"};
    const OperatorNode e{"E[][1]{}"};

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
    const OperatorNode myOp{"MyOp[][2]{}"};
    const OperatorNode a{"A[][1]{}"};
    const OperatorNode b{"B[][1]{}"};

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

TEST_CASE("Test makeControlled/makeAdjoint canonical ids", "[DecompGraph::Core]")
{
    const OperatorNode rx{"RX[f64][1]{}"};

    // Control folds into the id, accumulates, and is serialized control-outermost.
    REQUIRE(makeControlled(rx).id == "C(1, RX[f64][1]{})");
    REQUIRE(makeControlled(rx).numControlWires == 1);
    REQUIRE(makeControlled(rx, 2).id == "C(2, RX[f64][1]{})");
    REQUIRE(makeControlled(makeControlled(rx)) == makeControlled(rx, 2)); // accumulate, not nest

    // Adjoint and control commute: both application orders yield the same canonical node.
    REQUIRE(makeControlled(makeAdjoint(rx)) == makeAdjoint(makeControlled(rx)));
    REQUIRE(makeControlled(makeAdjoint(rx)).id == "C(1, Adjoint(RX[f64][1]{}))");
    REQUIRE(makeControlled(makeAdjoint(rx)).adjoint);
    REQUIRE(makeControlled(makeAdjoint(rx)).numControlWires == 1);

    // withoutControls strips only the controls, keeping adjoint; double adjoint still cancels.
    REQUIRE(withoutControls(makeControlled(makeAdjoint(rx))) == makeAdjoint(rx));
    REQUIRE(makeAdjoint(makeAdjoint(makeControlled(rx))) == makeControlled(rx));
}

TEST_CASE("Test DecompositionGraph synthesizes controlled rules from base rules",
          "[DecompGraph::Solver]")
{
    const OperatorNode rot{"Rot[f64,f64,f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};
    const OperatorNode ry{"RY[f64][1]{}"};

    const WeightedGateset gateset{{{makeControlled(rz), 1.0}, {makeControlled(ry), 1.0}}};
    const std::vector<RuleNode> rules{{"rot_decomp", rot, {{rz, 2}, {ry, 1}}}};

    // Controlled(Rot) is a root, so the builder synthesizes its controlled decomposition.
    const DecompositionGraph graph({makeControlled(rot)}, gateset, rules);

    REQUIRE(graph.getNumRules() == 2);
    REQUIRE(graph.hasOperator(makeControlled(rot)));

    const auto &ctrlRules = graph.getAllRulesFor(makeControlled(rot));
    REQUIRE(ctrlRules.size() == 1);
    REQUIRE(ctrlRules[0].name == "rot_decomp_controlled_1");
    REQUIRE(ctrlRules[0].origin == RuleOrigin::ControlGenerated);
    REQUIRE(ctrlRules[0].output == makeControlled(rot));
    REQUIRE(ctrlRules[0].inputs[0].op == makeControlled(rz));
    REQUIRE(ctrlRules[0].inputs[1].op == makeControlled(ry));
}

TEST_CASE("Test Controlled: solver picks the cheaper", "[DecompGraph::Solver]")
{
    const OperatorNode rot{"Rot[f64,f64,f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};
    const OperatorNode ry{"RY[f64][1]{}"};
    const OperatorNode e{"E[][1]{}"};

    const std::vector<RuleNode> baseRules{{"rot_decomp", rot, {{rz, 2}, {ry, 1}}}};

    SECTION("rot_decomp_controlled_1 is cheaper")
    {
        const WeightedGateset gateset{
            {{makeControlled(rz), 1.0}, {makeControlled(ry), 1.0}, {e, 10.0}}};
        std::vector<RuleNode> rules = baseRules;
        rules.push_back({"crot_direct", makeControlled(rot), {{e, 1}}}); // cost 10

        const DecompositionGraph graph({makeControlled(rot)}, gateset, rules);

        // Both an explicit controlled rule and the synthesized one exist for Controlled(Rot).
        REQUIRE(graph.getAllRulesFor(makeControlled(rot)).size() == 2);

        DecompositionSolver solver(graph);
        const auto result = solver.solve();
        const auto &chosen = result.at(makeControlled(rot));
        REQUIRE(chosen.ruleName == "rot_decomp_controlled_1");
        REQUIRE(chosen.origin == RuleOrigin::ControlGenerated);
        REQUIRE(chosen.totalCost == 3.0);
        REQUIRE(chosen.basisCounts.at(makeControlled(rz)) == 2);
        REQUIRE(chosen.basisCounts.at(makeControlled(ry)) == 1);
    }

    SECTION("crot_direct is cheaper")
    {
        const WeightedGateset gateset{{{makeControlled(rz), 1.0}, {makeControlled(ry), 1.0}}};
        std::vector<RuleNode> rules = baseRules;
        rules.push_back({"crot_direct", makeControlled(rot), {{makeControlled(rz), 1}}}); // cost 1

        const DecompositionGraph graph({makeControlled(rot)}, gateset, rules);
        DecompositionSolver solver(graph);
        const auto result = solver.solve();
        const auto &chosen = result.at(makeControlled(rot));
        REQUIRE(chosen.ruleName == "crot_direct");
        REQUIRE(chosen.origin == RuleOrigin::Default);
        REQUIRE(chosen.totalCost == 1.0);
    }
}

TEST_CASE("Test Controlled: multi-control synthesis", "[DecompGraph::Solver]")
{
    const OperatorNode rot{"Rot[f64,f64,f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};
    const OperatorNode ry{"RY[f64][1]{}"};

    // C(2, Rot): two control wires applied to every produced gate.
    const OperatorNode ccRot = makeControlled(rot, 2);
    const WeightedGateset gateset{{{makeControlled(rz, 2), 1.0}, {makeControlled(ry, 2), 1.0}}};
    const std::vector<RuleNode> rules{{"rot_decomp", rot, {{rz, 2}, {ry, 1}}}};

    const DecompositionGraph graph({ccRot}, gateset, rules);

    const auto &ctrlRules = graph.getAllRulesFor(ccRot);
    REQUIRE(ctrlRules.size() == 1);
    REQUIRE(ctrlRules[0].name == "rot_decomp_controlled_2");
    REQUIRE(ctrlRules[0].inputs[0].op == makeControlled(rz, 2));

    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    const auto &chosen = result.at(ccRot);
    REQUIRE(chosen.origin == RuleOrigin::ControlGenerated);
    REQUIRE(chosen.totalCost == 3.0);
    REQUIRE(chosen.basisCounts.at(makeControlled(rz, 2)) == 2);
    REQUIRE(chosen.basisCounts.at(makeControlled(ry, 2)) == 1);
}

TEST_CASE("Test Controlled: control pushed through a decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode myOp{"MyOp[][2]{}"};
    const OperatorNode a{"A[][1]{}"};
    const OperatorNode b{"B[][1]{}"};

    const WeightedGateset gateset{{{makeControlled(a), 1.0}, {makeControlled(b), 1.0}}};
    const std::vector<RuleNode> rules{{"myop_decomp", myOp, {{a, 1}, {b, 1}}}};

    const DecompositionGraph graph({makeControlled(myOp)}, gateset, rules);

    REQUIRE(graph.getAllRulesFor(makeControlled(myOp)).size() == 1);

    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    const auto &chosen = result.at(makeControlled(myOp));
    REQUIRE(chosen.ruleName == "myop_decomp_controlled_1");
    REQUIRE(chosen.origin == RuleOrigin::ControlGenerated);
    REQUIRE(chosen.totalCost == 2.0);
    REQUIRE(chosen.basisCounts.at(makeControlled(a)) == 1);
    REQUIRE(chosen.basisCounts.at(makeControlled(b)) == 1);
}

TEST_CASE("Test Controlled: suppressed Ctrl rule for special-cased operators",
          "[DecompGraph::Solver]")
{
    // `name` must be set so `isCtrlRuleRequired` can special-case GlobalPhase; program ops get it
    // from `getOperatorName()`. Field order: {id, adjoint, numControlWires, name}.
    const OperatorNode globalPhase{"GlobalPhase[f64][0]{}", false, 0, "GlobalPhase"};
    const OperatorNode rz{"RZ[f64][1]{}"};

    const WeightedGateset gateset{{{makeControlled(rz), 1.0}}};
    const std::vector<RuleNode> rules{
        {"gp_decomp", globalPhase, {{rz, 1}}},
        {"cgp_direct", makeControlled(globalPhase), {{makeControlled(rz), 1}}},
    };

    const DecompositionGraph graph({makeControlled(globalPhase)}, gateset, rules);

    // Control-each-gate is suppressed for GlobalPhase, so only the dedicated rule survives.
    const auto &ctrlRules = graph.getAllRulesFor(makeControlled(globalPhase));
    REQUIRE(ctrlRules.size() == 1);
    REQUIRE(ctrlRules[0].name == "cgp_direct");
    REQUIRE(ctrlRules[0].origin == RuleOrigin::Default);
}

TEST_CASE("Test Controlled+Adjoint: dedicated nested rule (Pathway 1)", "[DecompGraph::Solver]")
{
    const OperatorNode rx{"RX[f64][1]{}"};
    const OperatorNode crx{"CRX[f64][2]{}"};
    const OperatorNode cAdjRx = makeControlled(makeAdjoint(rx)); // C(Adjoint(RX))

    // A dedicated rule registered against the nested operator decomposes it directly.
    const WeightedGateset gateset{{{crx, 1.0}}};
    const std::vector<RuleNode> rules{{"c_adj_rx", cAdjRx, {{crx, 1}}}};

    const DecompositionGraph graph({cAdjRx}, gateset, rules);
    DecompositionSolver solver(graph);
    const auto result = solver.solve();

    const auto &chosen = result.at(cAdjRx);
    REQUIRE(chosen.ruleName == "c_adj_rx");
    REQUIRE(chosen.origin == RuleOrigin::Default);
    REQUIRE(chosen.totalCost == 1.0);
    REQUIRE(chosen.basisCounts.at(crx) == 1);
}

TEST_CASE("Test Controlled+Adjoint: synthesized nested decomposition (Pathway 2)",
          "[DecompGraph::Solver]")
{
    const OperatorNode myOp{"MyOp[][2]{}"};
    const OperatorNode a{"A[][1]{}"};
    const OperatorNode b{"B[][1]{}"};

    const WeightedGateset gateset{{{makeControlled(a), 1.0}, {makeControlled(b), 1.0}}};
    const std::vector<RuleNode> rules{
        {"myop_decomp", myOp, {{a, 1}, {b, 1}}},
        // self-adjoint rules so the adjointed produced gates bottom out at C(A) / C(B).
        {"self_adjoint_A", makeAdjoint(a), {{a, 1}}},
        {"self_adjoint_B", makeAdjoint(b), {{b, 1}}},
    };

    // Root C(Adjoint(MyOp)): no dedicated rule. The builder synthesizes it by first adjointing the
    // base decomposition (adjoint gen) and then controlling that (controlled gen).
    const OperatorNode root = makeControlled(makeAdjoint(myOp));
    const DecompositionGraph graph({root}, gateset, rules);

    REQUIRE(graph.getAllRulesFor(root).size() == 1);

    DecompositionSolver solver(graph);
    const auto result = solver.solve();
    REQUIRE(result.find(root) != result.end());
    const auto &chosen = result.at(root);
    REQUIRE(chosen.ruleName == "myop_decomp_adjoint_controlled_1");
    REQUIRE(chosen.origin == RuleOrigin::ControlGenerated);
    REQUIRE(chosen.totalCost == 2.0);
    REQUIRE(chosen.basisCounts.at(makeControlled(a)) == 1);
    REQUIRE(chosen.basisCounts.at(makeControlled(b)) == 1);
}
