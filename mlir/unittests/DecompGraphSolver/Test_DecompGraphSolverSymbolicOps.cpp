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

TEST_CASE("Test makeControlled", "[DecompGraph::Core]")
{
    const OperatorNode rx{"RX", 1, 1, false};
    const OperatorNode crx = makeControlled(rx);

    REQUIRE(crx.name == "RX");
    REQUIRE(crx.numControlWires == 1);
    REQUIRE(crx != rx);

    // Controls accumulate into a multi-controlled operator.
    REQUIRE(makeControlled(crx).numControlWires == 2);
    REQUIRE(makeControlled(rx, 3).numControlWires == 3);
}

TEST_CASE("Test makeControlledRule controls the output and every input", "[DecompGraph::Core]")
{
    const OperatorNode rot{"Rot", 1, 3, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};
    const RuleNode base{"rot_decomp", rot, {{rz, 2}, {ry, 1}}};

    const RuleNode ctrl = makeControlledRule(base, 1);

    REQUIRE(ctrl.name == "rot_decomp_controlled_1");
    REQUIRE(ctrl.origin == RuleOrigin::ControlGenerated);
    REQUIRE(ctrl.output == makeControlled(rot));
    REQUIRE(ctrl.output.numControlWires == 1);
    REQUIRE(ctrl.inputs.size() == 2);
    REQUIRE(ctrl.inputs[0].op == makeControlled(rz));
    REQUIRE(ctrl.inputs[0].op.numControlWires == 1);
    REQUIRE(ctrl.inputs[0].multiplicity == 2);
    REQUIRE(ctrl.inputs[1].op == makeControlled(ry));
    REQUIRE(ctrl.inputs[1].multiplicity == 1);
}

TEST_CASE("Test DecompositionGraph synthesizes controlled rules from base rules",
          "[DecompGraph::Solver]")
{
    const OperatorNode rot{"Rot", 1, 3, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};

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
    const OperatorNode rot{"Rot", 1, 3, false};
    const OperatorNode rz{"RZ", 1, 1, false};
    const OperatorNode ry{"RY", 1, 1, false};
    const OperatorNode e{"E", 1, 0, false};

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

TEST_CASE("Test Controlled: control pushed through a decomposition", "[DecompGraph::Solver]")
{
    const OperatorNode myOp{"MyOp", 2, 0, false};
    const OperatorNode a{"A", 1, 0, false};
    const OperatorNode b{"B", 1, 0, false};

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
    const OperatorNode globalPhase{"GlobalPhase", -1, -1, false};
    const OperatorNode rz{"RZ", 1, 1, false};

    const WeightedGateset gateset{{{makeControlled(rz), 1.0}}};
    const std::vector<RuleNode> rules{
        {"gp_decomp", globalPhase, {{rz, 1}}},
        {"cgp_direct", makeControlled(globalPhase), {{makeControlled(rz), 1}}},
    };

    const DecompositionGraph graph({makeControlled(globalPhase)}, gateset, rules);

    const auto &ctrlRules = graph.getAllRulesFor(makeControlled(globalPhase));
    REQUIRE(ctrlRules.size() == 1);
    REQUIRE(ctrlRules[0].name == "cgp_direct");
    REQUIRE(ctrlRules[0].origin == RuleOrigin::Default);
}
