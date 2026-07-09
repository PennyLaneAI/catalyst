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
