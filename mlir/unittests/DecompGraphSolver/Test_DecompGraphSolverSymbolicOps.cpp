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

TEST_CASE("Test makeAdjoint and cancels on double application",
          "[DecompGraph::Core]")
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
