// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_map>

#include "DGTypes.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace Catch::Matchers;
using namespace DecompGraph::Core;

TEST_CASE("Test OperatorNode construction", "[DecompGraph::Core]")
{
    const OperatorNode h{"Hadamard[][1]{}"};
    const OperatorNode cnot{"CNOT[][2]{}"};
    const OperatorNode rx{"RX[f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};

    REQUIRE(h.id == "Hadamard[][1]{}");
    REQUIRE(cnot.id == "CNOT[][2]{}");
    REQUIRE(rx.id == "RX[f64][1]{}");
    REQUIRE(rz.id == "RZ[f64][1]{}");
}

TEST_CASE("Test OperatorNode equality operator", "[DecompGraph::Core]")
{
    const OperatorNode h1{"Hadamard[][1]{}"};
    const OperatorNode h2{"Hadamard[][1]{}"};
    const OperatorNode cnot{"CNOT[][2]{}"};

    REQUIRE(h1 == h2);
    REQUIRE(h1 != cnot);
}

TEST_CASE("Test OperatorNodeHash", "[DecompGraph::Core]")
{
    const OperatorNode h1{"Hadamard[][1]{}"};
    const OperatorNode h2{"Hadamard[][1]{}"};
    const OperatorNode h3{"Hadamard[][1]{}"};
    const OperatorNode cnot{"CNOT[][2]{}"};

    const OperatorNodeHash hashFunc;
    REQUIRE(hashFunc(h1) == hashFunc(h2));
    REQUIRE(hashFunc(h1) == hashFunc(h3));
    REQUIRE(hashFunc(h1) != hashFunc(cnot));
}

TEST_CASE("Test OperatorNode in unordered_map", "[DecompGraph::Core]")
{
    std::unordered_map<OperatorNode, double, OperatorNodeHash> opMap;
    const OperatorNode h{"Hadamard[][1]{}"};
    const OperatorNode cnot{"CNOT[][2]{}"};
    const OperatorNode rx{"RX[f64][1]{}"};

    opMap[h] = 1.0;
    opMap[cnot] = 2.0;

    REQUIRE(opMap[h] == 1.0);
    REQUIRE(opMap[cnot] == 2.0);
    REQUIRE(opMap.find(rx) == opMap.end());
}

TEST_CASE("Test RuleNode construction", "[DecompGraph::Core]")
{
    const OperatorNode h{"Hadamard[][1]{}"};
    const OperatorNode rx{"RX[f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};

    const RuleNode h_to_rz_rx_rz{"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}};
    REQUIRE(h_to_rz_rx_rz.name == "h_to_rz_rx_rz");
    REQUIRE(h_to_rz_rx_rz.output == h);
    REQUIRE(h_to_rz_rx_rz.inputs.size() == 2);
    REQUIRE(h_to_rz_rx_rz.inputs[0].op == rz);
    REQUIRE(h_to_rz_rx_rz.inputs[0].multiplicity == 2);
    REQUIRE(h_to_rz_rx_rz.inputs[1].op == rx);
    REQUIRE(h_to_rz_rx_rz.inputs[1].multiplicity == 1);
}

TEST_CASE("Test WeightedGateset construction and contains", "[DecompGraph::Core]")
{
    const OperatorNode h{"Hadamard[][1]{}"};
    const OperatorNode cnot{"CNOT[][2]{}"};
    const OperatorNode rx{"RX[f64][1]{}"};

    const WeightedGateset gateset{{{h, 1.0}, {cnot, 2.0}}};

    REQUIRE(gateset.contains(h));
    REQUIRE(gateset.contains(cnot));
    REQUIRE_FALSE(gateset.contains(rx));

    REQUIRE(gateset.getCost(h) == 1.0);
    REQUIRE(gateset.getCost(cnot) == 2.0);
}

TEST_CASE("Test ChosenDecompRule construction", "[DecompGraph::Core]")
{
    const OperatorNode h{"Hadamard[][1]{}"};
    const OperatorNode rx{"RX[f64][1]{}"};
    const OperatorNode rz{"RZ[f64][1]{}"};

    const RuleTerm term1{rz, 2};
    const RuleTerm term2{rx, 1};

    ChosenDecompRule chosenRule{h, false, "h_to_rz_rx_rz", {term1, term2}, 3.0, {{rz, 2}, {rx, 1}}};

    REQUIRE(chosenRule.op == h);
    REQUIRE(chosenRule.isBasis == false);
    REQUIRE(chosenRule.ruleName == "h_to_rz_rx_rz");
    REQUIRE(chosenRule.inputs.size() == 2);
    REQUIRE(chosenRule.inputs[0].op == rz);
    REQUIRE(chosenRule.inputs[0].multiplicity == 2);
    REQUIRE(chosenRule.inputs[1].op == rx);
    REQUIRE(chosenRule.inputs[1].multiplicity == 1);
    REQUIRE(chosenRule.totalCost == 3.0);
    REQUIRE(chosenRule.basisCounts.size() == 2);
    REQUIRE(chosenRule.basisCounts[rz] == 2);
    REQUIRE(chosenRule.basisCounts[rx] == 1);
}
