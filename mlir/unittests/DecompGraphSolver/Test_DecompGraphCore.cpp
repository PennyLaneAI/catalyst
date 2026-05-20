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

#include "DGTypes.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace Catch::Matchers;
using namespace DecompGraph::Core;

TEST_CASE("Test OperatorNode construction", "[DecompGraph::Core]") {
    const OperatorNode op1{"H", 1, 0, false};
    const OperatorNode op2{"CNOT", 2, 0, false};
    const OperatorNode op3{"RX", 1, 1, false};
    const OperatorNode op4{"RZ", 1, 1, true};

    REQUIRE(op1.name == "H");
    REQUIRE(op1.numWires == 1);
    REQUIRE(op1.numParams == 0);
    REQUIRE(op1.adjoint == false);

    REQUIRE(op2.name == "CNOT");
    REQUIRE(op2.numWires == 2);
    REQUIRE(op2.numParams == 0);
    REQUIRE(op2.adjoint == false);

    REQUIRE(op3.name == "RX");
    REQUIRE(op3.numWires == 1);
    REQUIRE(op3.numParams == 1);
    REQUIRE(op3.adjoint == false);

    REQUIRE(op4.name == "RZ");
    REQUIRE(op4.numWires == 1);
    REQUIRE(op4.numParams == 1);
    REQUIRE(op4.adjoint == true);
}

TEST_CASE("Test OperatorNode equality operator", "[DecompGraph::Core]") {
    const OperatorNode op1{"H", 1, 0, false};
    const OperatorNode op2{"H", 1, 0, false};
    const OperatorNode op3{"H", 1, 0, true};
    const OperatorNode op4{"CNOT", 2, 0, false};

    REQUIRE(op1 == op2);
    REQUIRE_FALSE(op1 == op3);
    REQUIRE_FALSE(op1 == op4);
}

TEST_CASE("Test OperatorNodeHash", "[DecompGraph::Core]") {
    const OperatorNode op1{"H", 1, 0, false};
    const OperatorNode op2{"H", 1, 0, false};
    const OperatorNode op3{"H", 1, 0, true};
    const OperatorNode op4{"CNOT", 2, 0, false};

    const OperatorNodeHash hashFunc;
    REQUIRE(hashFunc(op1) == hashFunc(op2));
    REQUIRE(hashFunc(op1) == hashFunc(op3));
    REQUIRE(hashFunc(op1) != hashFunc(op4));
}

TEST_CASE("Test OperatorNode in unordered_map", "[DecompGraph::Core]") {
    std::unordered_map<OperatorNode, double, OperatorNodeHash> opMap;
    const OperatorNode op1{"H", 1, 0, false};
    const OperatorNode op2{"CNOT", 2, 0, false};
    const OperatorNode op3{"RX", 1, 1, false};

    opMap[op1] = 1.0;
    opMap[op2] = 2.0;

    REQUIRE(opMap[op1] == 1.0);
    REQUIRE(opMap[op2] == 2.0);
    REQUIRE(opMap.find(op3) == opMap.end());
}

TEST_CASE("Test RuleNode construction", "[DecompGraph::Core]") {
    const auto h = OperatorNode{"H"};
    const auto rz = OperatorNode{"RZ"};
    const auto rx = OperatorNode{"RX"};

    const RuleNode h_to_rz_rx_rz{"h_to_rz_rx_rz", h, {{rz, 2}, {rx, 1}}};
    REQUIRE(h_to_rz_rx_rz.name == "h_to_rz_rx_rz");
    REQUIRE(h_to_rz_rx_rz.output == h);
    REQUIRE(h_to_rz_rx_rz.inputs.size() == 2);
    REQUIRE(h_to_rz_rx_rz.inputs[0].op == rz);
    REQUIRE(h_to_rz_rx_rz.inputs[0].multiplicity == 2);
    REQUIRE(h_to_rz_rx_rz.inputs[1].op == rx);
    REQUIRE(h_to_rz_rx_rz.inputs[1].multiplicity == 1);
}

TEST_CASE("Test WeightedGateset construction and contains", "[DecompGraph::Core]") {
    const OperatorNode h{"H"};
    const OperatorNode cnot{"CNOT"};
    const OperatorNode rx{"RX"};

    const WeightedGateset gateset{{{h, 1.0}, {cnot, 2.0}}};

    REQUIRE(gateset.contains(h));
    REQUIRE(gateset.contains(cnot));
    REQUIRE_FALSE(gateset.contains(rx));

    REQUIRE(gateset.getCost(h) == 1.0);
    REQUIRE(gateset.getCost(cnot) == 2.0);
}

TEST_CASE("Test ChosenDecompRule construction", "[DecompGraph::Core]") {
    const OperatorNode h{"H"};
    const OperatorNode rz{"RZ"};
    const OperatorNode rx{"RX"};

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
