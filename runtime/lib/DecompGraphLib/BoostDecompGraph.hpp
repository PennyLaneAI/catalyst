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
 * @file BoostDecompGraph.hpp
 * @brief This file defines the Boost Graph Library-based representation
 * of the decomposition graph.
 */
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <unordered_map>

#include "GraphStructs.hpp"
#include "QuantumNodes.hpp"

namespace DecompGraph::Graph {
// Define the graph type using Boost Graph Library's adjacency_list,
// with custom vertex and edge properties.
using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, GraphVertex,
                                    GraphWeightedEdge>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Edge = boost::graph_traits<Graph>::edge_descriptor;

struct PartialRuleState {
    size_t solvedInputs = 0;
    size_t totalInputs = 0;
    bool feasible = true;
    GraphResource resource;
};

struct SearchConfig {
    std::unordered_map<Core::OperatorNode, double, Core::OperatorNodeHash> opWeights;
};

struct SearchResult {
    std::unordered_map<Vertex, GraphResource> bestOpSolution;
    std::unordered_map<Vertex, Vertex> bestRuleSolution;
    std::unordered_map<Vertex, PartialRuleState> partialRule;
};

class IDecompositionSearch {
  public:
    virtual ~IDecompositionSearch() = default;
    virtual SearchResult solve(const Graph &graph, Vertex start,
                               const SearchConfig &config) const = 0;
};

} // namespace DecompGraph::Graph
