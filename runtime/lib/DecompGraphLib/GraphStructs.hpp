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
 * @file GraphStructs.hpp
 * @brief This file defines the core data structures for representing the decomposition graph,
 * including vertices, edges, and resources.
 */

#pragma once

#include <limits>
#include <unordered_map>
#include <variant>

#include "QuantumNodes.hpp"

namespace DecompGraph::Graph {

/**
 * @brief This represents the start vertex in the decomposition graph.
 */
struct StartVertex {};

/**
 * @brief This represents an operator vertex in the decomposition graph.
 */
struct OperatorVertex {
    Core::OperatorNode op;
};

/**
 * @brief This represents a decomposition rule vertex in the decomposition graph.
 */
struct RuleVertex {
    Core::RuleNode rule;
};

/**
 * @brief This represents a vertex in the decomposition graph,
 * which can be a start vertex, an operator vertex, or a rule vertex.
 *
 * The GraphVertex struct uses std::variant to allow for different types of vertices
 * while maintaining type safety. The VertexType enum is used to identify the type of vertex,
 * and the payload holds the corresponding data for that vertex type.
 */
enum class VertexType { Start = 0, Operator = 1, Rule = 2 };

/**
 * @brief This represents the type of edge in the decomposition graph.
 *
 * The EdgeType enum defines the different types of edges that can exist in the graph:
 * - StartToBasisOp: An edge from the start vertex to a basis operator vertex.
 * - OperatorToRule: Indicating that the operator appears inside the decomposition rule.
 * - RuleToOperator: Indicating that the rule decomposes that operator.
 */
enum class EdgeType { StartToBasisOp = 0, OperatorToRule = 1, RuleToOperator = 2 };

/**
 * @brief This represents a vertex in the decomposition graph,
 * which can be a start vertex, an operator vertex, or a rule vertex.
 */
struct GraphVertex {
    VertexType type;
    std::variant<StartVertex, OperatorVertex, RuleVertex> payload;
};

/**
 * @brief This represents a weighted edge in the decomposition graph,
 * which includes the edge type and the weight of the edge.
 */
struct GraphWeightedEdge {
    EdgeType type;
    double weight = 0.0; // default 0.0 for StartToBasisOp & RuleToOperator
};

/**
 * @brief This represents the resources associated with a vertex in the decomposition graph.
 *
 * The GraphResource struct captures the resource usage of a vertex, which is crucial for the
 * graph decomposition solver to optimize the selection of decomposition rules based on resource
 * constraints. The opCounts map tracks the count of each operator node, and the weightedCost
 * provides a cost metric for the vertex, which can be used to guide the solver towards more
 * efficient decompositions.
 */
struct GraphResource {
    std::unordered_map<Core::OperatorNode, size_t, Core::OperatorNodeHash> opCounts;
    double weightedCost =
        std::numeric_limits<double>::infinity(); // default to infinity for non-rule vertices
};

} // namespace DecompGraph::Graph
