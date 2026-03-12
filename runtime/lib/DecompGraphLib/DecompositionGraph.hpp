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
 * @file DecompositionGraph.hpp
 */

#pragma once

#include <memory>
#include <vector>

#include "GraphStructs.hpp"
#include "QuantumNodes.hpp"

namespace DecompGraph::Solver {

class DecompositionGraph {
  private:
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    using RuleId = std::size_t;

    DecompositionGraph(std::vector<Core::OperatorNode> operators, Core::WeightedGateset gateset,
                       std::vector<Core::RuleNode> rules);
    ~DecompositionGraph();

    // copy and move constructors and assignment operators
    DecompositionGraph(const DecompositionGraph &other);
    DecompositionGraph(DecompositionGraph &&other) noexcept;
    DecompositionGraph &operator=(const DecompositionGraph &other);
    DecompositionGraph &operator=(DecompositionGraph &&other) noexcept;

    [[nodiscard]] const std::vector<Core::OperatorNode> &getRoots() const noexcept;
    [[nodiscard]] const Core::WeightedGateset &getGateset() const noexcept;
    [[nodiscard]] const std::vector<Core::RuleNode> &getRules() const noexcept;
    std::size_t getNumRules() const;
    std::size_t getNumOperators() const;

    // helper methods
    // TODO(Ali): move them to private/protected after testing
    const Core::RuleNode &getRule(RuleId id) const;
    const std::vector<Core::RuleNode> &getAllRulesFor(const Core::OperatorNode &op) const;
    bool isTargetGate(const Core::OperatorNode &op) const;
    bool hasOperator(const Core::OperatorNode &op) const;
};

} // namespace DecompGraph::Solver