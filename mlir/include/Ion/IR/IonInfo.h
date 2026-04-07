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

#pragma once

#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "Ion/IR/IonOps.h"

namespace catalyst {
namespace ion {

/// Helper class to store and query ion information from an IonOp.
class IonInfo {
  public:
    struct TransitionInfo {
        std::string level0;
        std::string level1;
        double einstein_a;
        std::string multipole;
    };

  private:
    llvm::StringMap<double> levelEnergyMap;
    llvm::SmallVector<TransitionInfo> transitions;

  public:
    explicit IonInfo(ion::IonOp op)
    {
        auto levelAttrs = op.getLevels();
        auto transitionsAttr = op.getTransitions();

        // Map from Level label to Energy value
        for (auto levelAttr : levelAttrs) {
            auto level = mlir::cast<LevelAttr>(levelAttr);
            std::string label = level.getLabel().getValue().str();
            double energy = level.getEnergy().getValueAsDouble();
            levelEnergyMap[label] = energy;
        }

        // Store transition information
        for (auto transitionAttr : transitionsAttr) {
            auto transition = mlir::cast<TransitionAttr>(transitionAttr);
            TransitionInfo info;
            info.level0 = transition.getLevel_0().getValue().str();
            info.level1 = transition.getLevel_1().getValue().str();
            info.einstein_a = transition.getEinsteinA().getValueAsDouble();
            info.multipole = transition.getMultipole().getValue().str();
            transitions.push_back(info);
        }
    }

    /// Get energy of a level by label
    std::optional<double> getLevelEnergy(llvm::StringRef label) const
    {
        auto it = levelEnergyMap.find(label.str());
        if (it != levelEnergyMap.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /// Get level energy of a transition by index
    template <int IndexT>
    std::optional<double> getTransitionLevelEnergy(size_t transitionIndex) const
    {
        static_assert(IndexT == 0 || IndexT == 1, "IndexT must be 0 or 1");

        if (transitionIndex >= transitions.size()) {
            return std::nullopt;
        }

        const auto &transition = transitions[transitionIndex];
        if constexpr (IndexT == 0) {
            return getLevelEnergy(transition.level0);
        }
        else {
            return getLevelEnergy(transition.level1);
        }
    }

    /// Get energy difference of a transition (level1 energy - level0 energy)
    std::optional<double> getTransitionEnergyDiff(size_t index) const
    {
        if (index >= transitions.size()) {
            return std::nullopt;
        }

        auto energy0 = getTransitionLevelEnergy<0>(index);
        auto energy1 = getTransitionLevelEnergy<1>(index);

        if (energy0.has_value() && energy1.has_value()) {
            return energy1.value() - energy0.value();
        }

        return std::nullopt;
    }

    /// Get number of transitions
    size_t getNumTransitions() const { return transitions.size(); }

    /// Get transition info by index
    std::optional<TransitionInfo> getTransition(size_t index) const
    {
        if (index < transitions.size()) {
            return transitions[index];
        }
        return std::nullopt;
    }
};

} // namespace ion
} // namespace catalyst
