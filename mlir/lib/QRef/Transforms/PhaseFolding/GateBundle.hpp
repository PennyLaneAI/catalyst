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

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h" // llvm::concat<>
#include "llvm/Support/raw_ostream.h"

namespace catalyst::phase_folding {

using GateID = int; // index of Operation pointers vector! (It's Loc in feynman and l in thesis)

struct GateBundle {
    std::vector<GateID> zeroAffineGates;
    std::vector<GateID> oneAffineGates;
    // Constructors
    GateBundle() = default;
    GateBundle(GateID gate, bool pol) { (pol ? oneAffineGates : zeroAffineGates).push_back(gate); }

    // Operators
    GateBundle &operator+=(const GateBundle &rhs);
    GateBundle operator+(const GateBundle &rhs) const;

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const GateBundle &GateBundle);
    [[nodiscard]] std::string toString() const;

    // Methods
    [[nodiscard]] size_t gateCount() const;
    [[nodiscard]] auto getAllGates();
    [[nodiscard]] GateID getMergeTarget() const;
    [[nodiscard]] bool isMergeTargetAffineZero() const;
    void flipGatesAffineValues();
};

inline size_t GateBundle::gateCount() const {
    return zeroAffineGates.size() + oneAffineGates.size();
}

inline auto GateBundle::getAllGates() {
    return llvm::concat<GateID>(zeroAffineGates, oneAffineGates);
}

inline bool GateBundle::isMergeTargetAffineZero() const { return !zeroAffineGates.empty(); }

inline void GateBundle::flipGatesAffineValues() { std::swap(zeroAffineGates, oneAffineGates); }

} // namespace catalyst::phase_folding
