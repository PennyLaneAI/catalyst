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

using GateID = int; // index of Operation pointers vector! (It's Loc in feynman and l in thesis)

struct PhaseBucket {
    std::vector<GateID> zeroAffineRZs;
    std::vector<GateID> oneAffineRZs;

    // Constructors
    PhaseBucket() = default;
    PhaseBucket(const GateID *gates_0, size_t n_0, const GateID *gates_1, size_t n_1)
        : zeroAffineRZs(gates_0, gates_0 + n_0), oneAffineRZs(gates_1, gates_1 + n_1)
    {
    }
    PhaseBucket(std::vector<GateID> gates_0, std::vector<GateID> gates_1)
        : zeroAffineRZs(std::move(gates_0)), oneAffineRZs(std::move(gates_1))
    {
    }
    PhaseBucket(GateID gate, bool pol) { (pol ? oneAffineRZs : zeroAffineRZs).push_back(gate); }

    // Operators
    PhaseBucket &operator+=(const PhaseBucket &rhs);
    PhaseBucket operator+(const PhaseBucket &rhs) const;

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const PhaseBucket &PhaseBucket);
    std::string algebraicView() const;

    // Methods
    size_t gateCount() const;
    auto getAllGatesMutable();
    GateID getMergeTarget() const;
    bool isMergeTargetAffineZero() const;
};

inline size_t PhaseBucket::gateCount() const { return zeroAffineRZs.size() + oneAffineRZs.size(); }

inline auto PhaseBucket::getAllGatesMutable()
{
    return llvm::concat<GateID>(zeroAffineRZs, oneAffineRZs);
}

inline bool PhaseBucket::isMergeTargetAffineZero() const { return !zeroAffineRZs.empty(); }