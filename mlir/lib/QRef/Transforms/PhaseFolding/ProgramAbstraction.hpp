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

#include <cassert>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "AffineTransform.hpp"
#include "Gate.hpp"
#include "PhaseAbstraction.hpp"

namespace catalyst::phase_folding {

struct RegionSummary;
class AffineRelation;

struct ProgramAbstraction {
    PhaseAbstraction phases;
    AffineTransform
        stateTransform; // row i (starting from 0) corresponds to qubit i, but col i doesn't!

    // Constructors
    ProgramAbstraction() = default;
    ProgramAbstraction(size_t numQubits)
        : phases(PhaseAbstraction()), stateTransform(AffineTransform(numQubits)) {}

    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const ProgramAbstraction &progAbs);

    // Getters
    [[nodiscard]] size_t numQubits() const;
    [[nodiscard]] TransformSchema getSchema() const;

    // Methods
    void extendQubitsBy(size_t addQubitNum);
    void prepareQubit(size_t wire, bool basisState);
    void applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> wires,
                   std::optional<GateID> gateId = std::nullopt);
    void applySummary(RegionSummary &&summary);

  private:
    bool areWiresInBound(llvm::ArrayRef<size_t> wires);
    void applyGateRZ(size_t wire, GateID gateId);
    void applyGateY(size_t wire, GateID gateId);
    void applyGateY_dag(size_t wire, GateID gateId);
    void normalizePhasesUnder(const AffineRelation &postcondRel);
};

inline size_t ProgramAbstraction::numQubits() const { return stateTransform.numQubits(); }

inline TransformSchema ProgramAbstraction::getSchema() const { return stateTransform.getSchema(); }

inline void ProgramAbstraction::extendQubitsBy(size_t addQubitNum) {
    stateTransform.extendQubitsBy(addQubitNum);
}

inline void ProgramAbstraction::prepareQubit(size_t wire, bool basisState) {
    assert(wire < numQubits()); // or +-1?
    stateTransform.prepareQubit(wire, basisState);
}

} // namespace catalyst::phase_folding
