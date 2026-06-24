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

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "Gate.h"
#include "AffineTransform.h"
#include "PhasePolynomial.h"

struct SymbolicCircuit {
    // std::vector<bool> isAux;
    PhasePolynomial phasePoly;
    AffineTransform stateTrans; // row i corresponds to qubit i, but col i doesn't!

    // Constructors
    SymbolicCircuit() = default;
    SymbolicCircuit(size_t qubitNum)
        : phasePoly(PhasePolynomial()), stateTrans(AffineTransform(qubitNum))
    {
    }
    SymbolicCircuit(PhasePolynomial phasePoly, AffineTransform stateTrans)
        : phasePoly(std::move(phasePoly)), stateTrans(std::move(stateTrans))
    {
    }

    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SymbolicCircuit &circ);

    // Getters
    [[nodiscard]] size_t getQubitNum() const;

    // Dimension Handling
    bool areIndsInBound(llvm::ArrayRef<size_t> qubitIndices);
    void extendQubitsBy(size_t newQubitNum);

    // Gate Applications
    void applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> qubitIndices,
                   std::optional<GateID> gateId = std::nullopt);
    void applyGateRZ(size_t qubitIndex, GateID gateId);
    void applyGateY(size_t qubitIndex, GateID gateId);
    void applyGateY_dag(size_t qubitIndex, GateID gateId);

    void initQubit(size_t qubitIndex, bool basisState);
};

inline size_t SymbolicCircuit::getQubitNum() const
{
  return stateTrans.getQubitNum();
}
