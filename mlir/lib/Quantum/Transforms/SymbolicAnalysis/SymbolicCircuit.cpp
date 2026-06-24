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

#include "SymbolicCircuit.h"

#include <cassert>

/*
    Operators:
*/
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SymbolicCircuit &circ)
{
    os << "Phase polynomial:\n" << circ.phasePoly;
    os << "Affine transformation:\n" << circ.stateTrans;
    // os << "Phase polynomial:\n" << circ.phasePoly.algebraicView(circ.qubitNum);
    // os << "State transformation:\n" << circ.stateTrans.algebraicView(circ.qubitNum);
    return os;
}

/*
    Dimension Handling:
*/
bool SymbolicCircuit::areIndsInBound(llvm::ArrayRef<size_t> indices)
{
    size_t maxIndex = 0;
    for (size_t idx : indices) {
        if (idx > maxIndex) {
            maxIndex = idx;
        }
    }
    return (maxIndex < getQubitNum());
}

void SymbolicCircuit::extendQubitsBy(size_t addQubitNum)
{
    stateTrans.extendQubitsTo(stateTrans.getQubitNum() + addQubitNum);
}

/*
    Gate Applications:
*/
void SymbolicCircuit::initQubit(size_t qubitIndex, bool basisState)
{
    assert(qubitIndex < getQubitNum()); // or +-1?
    stateTrans.initQubit(qubitIndex, basisState);
}

void SymbolicCircuit::applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> qubitIndices,
                                std::optional<GateID> gateId)
{
    assert(areIndsInBound(qubitIndices));
    assert(arity(gate) == DYNAMIC_ARITY || qubitIndices.size() == arity(gate));
    assert(!isPhaseGate(gate) || gateId.has_value());

    switch (gate) {
    case Gate::I:
        break;
    case Gate::H:
        stateTrans.applyGateH(qubitIndices[0]);
        break;
    case Gate::X:
        stateTrans.applyGateX(qubitIndices[0]);
        break;
    case Gate::Y:
        if (isAdjoint)
            applyGateY_dag(qubitIndices[0], gateId.value());
        else
            applyGateY(qubitIndices[0], gateId.value());
        break;
    case Gate::Z:
    case Gate::S:
    case Gate::T:
    case Gate::RZ:
        applyGateRZ(qubitIndices[0], gateId.value());
        break;
    case Gate::CNOT:
        stateTrans.applyGateCNOT(qubitIndices[0], qubitIndices[1]);
        break;
    case Gate::SWAP:
        stateTrans.applyGateSWAP(qubitIndices[0], qubitIndices[1]);
        break;
    case Gate::U:
        stateTrans.applyGateU(qubitIndices);
        break;
    case Gate::GP:
        break; // figure out later.
    }
}

void SymbolicCircuit::applyGateRZ(size_t qubitIndex, GateID gateId)
{
    Parity &parity = stateTrans.getExprMutable(qubitIndex);

    bool affineVal = parity.getAffineValue();
    PhaseBucket contributor = PhaseBucket(gateId, affineVal);

    parity.clearAffineValue();
    phasePoly.insertContributor(parity, contributor);
    parity.assignAffineValue(affineVal);
}

void SymbolicCircuit::applyGateY(size_t qubitIndex, GateID gateId)
{
    stateTrans.applyGateX(qubitIndex);
    applyGateRZ(qubitIndex, gateId);
    // global phase of +i.
}

void SymbolicCircuit::applyGateY_dag(size_t qubitIndex, GateID gateId)
{
    applyGateRZ(qubitIndex, gateId);
    stateTrans.applyGateX(qubitIndex);
    // global phase of -i.
}
