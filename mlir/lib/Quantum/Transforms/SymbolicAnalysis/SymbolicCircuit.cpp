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
void SymbolicCircuit::ensureCapacity(llvm::ArrayRef<size_t> indices)
{
    size_t maxIndex = 0;
    for (size_t idx : indices) {
        if (idx > maxIndex) {
            maxIndex = idx;
        }
    }
    assert(maxIndex < qubitNum);
}

void SymbolicCircuit::extendQubitsBy(size_t addQubitNum)
{
    size_t newQubitNum = qubitNum + addQubitNum;
    stateTrans.extendTo(newQubitNum, auxVarNum);
    qubitNum = newQubitNum;
}

/*
    Gate Applications:
*/
void SymbolicCircuit::applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> qubitIndices,
                                std::optional<GateID> gateId)
{
    ensureCapacity(qubitIndices);

    assert(arity(gate) == DYNAMIC_ARITY || qubitIndices.size() == arity(gate));
    assert(!isPhaseGate(gate) || gateId.has_value());

    switch (gate) {
    case Gate::I:
        break;
    case Gate::H:
        applyGateH(qubitIndices[0]);
        break;
    case Gate::X:
        applyGateX(qubitIndices[0]);
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
        applyGateCNOT(qubitIndices[0], qubitIndices[1]);
        break;
    case Gate::SWAP:
        applyGateSWAP(qubitIndices[0], qubitIndices[1]);
        break;
    case Gate::U:
        applyGateU(qubitIndices);
        break;
    case Gate::GP:
        break; // figure out later.
    }
}

void SymbolicCircuit::applyGateRZ(size_t qubitIndex, GateID gateId)
{
    Parity &parity = stateTrans.getRowMutable(qubitIndex);

    bool affineVal = parity.getAffineValue();
    PhaseBucket contributor = PhaseBucket(gateId, affineVal);

    parity.clearAffineValue();
    phasePoly.insertContributor(parity, contributor);
    parity.assignAffineValue(affineVal);
}

void SymbolicCircuit::applyGateX(size_t qubitIndex) { stateTrans.flipAffineValueAtRow(qubitIndex); }

void SymbolicCircuit::applyGateY(size_t qubitIndex, GateID gateId)
{
    applyGateX(qubitIndex);
    applyGateRZ(qubitIndex, gateId);
    // global phase of +i.
}

void SymbolicCircuit::applyGateY_dag(size_t qubitIndex, GateID gateId)
{
    applyGateRZ(qubitIndex, gateId);
    applyGateX(qubitIndex);
    // global phase of -i.
}

void SymbolicCircuit::applyGateCNOT(size_t controlIndex, size_t targetIndex)
{
    stateTrans.addRows(controlIndex, targetIndex);
}

void SymbolicCircuit::applyGateSWAP(size_t qubitIndex1, size_t qubitIndex2)
{
    stateTrans.swapRows(qubitIndex1, qubitIndex2);
}

void SymbolicCircuit::applyGateH(size_t qubitIndex)
{
    auxVarNum++;
    stateTrans.setRow(qubitIndex, Parity::eVec(qubitNum + auxVarNum, qubitNum + auxVarNum));
}

// uninterpreted gates.
void SymbolicCircuit::applyGateU(llvm::ArrayRef<size_t> qubitIndices)
{
    llvm::outs() << "U on qubits ";
    for (size_t index : qubitIndices) {
        llvm::outs() << index << ", ";
    }
    llvm::outs() << ":\n";

    size_t n = qubitIndices.size();
    auxVarNum += n;
    for (size_t i = 0; i < n; i++) {
        stateTrans.setRow(qubitIndices[i],
                          Parity::eVec(qubitNum + auxVarNum, qubitNum + auxVarNum - n + i + 1));
    }
} // make sure nothing leads to segment fault and index out of bounds.

void SymbolicCircuit::initQubit(size_t qubitIndex, bool basisState)
{
    stateTrans.resetRow(qubitIndex);
    if (basisState) {
        stateTrans.flipAffineValueAtRow(qubitIndex);
    }
}
