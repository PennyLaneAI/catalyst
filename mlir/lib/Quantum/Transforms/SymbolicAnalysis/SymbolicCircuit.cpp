// #include <iostream>
#include "SymbolicCircuit.h"
#include <cassert>

/*.................
    Operators:
...................*/
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const SymbolicCircuit& circ) {
    // os << "Phase polynomial:\n" << circ.phasePoly;
    // os << "Affine transformation:\n" << circ.stateTrans;
    os << "Phase polynomial:\n" << circ.phasePoly.algebraicView(circ.qubitNum);
    os << "State transformation:\n" << circ.stateTrans.algebraicView(circ.qubitNum);
    return os;
}

/*.................
    Gate Applications:
...................*/
void SymbolicCircuit::applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> qubitIndices, GateID gateId) {
    ensureCapacity(qubitIndices);

    assert(arity(gate) == DYNAMIC_ARITY || 
            qubitIndices.size() == arity(gate));

    switch (gate) {
    case Gate::I:     break;
    case Gate::H:     applyGateH(qubitIndices[0]);    break;
    case Gate::X:     applyGateX(qubitIndices[0]);    break;
    case Gate::Y:     
        if (isAdjoint)  applyGateY_dag(qubitIndices[0], gateId);
        else            applyGateY(qubitIndices[0], gateId);
        break;
    case Gate::Z:
    case Gate::S:
    case Gate::T:
    case Gate::RZ:    applyGateRZ(qubitIndices[0], gateId);   break;
    case Gate::CNOT:  applyGateCNOT(qubitIndices[0], qubitIndices[1]);    break;
    case Gate::SWAP:  applyGateSWAP(qubitIndices[0], qubitIndices[1]);    break;
    case Gate::U:     applyGateU(qubitIndices);   break;
    case Gate::GP:    break;  // figure out later.
    }
}

void SymbolicCircuit::ensureCapacity(llvm::ArrayRef<size_t> indices) {
    size_t maxIndex = 0;
    for (size_t idx : indices) {
        if (idx > maxIndex) {
            maxIndex = idx;
        }
    }

    if (maxIndex > qubitNum) {
        extendQubitsTo(maxIndex);
        // stateTrans.extendTo(maxIndex);
        // qubitNum = maxIndex;
    }
}

void SymbolicCircuit::extendQubitsTo(size_t newQubitNum) {
    assert(newQubitNum > qubitNum);
    stateTrans.extendTo(newQubitNum);
    qubitNum = newQubitNum;
}

void SymbolicCircuit::applyGateRZ(size_t qubitIndex, GateID gateId) {
    Parity& parity = stateTrans.getRow(qubitIndex);

    bool affineVal = parity.getAffineValue();
    PhaseBucket contributor = PhaseBucket(gateId, affineVal);
    
    parity.clearAffineValue();
    phasePoly.insertContributor(parity, contributor);
    parity.setAffineValue(affineVal);
}

void SymbolicCircuit::applyGateX(size_t qubitIndex) {
    stateTrans.flipAffineValueAtRow(qubitIndex);
}

void SymbolicCircuit::applyGateY(size_t qubitIndex, GateID gateId) {
    applyGateX(qubitIndex);
    applyGateRZ(qubitIndex, gateId);
    // global phase of +i.
}

void SymbolicCircuit::applyGateY_dag(size_t qubitIndex, GateID gateId) {
    applyGateRZ(qubitIndex, gateId);
    applyGateX(qubitIndex);
    // global phase of -i.
}

void SymbolicCircuit::applyGateCNOT(size_t controlIndex, size_t targetIndex) {
    stateTrans.addRows(controlIndex, targetIndex);
}

void SymbolicCircuit::applyGateSWAP(size_t qubitIndex1, size_t qubitIndex2) {
    stateTrans.swapRows(qubitIndex1, qubitIndex2);
}

// uninterpreted gates.
void SymbolicCircuit::applyGateH(size_t qubitIndex) {   
    auxVarNum++;
    stateTrans.setRow(qubitIndex, Parity::eVec(qubitNum + auxVarNum, qubitNum + auxVarNum));  // are the indices correct?
}   // make sure nothing leads to segment fault and index out of bounds. also update operations to support different dimendions accordingly.

void SymbolicCircuit::applyGateU(llvm::ArrayRef<size_t> qubitIndices) {
    llvm::outs() << "U on qubits ";
    for (size_t index : qubitIndices) {
        llvm::outs() << index << ", ";
    }
    llvm::outs() << ":\n";
    size_t n = qubitIndices.size();
    auxVarNum += n;
    for (size_t i = 0; i < n; i++) {
        stateTrans.setRow(qubitIndices[i], Parity::eVec(qubitNum + auxVarNum, qubitNum + auxVarNum - n + i)); // are the indices correct?
    }
}
