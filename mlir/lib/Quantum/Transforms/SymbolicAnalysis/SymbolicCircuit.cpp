// #include <iostream>
#include "SymbolicCircuit.h"

/*.................
    Operators:
...................*/
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const SymbolicCircuit& circ) {
    // os << "Phase polynomial:\n" << circ.phasePoly;
    // os << "Affine transformation:\n" << circ.affTrans;
    os << "Phase polynomial:\n" << circ.phasePoly.algebraicView(circ.qubitNum);
    os << "Affine transformation:\n" << circ.affTrans.algebraicView(circ.qubitNum);
    return os;
}

/*.................
    Gate Applications:
...................*/
void SymbolicCircuit::applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> qubitIndices, GateID gateId) {
    llvm::SmallVector<size_t, 4> indices = convertIndicesBase(qubitIndices);
    ensureCapacity(indices);

    assert(SymbolicCircuit::getGateArity(gate) == SymbolicCircuit::DYNAMIC_ARITY || 
            indices.size() == SymbolicCircuit::getGateArity(gate));

    switch (gate) {
    case I:     break;
    case H:     applyGateH(indices[0]);    break;
    case X:     applyGateX(indices[0]);    break;
    case Y:     
        if (isAdjoint)  applyGateY_dag(indices[0], gateId);
        else            applyGateY(indices[0], gateId);
        break;
    case Z:
    case S:
    case T:
    case RZ:    applyGateRZ(indices[0], gateId);   break;
    case CNOT:  applyGateCNOT(indices[0], indices[1]);    break;
    case SWAP:  applyGateSWAP(indices[0], indices[1]);    break;
    case U:     applyGateU(indices);   break;
    case GP:    break;  // figure out later.
    }
}

llvm::SmallVector<size_t, 4> SymbolicCircuit::convertIndicesBase(llvm::ArrayRef<size_t> indices) {
    llvm::SmallVector<size_t, 4> oneBasedIndices;
    oneBasedIndices.reserve(indices.size());
    
    for (size_t idx : indices) {
        oneBasedIndices.push_back(idx + 1);
    }
    return oneBasedIndices;
}

void SymbolicCircuit::ensureCapacity(llvm::ArrayRef<size_t> indices) {
    size_t maxIndex = 0;
    for (size_t idx : indices) {
        if (idx > maxIndex) {
            maxIndex = idx;
        }
    }

    if (maxIndex > qubitNum) {
        affTrans.extendTo(maxIndex);
        qubitNum = maxIndex;
    }
}

// Do I need seperate Z, S, Sdag, T, Tdag functions?
void SymbolicCircuit::applyGateRZ(size_t qubitIndex, GateID gateId) {
    llvm::outs() << "R_z on q" << qubitIndex << " at l" << gateId << ":\n";

    const Parity& parity = affTrans.getRow(qubitIndex);
    Term term = Term(gateId, parity.getAffineValue());
    phasePoly.insertTerm(parity, term);
}

void SymbolicCircuit::applyGateX(size_t qubitIndex) {
    llvm::outs() << "X on q" << qubitIndex << ":\n";
    affTrans.flipAffineValueAtRow(qubitIndex);
}

void SymbolicCircuit::applyGateY(size_t qubitIndex, GateID gateId) {
    llvm::outs() << "Y on q" << qubitIndex << ":\n";
    applyGateX(qubitIndex);
    applyGateRZ(qubitIndex, gateId);
    // global phase of +i.
}

void SymbolicCircuit::applyGateY_dag(size_t qubitIndex, GateID gateId) {
    llvm::outs() << "Y† on q" << qubitIndex << ":\n";

    applyGateRZ(qubitIndex, gateId);
    applyGateX(qubitIndex);
    // global phase of -i.
}

void SymbolicCircuit::applyGateCNOT(size_t controlIndex, size_t targetIndex) {
    llvm::outs() << "CNOT on q" << controlIndex << " q" << targetIndex << ":\n";
    affTrans.addRows(controlIndex, targetIndex);
}

void SymbolicCircuit::applyGateSWAP(size_t qubitIndex1, size_t qubitIndex2) {
    llvm::outs() << "SWAP on q" << qubitIndex1 << " q" << qubitIndex2 << ":\n";
    affTrans.swapRows(qubitIndex1, qubitIndex2);
}

// uninterpreted gates.
void SymbolicCircuit::applyGateH(size_t qubitIndex) {
    llvm::outs() << "H on q" << qubitIndex << ":\n";
    
    auxVarNum++;
    affTrans.setRow(qubitIndex, Parity::eVec(qubitNum + auxVarNum, qubitNum + auxVarNum));  // are the indices correct?
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
        affTrans.setRow(qubitIndices[i], Parity::eVec(qubitNum + auxVarNum, qubitNum + auxVarNum - n + i)); // are the indices correct?
    }
}