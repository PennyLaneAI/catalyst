// #include <iostream>
#include "SymbolicCircuit.h"

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const SymbolicCircuit& circ) {
    os << "Phase polynomial:\n" << circ.phasePoly;
    os << "Affine transformation:\n" << circ.affTrans;
    return os;
}

// Do I need seperate Z, S, Sdag, T, Tdag functions?
void SymbolicCircuit::applyGateRZ(const int qubitIndex, const GateID gateId) {
    llvm::errs() << "R_z on q" << qubitIndex << " at l" << gateId << ":\n";

    const Parity& parity = affTrans.getRow(qubitIndex);
    Term term = Term(gateId, parity.getAffineValue());
    phasePoly.insertTerm(parity, term);
}

void SymbolicCircuit::applyGateX(const int qubitIndex) {
    llvm::errs() << "X on q" << qubitIndex << ":\n";
    affTrans.flipAffineValueAtRow(qubitIndex);
}

void SymbolicCircuit::applyGateY(const int qubitIndex, const GateID gateId) {
    llvm::errs() << "Y on q" << qubitIndex << ":\n";
    applyGateX(qubitIndex);
    applyGateRZ(qubitIndex, gateId);
    // global phase of +i.
}

void SymbolicCircuit::applyGateCNOT(const int controlIndex, const int targetIndex) {
    llvm::errs() << "CNOT on q" << controlIndex << " q" << targetIndex << ":\n";
    affTrans.addRows(controlIndex, targetIndex);
}

// uninterpreted gates.
void SymbolicCircuit::applyGateH(const int qubitIndex) {
    llvm::errs() << "H on q" << qubitIndex << ":\n";
    
    auxVarNum++;
    affTrans.setRow(qubitIndex, Parity::eVec(qubitNum + auxVarNum, qubitNum + auxVarNum));  // are the indices correct?
}   // make sure nothing leads to segment fault and index out of bounds. also update operations to support different dimendions accordingly.
