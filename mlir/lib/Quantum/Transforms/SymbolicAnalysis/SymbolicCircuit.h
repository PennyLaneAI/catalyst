#pragma once

// #include <iosfwd>
#include <utility>
#include "PhasePolynomial.h"
#include "AffineTransform.h"
#include "Gate.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ArrayRef.h"


// PathSum?
struct SymbolicCircuit {    // indices are 1-based
    size_t qubitNum;
    size_t auxVarNum;           // is it really important to distinguish aux and qubit vars?
    // vactor<bool> isAux;
    PhasePolynomial phasePoly;
    AffineTransform stateTrans;
    
    // Constructors
    SymbolicCircuit() = default;
    SymbolicCircuit(size_t qubitNum) :
        qubitNum(qubitNum), auxVarNum(0), 
        phasePoly(PhasePolynomial()), stateTrans(AffineTransform::identity(qubitNum)) {}
    SymbolicCircuit(size_t qubitNum, size_t auxVarNum, PhasePolynomial phasePoly, AffineTransform stateTrans) :
        qubitNum(qubitNum), auxVarNum(auxVarNum), 
        phasePoly(std::move(phasePoly)), stateTrans(std::move(stateTrans)) {}

    // Operators
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const SymbolicCircuit& circ);

    // Gate Applications
    void ensureCapacity(llvm::ArrayRef<size_t> qubitIndices);
    void extendQubitsTo(size_t newQubitNum);
    void applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> qubitIndices, GateID gateId);
    void applyGateRZ(size_t qubitIndex, GateID gateId);
    void applyGateX(size_t qubitIndex);
    void applyGateY(size_t qubitIndex, GateID gateId);
    void applyGateY_dag(size_t qubitIndex, GateID gateId);
    void applyGateCNOT(size_t controlIndex, size_t targetIndex);
    void applyGateSWAP(size_t qubitIndex1, size_t qubitIndex2);
    void applyGateH(size_t qubitIndex);
    void applyGateU(llvm::ArrayRef<size_t> qubitIndices);
};

// aux vars should be after all qubit vars. if it's not possible, we should change the data structure of parity, or storing the types of vars somewhere.
