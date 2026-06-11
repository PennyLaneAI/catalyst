#pragma once

#include "PhasePolynomial.h"
#include "AffineTransform.h"
#include "Gate.h"
#include <utility>

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ArrayRef.h"


struct SymbolicCircuit { // indices are 1-based
    size_t qubitNum;
    size_t auxVarNum;
    // std::vector<bool> isAux;
    PhasePolynomial phasePoly;
    AffineTransform stateTrans; // row i corresponds to qubit i, but col i doesn't!
    
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

    // Dimension Handling
    void ensureCapacity(llvm::ArrayRef<size_t> qubitIndices);
    void extendQubitsBy(size_t newQubitNum);
    
    // Gate Applications
    void applyGate(Gate gate, bool isAdjoint, llvm::ArrayRef<size_t> qubitIndices, GateID gateId);
    void applyGateRZ(size_t qubitIndex, GateID gateId);
    void applyGateX(size_t qubitIndex);
    void applyGateY(size_t qubitIndex, GateID gateId);
    void applyGateY_dag(size_t qubitIndex, GateID gateId);
    void applyGateCNOT(size_t controlIndex, size_t targetIndex);
    void applyGateSWAP(size_t qubitIndex1, size_t qubitIndex2);
    void applyGateH(size_t qubitIndex);
    void applyGateU(llvm::ArrayRef<size_t> qubitIndices);

    void initQubit(size_t qubitIndex, bool basisState);
};
