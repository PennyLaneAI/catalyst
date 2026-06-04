#pragma once

// #include <iosfwd>
#include <utility>
#include "PhasePolynomial.h"
#include "AffineTransform.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ArrayRef.h"

// PathSum?
struct SymbolicCircuit {    // indices are 1-based
    size_t qubitNum;
    size_t auxVarNum;
    PhasePolynomial phasePoly;
    AffineTransform stateTrans;

    enum Gate { I, H, X, Y, Z, S, T, RZ, CNOT, SWAP, U, GP };
    static constexpr size_t DYNAMIC_ARITY = 3;
    static constexpr size_t arities[] = { DYNAMIC_ARITY, 1, 1, 1, 1, 1, 1, 1, 2, 2, DYNAMIC_ARITY, 0 };
    static constexpr size_t getGateArity(Gate gate) { return arities[gate]; }
    // static constexpr bool isRZ(Gate gate) { return ((gate == Z) || (gate == S) || (gate == T) || (gate == RZ)); }
    static constexpr bool isRZ(Gate gate) { return ((gate == T) || (gate == RZ)); }

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
// what are you going to do with Z and S?