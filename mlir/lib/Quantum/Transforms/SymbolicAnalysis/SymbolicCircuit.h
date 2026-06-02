#pragma once

// #include <iosfwd>
#include <utility>
#include "PhasePolynomial.h"
#include "AffineTransform.h"

#include "llvm/Support/raw_ostream.h"

// PathSum?
struct SymbolicCircuit {    // indices are 1-based
    size_t qubitNum;
    size_t auxVarNum;
    PhasePolynomial phasePoly;
    AffineTransform affTrans;

    enum Gate { I, H, X, Y, Y_dag, Z, S, T, RZ, CNOT, SWAP, U, GP };
    static constexpr size_t DYNAMIC_ARITY = 3;
    static constexpr size_t getGateArity(Gate gate) {
        switch (gate) {
        case U:
        case I:     return DYNAMIC_ARITY;
        case CNOT:
        case SWAP:  return 2;
        case GP:    return 0;
        default:    return 1;
        }
    }

    // Constructors
    SymbolicCircuit() = default;
    SymbolicCircuit(size_t qubitNum) :
        qubitNum(qubitNum), auxVarNum(0), 
        phasePoly(PhasePolynomial()), affTrans(AffineTransform::identity(qubitNum)) {}
    SymbolicCircuit(size_t qubitNum, size_t auxVarNum, PhasePolynomial phasePoly, AffineTransform affTrans) :
        qubitNum(qubitNum), auxVarNum(auxVarNum), 
        phasePoly(std::move(phasePoly)), affTrans(std::move(affTrans)) {}

    // Operators
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const SymbolicCircuit& circ);

    // Gate Applications
    void matchDim(std::vector<size_t>* qubitIndices);
    void applyGate(Gate gate, std::vector<size_t>* qubitIndices, GateID gateId);
    void applyGateRZ(size_t qubitIndex, GateID gateId);
    void applyGateX(size_t qubitIndex);
    void applyGateY(size_t qubitIndex, GateID gateId);
    void applyGateY_dag(size_t qubitIndex, GateID gateId);
    void applyGateCNOT(size_t controlIndex, size_t targetIndex);
    void applyGateSWAP(size_t qubitIndex1, size_t qubitIndex2);
    void applyGateH(size_t qubitIndex);
    void applyGateU(const std::vector<size_t>& qubitIndices);
};
// what are you going to do with Z and S?