#pragma once

#include <iosfwd>
#include <utility>
#include "PhasePolynomial.h"
#include "AffineTransform.h"

// PathSum?
struct SymbolicCircuit {    // indices are 1-based
    int qubitNum;
    int auxVarNum;
    PhasePolynomial phasePoly;
    AffineTransform affTrans;

    // Constructors
    SymbolicCircuit() = default;
    SymbolicCircuit(int qubitNum) :
        qubitNum(qubitNum), auxVarNum(0), 
        phasePoly(PhasePolynomial()), affTrans(AffineTransform::identity(qubitNum)) {}
    SymbolicCircuit(int qubitNum, int auxVarNum, PhasePolynomial phasePoly, AffineTransform affTrans) :
        qubitNum(qubitNum), auxVarNum(auxVarNum), 
        phasePoly(std::move(phasePoly)), affTrans(std::move(affTrans)) {}

    // Operators
    friend std::ostream& operator<<(std::ostream& os, const SymbolicCircuit& circ);

    // Gate Applications
    void applyGateRZ(const int qubitIndex, const GateID gateId);
    void applyGateX(const int qubitIndex);
    void applyGateY(const int qubitIndex, const GateID gateId);
    void applyGateCNOT(const int controlIndex, const int targetIndex);
    void applyGateH(const int qubitIndex);
};
