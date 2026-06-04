#pragma once

// #include <iosfwd>
#include <vector>
#include <utility>
#include <string>

#include "llvm/Support/raw_ostream.h"

using GateID = int; // pointer to gates in catalyst! UID in Catalyst or needs a preprocessing for putting this location enumerators. (It's Loc in feynman and l in thesis)
// TODO: rename
struct Term {
    std::vector<GateID> gateRefPol_0;    // small_vector in mlir
    std::vector<GateID> gateRefPol_1;

    // Constructors
    Term() = default;
    Term(const GateID* gates_0, size_t n_0, const GateID* gates_1, size_t n_1) :
        gateRefPol_0(gates_0, gates_0 + n_0), 
        gateRefPol_1(gates_1, gates_1 + n_1) {}
    Term(std::vector<GateID> v0, std::vector<GateID> v1) :
        gateRefPol_0(std::move(v0)), gateRefPol_1(std::move(v1)) {}
    Term(GateID gate, bool pol) 
        { (pol ? gateRefPol_1 : gateRefPol_0).push_back(gate); }

    // Operators
    Term& operator+=(const Term& rhs);
    Term operator+(const Term& rhs) const;

    size_t gateNum() const;
    GateID getHead() const;
    bool isHead0() const;

    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Term& term);
    std::string algebraicView() const;
};

inline size_t Term::gateNum() const {
    return gateRefPol_0.size() + gateRefPol_1.size();
}

inline bool Term::isHead0() const {
    return !gateRefPol_0.empty();
}