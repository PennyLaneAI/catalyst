#pragma once

// #include <iosfwd>
#include <unordered_map>
#include <utility>
#include <string>
#include "Parity.h"
#include "Term.h"

#include "llvm/Support/raw_ostream.h"

struct PhasePolynomial {
    std::unordered_map<Parity, Term> poly;   // dense_map in mlir
    // bots are not seperated and duplicate!

    // Constructors
    PhasePolynomial() = default;
    PhasePolynomial(std::unordered_map<Parity, Term> poly) :
        poly(std::move(poly)) {}

    // Operators
    PhasePolynomial& operator+=(const PhasePolynomial& rhs);
    PhasePolynomial operator+(const PhasePolynomial& rhs) const;

    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const PhasePolynomial& pp);
    std::string algebraicView(size_t qubitNum) const;

    // Methods
    void insertTerm(const Parity& parity, const Term& term);
};