#pragma once

#include "Parity.h"
#include "PhaseBucket.h"
#include <utility>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

struct PhasePolynomial {
    llvm::DenseMap<Parity, PhaseBucket> terms;
    // bots are not seperated and duplicate!

    // Constructors
    PhasePolynomial() = default;
    PhasePolynomial(llvm::DenseMap<Parity, PhaseBucket> terms) :
        terms(std::move(terms)) {}

    // Operators
    PhasePolynomial& operator+=(const PhasePolynomial& rhs);
    PhasePolynomial operator+(const PhasePolynomial& rhs) const;

    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const PhasePolynomial& pp);
    std::string algebraicView(size_t qubitNum) const;

    // Methods
    void insertContributor(const Parity& parity, const PhaseBucket& contributor);
};