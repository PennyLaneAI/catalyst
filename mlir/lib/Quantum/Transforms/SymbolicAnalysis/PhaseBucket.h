#pragma once

// #include <iosfwd>
#include <vector>
#include <utility>
#include <string>

#include "llvm/Support/raw_ostream.h"

using GateID = int; // index of Operation pointers vector! (It's Loc in feynman and l in thesis)

struct PhaseBucket {
    std::vector<GateID> zeroAffineRZs;    // small_vector in mlir
    std::vector<GateID> oneAffineRZs;

    // Constructors
    PhaseBucket() = default;
    PhaseBucket(const GateID* gates_0, size_t n_0, const GateID* gates_1, size_t n_1) :
        zeroAffineRZs(gates_0, gates_0 + n_0), 
        oneAffineRZs(gates_1, gates_1 + n_1) {}
    PhaseBucket(std::vector<GateID> gates_0, std::vector<GateID> gates_1) :
        zeroAffineRZs(std::move(gates_0)), oneAffineRZs(std::move(gates_1)) {}
    PhaseBucket(GateID gate, bool pol) 
        { (pol ? oneAffineRZs : zeroAffineRZs).push_back(gate); }

    // Operators
    PhaseBucket& operator+=(const PhaseBucket& rhs);
    PhaseBucket operator+(const PhaseBucket& rhs) const;

    size_t gateCount() const;
    GateID mergeTarget() const;
    bool isMergeTargetAffineZero() const;

    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const PhaseBucket& PhaseBucket);
    std::string algebraicView() const;
};

inline size_t PhaseBucket::gateCount() const {
    return zeroAffineRZs.size() + oneAffineRZs.size();
}

inline bool PhaseBucket::isMergeTargetAffineZero() const {
    return !zeroAffineRZs.empty();
}