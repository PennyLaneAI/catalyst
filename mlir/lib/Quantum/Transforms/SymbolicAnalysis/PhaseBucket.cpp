#include "PhaseBucket.h"

/*
    Operators:
*/
// how about redundancy? I guess we can't have a single gate in 2 different buckets, so probably doesn't matter.
PhaseBucket& PhaseBucket::operator+=(const PhaseBucket& rhs) {
    zeroAffineRZs.reserve(zeroAffineRZs.size() + rhs.zeroAffineRZs.size());
    oneAffineRZs.reserve(oneAffineRZs.size() + rhs.oneAffineRZs.size());

    zeroAffineRZs.insert(zeroAffineRZs.end(), rhs.zeroAffineRZs.begin(), rhs.zeroAffineRZs.end());
    oneAffineRZs.insert(oneAffineRZs.end(), rhs.oneAffineRZs.begin(), rhs.oneAffineRZs.end());

    return *this;
}

PhaseBucket PhaseBucket::operator+(const PhaseBucket& rhs) const {
    PhaseBucket res = *this;
    res += rhs;
    return res;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const std::vector<GateID>& gates) {
    os << ": (";
    for (size_t i = 0; i < gates.size(); i++) {
        if (i > 0) {
            os << ", ";
        }
        os << gates[i];
    }
    os << ")";
    return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const PhaseBucket& bucket) {
    os << '[';
    os << "0" << bucket.zeroAffineRZs;
    os << " __ ";
    os << "1" << bucket.oneAffineRZs;
    os << ']';
    return os;
}

std::string PhaseBucket::algebraicView() const {
    std::string res = "";

    if (!zeroAffineRZs.empty()) {
        res += "[0: (";

        for (size_t i = 0; i < zeroAffineRZs.size(); i++) {
            if (i > 0) {
                res += ", ";
            }
            res += (std::to_string(zeroAffineRZs[i] + 1));
        }
        res += ")";

        res += (oneAffineRZs.empty() ? "]" : ", ");
    }
    
    if (!oneAffineRZs.empty()) {

        res += (zeroAffineRZs.empty() ? "[" : "");

        res += "1: (";

        for (size_t i = 0; i < oneAffineRZs.size(); i++) {
            if (i > 0) {
                res += ", ";
            }
            res += (std::to_string(oneAffineRZs[i] + 1));
        }
        res += ")]";        
    }
    return res;
}

/*
    Methods:
*/
GateID PhaseBucket::getMergeTarget() const {
    if (!zeroAffineRZs.empty()) {
        return zeroAffineRZs[0];
    }
    if (!oneAffineRZs.empty()) {
        return oneAffineRZs[0];
    }
    return -1;
}
