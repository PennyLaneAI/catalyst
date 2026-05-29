// #include <ostream>
#include "Term.h"

/*.................
    Operators:
...................*/
// how about redundancy? I guess we can't have a single gate in 2 different terms, so probably doesn't matter.
Term& Term::operator+=(const Term& rhs) {
    gateRefPol_0.reserve(gateRefPol_0.size() + rhs.gateRefPol_0.size());
    gateRefPol_1.reserve(gateRefPol_1.size() + rhs.gateRefPol_1.size());

    gateRefPol_0.insert(gateRefPol_0.end(), rhs.gateRefPol_0.begin(), rhs.gateRefPol_0.end());
    gateRefPol_1.insert(gateRefPol_1.end(), rhs.gateRefPol_1.begin(), rhs.gateRefPol_1.end());

    return *this;
}

Term Term::operator+(const Term& rhs) const {
    Term res = *this;
    res += rhs;
    return res;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Term& term) {
    if (!term.gateRefPol_0.empty()) {
        os << "(";
        for (const GateID& gate : term.gateRefPol_0) {
            os << gate << ", ";
        }
        os << "0) ";
    }
    
    if (!term.gateRefPol_1.empty()) {
        os << "(";
        for (const GateID& gate : term.gateRefPol_1) {
            os << gate << ", ";
        }
        os << "1)";
    }
    return os;
}
