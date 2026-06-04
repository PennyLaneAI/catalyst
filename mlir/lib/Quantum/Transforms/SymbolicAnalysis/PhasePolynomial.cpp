// #include <ostream>
#include "PhasePolynomial.h"

/*.................
    Operators:
...................*/
PhasePolynomial& PhasePolynomial::operator+=(const PhasePolynomial& rhs) {
    if (poly.empty()) {
        poly.reserve(rhs.poly.size());
    }
    for (const auto& [key, value] : rhs.poly) {
        insertTerm(key, value);
    }
    return *this;
}

PhasePolynomial PhasePolynomial::operator+(const PhasePolynomial& rhs) const {
    PhasePolynomial res = *this;
    res += rhs;
    return res;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const PhasePolynomial& pp) {
    for (const auto& [key, value] : pp.poly) {
        os << key << " -> " << value << "\n";
    }
    return os;
}

std::string PhasePolynomial::algebraicView(size_t qubitNum) const {
    std::string res = "";
    for (const auto& [key, value] : poly) {
        res += (key.algebraicView(qubitNum) + " -> " + value.algebraicView() + "\n");
    }
    return res;
}

/*.................
    Methods:
...................*/
void PhasePolynomial::insertTerm(const Parity& parity, const Term& term) {
    // auto [it, inserted] = poly.try_emplace(parity, term);
    // if (!inserted) {
    //     it->second += term;
    // }    // needs C++17

    auto it = poly.find(parity);
    if (it != poly.end()) {
        it->second += term;
    } else {
        poly[parity] = term;
    }
}
