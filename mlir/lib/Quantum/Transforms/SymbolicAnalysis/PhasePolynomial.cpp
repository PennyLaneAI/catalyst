// #include <ostream>
#include "PhasePolynomial.h"

/*.................
    Operators:
...................*/
PhasePolynomial& PhasePolynomial::operator+=(const PhasePolynomial& rhs) {
    if (terms.empty()) {
        terms.reserve(rhs.terms.size());
    }
    for (const auto& [parity, contributors] : rhs.terms) {
        insertContributor(parity, contributors);
    }
    return *this;
}

PhasePolynomial PhasePolynomial::operator+(const PhasePolynomial& rhs) const {
    PhasePolynomial res = *this;
    res += rhs;
    return res;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const PhasePolynomial& pp) {
    for (const auto& [parity, contributors] : pp.terms) {
        os << parity << " -> " << contributors << "\n";
    }
    return os;
}

std::string PhasePolynomial::algebraicView(size_t qubitNum) const {
    std::string res = "";
    for (const auto& [parity, contributors] : terms) {
        res += (parity.algebraicView(qubitNum) + " -> " + contributors.algebraicView() + "\n");
    }
    return res;
}

/*.................
    Methods:
...................*/
void PhasePolynomial::insertContributor(const Parity& parity, const Term& contributor) {
    // auto [it, inserted] = poly.try_emplace(parity, term);
    // if (!inserted) {
    //     it->second += term;
    // }    // needs C++17

    auto it = terms.find(parity);
    if (it != terms.end()) {
        it->second += contributor;
    } else {
        terms[parity] = contributor;
    }
}
