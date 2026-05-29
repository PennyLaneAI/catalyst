#include <ostream>
#include <vector>
#include "AffineTransform.h"

/*.................
    Constructors:
...................*/
AffineTransform::AffineTransform(size_t n) {
    exprMatrix.reserve(n);
    for (size_t i = 0; i < n; i++) {
        exprMatrix.push_back(Parity::eVec(n, i + 1));
    }
}

AffineTransform AffineTransform::identity(size_t n) {
    return AffineTransform(n);
}

/*.................
    Operators:
...................*/
std::ostream& operator<<(std::ostream& os, const AffineTransform& trans) {
    for (auto it = trans.exprMatrix.begin(); it != trans.exprMatrix.end(); it++) {
        os << *it << '\n';
    }
    return os;
}

/*.................
    Getters and Setters:
...................*/
void AffineTransform::setRow(size_t row, const Parity& parity) {
    getRowRef(row) = parity;
}

void AffineTransform::flipAffineValueAtRow(size_t row) {
    getRowRef(row).flipAffineValue();
}

/*.................
    Methods:
...................*/
void AffineTransform::addRowWithParity(size_t row, const Parity& parity) {
    getRowRef(row) += parity;
}

void AffineTransform::addRows(size_t sourceRow, size_t targetRow) {    // E_i,j
    getRowRef(targetRow) += getRow(sourceRow);
}
