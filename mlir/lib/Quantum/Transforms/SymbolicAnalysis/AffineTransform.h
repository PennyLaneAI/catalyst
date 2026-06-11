#pragma once

#include <vector>
#include <string_view>
#include <utility>
#include <cassert>
#include "Parity.h"

#include "llvm/Support/raw_ostream.h"

class AffineTransform { // indices are 1-based
public:
    // Constructors
    AffineTransform() = default;
    AffineTransform(const Parity* rows, size_t n) :
        exprMatrix(rows, rows + n) {}

    // Static Factories
    static AffineTransform identity(size_t n);

    // Operators
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const AffineTransform& trans);
    std::string algebraicView(size_t qubitNum) const;

    // Getters
    [[nodiscard]] size_t getRowNum() const;
    [[nodiscard]] size_t getColNum(size_t row) const;
    [[nodiscard]] const Parity& getRow(size_t row) const;
    [[nodiscard]] Parity& getRowMutable(size_t row) const;
    
    // Setters
    void setRow(size_t row, const Parity& parity);
    void resetRow(size_t row);
    void flipAffineValueAtRow(size_t row);

    // Methods
    void extendTo(size_t newRowNum, size_t auxVarNum);
    void swapRows(size_t row1, size_t row2);
    void addRows(size_t sourceRow, size_t targetRow);
    void addRowWithParity(size_t row, const Parity& parity);

private:
    std::vector<Parity> exprMatrix;  // n x (n + m + 1) binary matrix

   // Helper Methods
    explicit AffineTransform(size_t n);   // Identity matrix by default
};

inline size_t AffineTransform::getRowNum() const {
    return exprMatrix.size();
}

inline size_t AffineTransform::getColNum(size_t row) const {
    return getRow(row).getVarNum();
}

inline const Parity& AffineTransform::getRow(size_t row) const {
    assert(row > 0 && row <= getRowNum());
    return exprMatrix[row - 1];
}

inline Parity& AffineTransform::getRowMutable(size_t row) const {
    return const_cast<Parity&>(static_cast<const AffineTransform&>(*this).getRow(row));
}
