// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cassert>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/Support/raw_ostream.h"

#include "Parity.h"

class AffineTransform {
  public:
    // Constructors
    AffineTransform() = default;
    AffineTransform(const Parity *rows, size_t n) : exprMatrix(rows, rows + n) {}

    // Static Factories
    static AffineTransform identity(size_t n);

    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const AffineTransform &trans);
    std::string algebraicView(size_t qubitNum) const;

    // Getters
    [[nodiscard]] size_t getRowNum() const;
    [[nodiscard]] size_t getColNum(size_t row) const;
    [[nodiscard]] const Parity &getRow(size_t row) const;
    [[nodiscard]] Parity &getRowMutable(size_t row) const;

    // Setters
    void setRow(size_t row, const Parity &parity);
    void resetRow(size_t row);
    void flipAffineValueAtRow(size_t row);

    // Methods
    void extendTo(size_t newRowNum, size_t auxVarNum);
    void swapRows(size_t row1, size_t row2);
    void addRows(size_t sourceRow, size_t targetRow);
    void addRowWithParity(size_t row, const Parity &parity);

  private:
    std::vector<Parity> exprMatrix; // n x (n + m + 1) binary matrix

    // Helper Methods
    explicit AffineTransform(size_t n); // Identity matrix by default
};

inline size_t AffineTransform::getRowNum() const { return exprMatrix.size(); }

inline size_t AffineTransform::getColNum(size_t row) const { return getRow(row).getVarNum(); }

inline const Parity &AffineTransform::getRow(size_t row) const
{
    assert(row >= 0 && row < getRowNum());
    return exprMatrix[row];
}

inline Parity &AffineTransform::getRowMutable(size_t row) const
{
    return const_cast<Parity &>(static_cast<const AffineTransform &>(*this).getRow(row));
}
