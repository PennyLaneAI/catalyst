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
#include <utility>
#include <vector>

#include "llvm/Support/raw_ostream.h"

#include "Parity.h"

class BinaryMatrix {

public:
    // Constructors
    BinaryMatrix() = default;

    // Static Factories
    static BinaryMatrix identity(size_t n);

    // Operators
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const BinaryMatrix &mat);
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
    void extendRowsTo(size_t newRowNum, size_t auxVarNum);
    void swapRows(size_t row1, size_t row2);
    void addRowToRow(size_t sourceRow, size_t targetRow);
    void addParityToRow(size_t row, const Parity &parity);
    void toEchelonForm(std::vector<int>& colOrd);   // maybe get a col order as input
    void toReducedEchelonForm(std::vector<int>& colOrd);

private:
    std::vector<Parity> exprRows; // n x (m + 1) binary matrix

    // Helper Methods
    explicit BinaryMatrix(size_t n); // Identity matrix by default
    bool setPivotRow(size_t pivotRow, size_t pivotCol);
    void rowReduceWithPivot(size_t pivotRow, size_t pivotCol);
};

inline size_t BinaryMatrix::getRowNum() const { return exprRows.size(); }

inline size_t BinaryMatrix::getColNum(size_t row) const { return getRow(row).getVarNum(); }

inline const Parity &BinaryMatrix::getRow(size_t row) const
{
    assert(row >= 0 && row < getRowNum());
    return exprRows[row];
}

inline Parity &BinaryMatrix::getRowMutable(size_t row) const
{
    return const_cast<Parity &>(static_cast<const BinaryMatrix &>(*this).getRow(row));
}
