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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "Parity.hpp"

class BinaryMatrix {

public:
    // Constructors
    BinaryMatrix() = default;
    explicit BinaryMatrix(size_t numRows)
    {
        rows.reserve(numRows);
    }

    // Static Factories
    static BinaryMatrix Identity(size_t numRows, std::optional<size_t> numBlocks=std::nullopt);

    // Operators
    bool operator==(const BinaryMatrix &rhs) const;

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const BinaryMatrix &mat);
    
    template <typename ColOrderRange>
    [[nodiscard]] std::string toStringWithOrder(ColOrderRange order) const;

    // Getters
    [[nodiscard]] const Parity &getRowAt(size_t row) const;
    [[nodiscard]] Parity &getRowMutableAt(size_t row) const;
    [[nodiscard]] const std::vector<Parity>& getRows() const;
    [[nodiscard]] std::vector<Parity>& getRowsMutable();

    // Stats
    size_t getNumRows() const;

    // Setters
    void setRow(size_t row, const Parity &parity);
    void setRowToBasis(size_t row, BitLocation oneLoc);

    // Checks & Inspections
    bool isEmpty();

    // Methods
    Parity &allocRow();
    void extendRowsFor(IdxView newVars, size_t maxBlock);
    void reserveRowsFor(size_t addNumRows);
    void resetRow(size_t row, std::optional<size_t> newBlockNum=1);
    void flipBitAtRowAtLoc(size_t row, BitLocation loc);
    void swapRows(size_t row1, size_t row2);
    void addRowToRow(size_t sourceRow, size_t targetRow);
    void addParityToRow(size_t row, const Parity &parity);
    void dropTopRows(size_t numRows);
    void keepTopRows(size_t numRows);
    [[nodiscard]] size_t firstTrivialRow() const;
    template <typename ColOrderRange>
    [[nodiscard]] size_t firstTrivialInRangeRow(ColOrderRange rng) const;
    template <typename ColOrderRange>
    void normalize(ColOrderRange colOrd);
    template <typename ColOrderRange>
    void semiNormalize(ColOrderRange colOrd);
    template <typename ColOrderRange>
    void toREF(ColOrderRange colOrd);   // normal order shouldn't need input
    
private:
    std::vector<Parity> rows; // n x (m + 1) binary matrix

    // Helper Methods
    bool establishPivotAt(size_t pvtRow, BitLocation pvtCol);
    void forwardElimWrtPivot(size_t pvtRow, BitLocation pvtCol);
    void backwardElimWrtPivot(size_t pvtRow, BitLocation pvtCol);
    void rowReduceWrtPivot(size_t row, size_t pvtRow, BitLocation pvtCol);
    void dropTrivialRows();
    template <typename ColOrderRange>
    void toRREF(ColOrderRange colOrd);
    template <bool IsRREF, typename ColOrderRange>
    void computeEchelonForm(ColOrderRange colOrd);
    template <typename Predicate>
    size_t findFirstRowIf(Predicate cond) const;
};

template <typename ColOrderRange>
std::string BinaryMatrix::toStringWithOrder(ColOrderRange order) const
{
    std::string res = "";
    for (size_t i = 0; i < rows.size(); ++i) {
        res += rows[i].toStringWithOrder(order) + '\n';
    }
    return res;
}

template <bool IsRREF, typename ColOrderRange>
void BinaryMatrix::computeEchelonForm(ColOrderRange colOrd)
{
    const size_t numRows = getNumRows();
    size_t i = 0;

    for (BitLocation col : colOrd) {
        if (i >= numRows) {
            return;
        }
        if (establishPivotAt(i, col)) {
            forwardElimWrtPivot(i, col);
            if constexpr (IsRREF) {
                backwardElimWrtPivot(i, col);
            }
            ++i;
        }
    }
}

template <typename ColOrderRange>
size_t BinaryMatrix::firstTrivialInRangeRow(ColOrderRange rng) const
{
    return findFirstRowIf([rng](const Parity& row) {
        return row.isTrivialInRange(rng); 
    });
}

template <typename Predicate>
size_t BinaryMatrix::findFirstRowIf(Predicate cond) const  
{
    const size_t numRows = getNumRows();
    for (size_t i = 0; i < numRows; ++i) {
        if (cond(rows[i])) {
            return i;
        }
    }
    return numRows;
}

inline bool BinaryMatrix::operator==(const BinaryMatrix &rhs) const { return rows == rhs.rows; }

// Getters
inline size_t BinaryMatrix::getNumRows() const { return rows.size(); }

inline const Parity &BinaryMatrix::getRowAt(size_t row) const
{
    assert(row >= 0 && row < getNumRows());
    return rows[row];
}

inline Parity &BinaryMatrix::getRowMutableAt(size_t row) const
{
    return const_cast<Parity &>(static_cast<const BinaryMatrix &>(*this).getRowAt(row));
}

inline const std::vector<Parity>& BinaryMatrix::getRows() const { return rows; }

inline std::vector<Parity>& BinaryMatrix::getRowsMutable() { return rows; }

// Setters:
inline void BinaryMatrix::setRow(size_t row, const Parity &parity) { getRowMutableAt(row) = parity; }

inline void BinaryMatrix::setRowToBasis(size_t row, BitLocation oneLoc)
{
    getRowMutableAt(row).mkBasis(oneLoc, oneLoc.block);
}

inline void BinaryMatrix::resetRow(size_t row, std::optional<size_t> newNumBlocks) { getRowMutableAt(row).reset(newNumBlocks); }

inline void BinaryMatrix::flipBitAtRowAtLoc(size_t row, BitLocation loc) { getRowMutableAt(row).flipBitAtLoc(loc); }

// Checks
inline bool BinaryMatrix::isEmpty() { return rows.empty(); }

// Methods
inline Parity &BinaryMatrix::allocRow()
{
    rows.emplace_back();
    return rows.back();
}

inline void BinaryMatrix::reserveRowsFor(size_t addNumRows) { rows.reserve(rows.size() + addNumRows); }

inline void BinaryMatrix::swapRows(size_t row1, size_t row2)
{
    std::swap(getRowMutableAt(row1), getRowMutableAt(row2));
}

inline void BinaryMatrix::addParityToRow(size_t row, const Parity &parity)
{
    getRowMutableAt(row) += parity;
}

inline void BinaryMatrix::addRowToRow(size_t sourceRow, size_t targetRow)
{ // E_i,j
    getRowMutableAt(targetRow) += getRowAt(sourceRow);
}

template <typename ColOrderRange>
inline void BinaryMatrix::semiNormalize(ColOrderRange colOrd)
{
    toREF(colOrd);
    dropTrivialRows();
}

template <typename ColOrderRange>
inline void BinaryMatrix::normalize(ColOrderRange colOrd)
{
    toRREF(colOrd);
    dropTrivialRows();
}

template <typename ColOrderRange>
inline void BinaryMatrix::toREF(ColOrderRange colOrd) { computeEchelonForm<false>(colOrd); }

template <typename ColOrderRange>
inline void BinaryMatrix::toRREF(ColOrderRange colOrd) { computeEchelonForm<true>(colOrd); }

inline void BinaryMatrix::dropTopRows(size_t numRows) { rows.erase(rows.begin(), rows.begin() + numRows); } // test end

inline void BinaryMatrix::keepTopRows(size_t numRows) { rows.resize(numRows); }

inline void BinaryMatrix::dropTrivialRows() { keepTopRows(firstTrivialRow()); }
