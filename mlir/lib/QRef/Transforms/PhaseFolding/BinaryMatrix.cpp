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

#include "BinaryMatrix.hpp"

using namespace catalyst::phase_folding;

/*
    Static Factories:
*/
BinaryMatrix BinaryMatrix::Identity(size_t numRows, std::optional<size_t> numBlocks) {
    size_t blocks = numBlocks.value_or(BitLocation::requiredBlockNum(numRows));
    BinaryMatrix mat(numRows);
    BitLocation loc(0, 0);

    for (size_t i = 0; i < numRows; ++i) {
        Parity &curRow = mat.allocRow();
        curRow.mkBasis(++loc, blocks);
    }
    return mat;
}

/*
    Methods:
*/
void BinaryMatrix::extendRowsFor(IdxView newVars, size_t maxBlock) {
    rows.reserve(getNumRows() + newVars.size());
    for (size_t i = 0; i < newVars.size(); ++i) {
        Parity &newRow = allocRow();
        newRow.mkBasis(newVars[i], maxBlock);
    }
}

std::pair<bool, size_t> BinaryMatrix::establishPivotAt(size_t pvtRow, BitLocation pvtCol,
                                                       size_t trackingRow) {
    if (getRowAt(pvtRow).getBitAtLoc(pvtCol)) {
        return {true, trackingRow};
    }

    const size_t rowNum = getNumRows();
    for (size_t i = pvtRow + 1; i < rowNum; ++i) {
        if (getRowAt(i).getBitAtLoc(pvtCol)) {

            if (trackingRow != UNDIFINED_ROW) {
                if (trackingRow == i) {
                    trackingRow = pvtRow;
                } else if (trackingRow == pvtRow) {
                    trackingRow = i;
                }
            }

            swapRows(pvtRow, i);
            return {true, trackingRow};
        }
    }
    return {false, trackingRow};
}

void BinaryMatrix::forwardElimWrtPivot(size_t pvtRow, BitLocation pvtCol) {
    const size_t rowNum = getNumRows();
    for (size_t i = pvtRow + 1; i < rowNum; ++i) {
        rowReduceWrtPivot(i, pvtRow, pvtCol);
    }
}

void BinaryMatrix::backwardElimWrtPivot(size_t pvtRow, BitLocation pvtCol) {
    for (int i = pvtRow - 1; i >= 0; --i) {
        rowReduceWrtPivot(i, pvtRow, pvtCol);
    }
}

void BinaryMatrix::rowReduceWrtPivot(size_t row, size_t pvtRow, BitLocation pvtCol) {
    if (getRowAt(row).getBitAtLoc(pvtCol)) {
        addRowToRow(pvtRow, row);
    }
}

size_t BinaryMatrix::firstTrivialRow() const {
    return findFirstRowIf([](const Parity &row) { return row.isTrivial(); });
}

/*
    Print:
*/
namespace catalyst::phase_folding {
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const BinaryMatrix &mat) {
    for (auto it = mat.rows.begin(); it != mat.rows.end(); ++it) {
        os << *it << '\n';
    }
    return os;
}
} // namespace catalyst::phase_folding