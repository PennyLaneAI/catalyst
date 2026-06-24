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

#include "BinaryMatrix.h"

/*
    Constructors:
*/
BinaryMatrix::BinaryMatrix(size_t n)
{
    exprRows.reserve(n);
    for (size_t i = 0; i < n; i++) {
        exprRows.push_back(Parity::eVec(n, i + 1));
    }
}

BinaryMatrix BinaryMatrix::identity(size_t n) { return BinaryMatrix(n); }

/*
    Operators:
*/
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const BinaryMatrix &mat)
{
    for (auto it = mat.exprRows.begin(); it != mat.exprRows.end(); it++) {
        os << *it << '\n';
    }
    return os;
}

std::string BinaryMatrix::algebraicView(size_t qubitNum) const
{
    std::string res = "";
    for (size_t i = 0; i < exprRows.size(); i++) {
        res +=
            ("x'" + std::to_string(i + 1) + " = " + exprRows[i].algebraicView(qubitNum) + '\n');
    }
    return res;
}

/*
    Getters and Setters:
*/
void BinaryMatrix::setRow(size_t row, const Parity &parity) { getRowMutable(row) = parity; }

void BinaryMatrix::resetRow(size_t row) { getRowMutable(row).reset(); }

void BinaryMatrix::flipAffineValueAtRow(size_t row) { getRowMutable(row).flipAffineValue(); }

/*
    Methods:
*/
void BinaryMatrix::extendRowsTo(size_t newRowNum, size_t auxVarNum)
{
    if (newRowNum > getRowNum()) {
        exprRows.reserve(newRowNum);
        for (size_t i = getRowNum() + 1; i <= newRowNum; i++) {
            exprRows.push_back(Parity::eVec(newRowNum + auxVarNum, i + auxVarNum));
        }
    } // it might be more efficient to resize to newRowNum, and then just turn the eVec bits on for
      // the new rows.
}

void BinaryMatrix::swapRows(size_t row1, size_t row2)
{
    std::swap(getRowMutable(row1), getRowMutable(row2));
}

void BinaryMatrix::addParityToRow(size_t row, const Parity &parity)
{
    getRowMutable(row) += parity;
}

void BinaryMatrix::addRowToRow(size_t sourceRow, size_t targetRow)
{ // E_i,j
    getRowMutable(targetRow) += getRow(sourceRow);
}

void BinaryMatrix::toEchelonForm(std::vector<int>& colOrd)
{
    for (size_t i = 0, j = 0; i < getRowNum() && j < colOrd.size(); j++) {
        if (setPivotRow(i, colOrd[j])) {
            rowReduceWithPivot(i, colOrd[j]);
            i++;
        }        
    }
}

void BinaryMatrix::toReducedEchelonForm(std::vector<int>& colOrd)
{
    // toEchelonForm(colOrd);
}

bool BinaryMatrix::setPivotRow(size_t pivotRow, size_t pivotCol)
{
    if (getRow(pivotRow).getBitAt(pivotCol)) {
        return true;
    }
    for (size_t i = pivotRow + 1; i < getRowNum(); i++) {
        if (getRow(i).getBitAt(pivotCol)) {
            swapRows(pivotRow, i);
            return true;
        }
    }
    return false;
}

void BinaryMatrix::rowReduceWithPivot(size_t pivotRow, size_t pivotCol)
{
    for (size_t i = pivotRow + 1; i < getRowNum(); i++) {
        if (getRow(i).getBitAt(pivotCol)) {
            addRowToRow(pivotRow, i);
        }
    }
}
