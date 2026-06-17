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

#include "AffineTransform.h"

/*
    Constructors:
*/
AffineTransform::AffineTransform(size_t n)
{
    exprMatrix.reserve(n);
    for (size_t i = 0; i < n; i++) {
        exprMatrix.push_back(Parity::eVec(n, i + 1));
    }
}

AffineTransform AffineTransform::identity(size_t n) { return AffineTransform(n); }

/*
    Operators:
*/
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const AffineTransform &trans)
{
    for (auto it = trans.exprMatrix.begin(); it != trans.exprMatrix.end(); it++) {
        os << *it << '\n';
    }
    return os;
}

std::string AffineTransform::algebraicView(size_t qubitNum) const
{
    std::string res = "";
    for (size_t i = 0; i < exprMatrix.size(); i++) {
        res +=
            ("x'" + std::to_string(i + 1) + " = " + exprMatrix[i].algebraicView(qubitNum) + '\n');
    }
    return res;
}

/*
    Getters and Setters:
*/
void AffineTransform::setRow(size_t row, const Parity &parity) { getRowMutable(row) = parity; }

void AffineTransform::resetRow(size_t row) { getRowMutable(row).reset(); }

void AffineTransform::flipAffineValueAtRow(size_t row) { getRowMutable(row).flipAffineValue(); }

/*
    Methods:
*/
void AffineTransform::addRowWithParity(size_t row, const Parity &parity)
{
    getRowMutable(row) += parity;
}

void AffineTransform::addRows(size_t sourceRow, size_t targetRow)
{ // E_i,j
    getRowMutable(targetRow) += getRow(sourceRow);
}

void AffineTransform::swapRows(size_t row1, size_t row2)
{
    std::swap(getRowMutable(row1), getRowMutable(row2));
}

void AffineTransform::extendTo(size_t newRowNum, size_t auxVarNum)
{
    if (newRowNum > getRowNum()) {
        exprMatrix.reserve(newRowNum);
        for (size_t i = getRowNum() + 1; i <= newRowNum; i++) {
            exprMatrix.push_back(Parity::eVec(newRowNum + auxVarNum, i + auxVarNum));
        }
    } // it might be more efficient to resize to newRowNum, and then just turn the eVec bits on for
      // the new rows.
}
