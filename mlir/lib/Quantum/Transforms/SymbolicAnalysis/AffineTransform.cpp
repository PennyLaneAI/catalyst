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
TransformLayout::TransformLayout(size_t n)
{
    inVars.reserve(n);
    std::iota(inVars.begin(), inVars.end(), 0); 
}
/*
    Operators:
*/
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const AffineTransform &trans)
{
    os << trans.mat;
    return os;
}

std::string AffineTransform::algebraicView() const
{
    return mat.algebraicView(getQubitNum());
}

/*
    Methods:
*/
void AffineTransform::extendQubitsTo(size_t newQubitNum)
{
    size_t curQubitNum = getQubitNum();
    size_t auxVarNum = getAuxVarNum();

    mat.extendRowsTo(newQubitNum, auxVarNum);
    
    layout.inVars.reserve(newQubitNum);
    for (size_t i = curQubitNum + 1; i <= newQubitNum; i++) {
        layout.inVars.push_back(i + auxVarNum);
    }
}

void AffineTransform::initQubit(size_t qubitIndex, bool basisState)
{
    mat.resetRow(qubitIndex);
    if (basisState == 1) {
        mat.flipAffineValueAtRow(qubitIndex);
    }
}

void AffineTransform::applyGateX(size_t qubitIndex) 
{
    mat.flipAffineValueAtRow(qubitIndex);
}

void AffineTransform::applyGateCNOT(size_t controlIndex, size_t targetIndex)
{
    mat.addRowToRow(controlIndex, targetIndex);
}

void AffineTransform::applyGateSWAP(size_t qubitIndex1, size_t qubitIndex2)
{
    mat.swapRows(qubitIndex1, qubitIndex2);
}

void AffineTransform::applyGateH(size_t qubitIndex)
{   
    size_t lastInd = getQubitNum() + getAuxVarNum() + 1;
    mat.setRow(qubitIndex, Parity::eVec(lastInd, lastInd));
    layout.auxVars.push_back(lastInd);
}

// uninterpreted gates.
void AffineTransform::applyGateU(llvm::ArrayRef<size_t> qubitIndices)
{
    llvm::outs() << "U on qubits ";
    for (size_t index : qubitIndices) {
        llvm::outs() << index << ", ";
    }
    llvm::outs() << ":\n";
    
    size_t n = qubitIndices.size();
    size_t lastInd = getQubitNum() + getAuxVarNum();

    layout.auxVars.reserve(getAuxVarNum() + n);
    for (size_t i = 0; i < n; i++) {
        mat.setRow(qubitIndices[i],
                          Parity::eVec(lastInd + n, lastInd + i + 1));
        layout.auxVars.push_back(lastInd + i + 1);
    }
} // make sure nothing leads to segment fault and index out of bounds.
