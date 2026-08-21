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

#include "AffineTransform.hpp"

using namespace catalyst::phase_folding;

/*
    Print:
*/
std::string AffineTransform::toString() const {
    std::string res = "";

    if (matrix.getNumRows() > 0) {
        res += "X";
        if (schema.auxVars.size() > 0) {
            res += " Y";
        }
        res += " | c\n";
    } 


    for (size_t i = 0; i < matrix.getNumRows(); ++i) {
        std::string pre = matrix.getRowAt(i).toStringWithOrder(schema.preVars);
        std::string aux = matrix.getRowAt(i).toStringWithOrder(schema.auxVars);
        bool affVal = matrix.getRowAt(i).getBitAtLoc(schema.affVal);

        res += (pre + " " + aux + " | " + std::to_string(affVal)) + "\n";
    }
    return res;
}

/*
    Methods:
*/
void AffineTransform::extendQubitsBy(size_t addQubitNum) {
    IdxView newVars = schema.allocPreVars(addQubitNum);
    matrix.extendRowsFor(newVars, schema.maxBlock());
}

void AffineTransform::prepareQubit(size_t wire, bool basisState) {
    matrix.resetRow(wire, schema.affVal.block);
    if (basisState == 1) {
        matrix.getRowMutableAt(wire).setBitAtLoc(schema.affVal);
    }
}

// uninterpreted gates.
void AffineTransform::applyGateU(llvm::ArrayRef<size_t> wires) {
    const size_t n = wires.size();
    IdxView auxVarLocs = schema.allocAuxVars(n);

    for (size_t i = 0; i < n; ++i) {
        matrix.setRowToBasis(wires[i], auxVarLocs[i]);
    }
}
