// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "QEC/Utils/PauliStringWrapper.h"

namespace catalyst {
namespace qec {

PauliStringWrapper PauliStringWrapper::computeCommutationRulesWith(PauliStringWrapper &rhs)
{
    // P * P' * i
    FlexPauliString result = flex() * rhs.flex() * FlexPauliString::from_text("i");
    assert(!result.imag && "Resulting Pauli string should be real");
    return PauliStringWrapper(result);
}

template <typename T>
PauliWord expandPauliWord(const llvm::SetVector<Value> &operands, const T &inOutOperands,
                          QECOpInterface &op)
{
    PauliWord pauliWord(operands.size(), "I");
    for (auto [qubit, pauli] : llvm::zip(inOutOperands, op.getPauliProduct())) {
        // Find the position of the qubit in array of qubits
        auto it = std::find(operands.begin(), operands.end(), qubit);
        if (it != operands.end()) {
            auto position = std::distance(operands.begin(), it);
            auto pauliStr = mlir::cast<mlir::StringAttr>(pauli).getValue();
            pauliWord[position] = pauliStr;
        }
    }
    return pauliWord;
}

PauliWordPair normalizePPROps(QECOpInterface lhs, QECOpInterface rhs)
{
    auto lhsQubits = lhs.getOutQubits();
    auto rhsQubits = rhs.getInQubits();

    llvm::SetVector<Value> qubits;
    qubits.insert(lhsQubits.begin(), lhsQubits.end());
    qubits.insert(rhsQubits.begin(), rhsQubits.end());

    PauliWord lhsPauliWord = expandPauliWord(qubits, lhsQubits, lhs);
    PauliWord rhsPauliWord = expandPauliWord(qubits, rhsQubits, rhs);

    PauliStringWrapper lhsPSWrapper = PauliStringWrapper::from_pauli_word(lhsPauliWord);
    PauliStringWrapper rhsPSWrapper = PauliStringWrapper::from_pauli_word(rhsPauliWord);

    lhsPSWrapper.correspondingQubits = std::vector<Value>(qubits.begin(), qubits.end());
    rhsPSWrapper.correspondingQubits = lhsPSWrapper.correspondingQubits;

    lhsPSWrapper.op = lhs;
    rhsPSWrapper.op = rhs;

    lhsPSWrapper.pauliString.sign = (int16_t)lhs.getRotationKind() < 0;
    rhsPSWrapper.pauliString.sign = (int16_t)rhs.getRotationKind() < 0;

    return std::make_pair(lhsPSWrapper, rhsPSWrapper);
}

SmallVector<StringRef> removeIdentityPauli(QECOpInterface rhs, SmallVector<Value> &newRHSOperands)
{
    auto pauliProduct = rhs.getPauliProduct();
    SmallVector<StringRef> pauliProductArrayRef;

    for (auto [i, pauli] : llvm::enumerate(pauliProduct)) {
        auto pauliStr = mlir::cast<mlir::StringAttr>(pauli).getValue();
        if (pauliStr == "I" || pauliStr == "_") {
            newRHSOperands.erase(newRHSOperands.begin() + i);
            continue;
        }
        pauliProductArrayRef.push_back(pauliStr);
    }

    return pauliProductArrayRef;
}

SmallVector<Value> replaceValueWithOperands(PauliStringWrapper lhsPauliWrapper,
                                            PauliStringWrapper rhsPauliWrapper)
{
    auto lhs = lhsPauliWrapper.op;
    auto rhs = rhsPauliWrapper.op;
    auto rhsPauliSize = rhsPauliWrapper.correspondingQubits.size();

    // lhsPauli.correspondingQubits consists of:
    //    - OutQubit: initilized qubit
    //    - InQubit: new qubits from RHS op
    SmallVector<Value> newRHSOperands(rhsPauliSize, nullptr);
    for (unsigned i = 0; i < rhsPauliSize; i++) {
        for (unsigned j = 0; j < rhs.getInQubits().size(); j++) {
            if (rhs.getInQubits()[j] == rhsPauliWrapper.correspondingQubits[i]) {
                newRHSOperands[i] = rhs.getInQubits()[j];
            }
        }
        for (unsigned j = 0; j < lhs.getOutQubits().size(); j++) {
            if (lhs.getOutQubits()[j] == rhsPauliWrapper.correspondingQubits[i]) {
                newRHSOperands[i] = lhs.getInQubits()[j];
            }
        }
    }
    return newRHSOperands;
}

void updatePauliWord(QECOpInterface op, PauliWord newPauliWord, PatternRewriter &rewriter)
{
    SmallVector<StringRef> pauliProductArrayAttr(newPauliWord.begin(), newPauliWord.end());
    auto pauliProduct = rewriter.getStrArrayAttr(pauliProductArrayAttr);
    op.setPauliProductAttr(pauliProduct);
}

void updatePauliWordSign(QECOpInterface op, bool isNegated, PatternRewriter &rewriter)
{
    int16_t rotationKind = static_cast<int16_t>(op.getRotationKind());
    int16_t sign = isNegated ? -1 : 1;
    rotationKind = (rotationKind < 0 ? -rotationKind : rotationKind) * sign;
    op.setRotationKind(rotationKind);
}

} // namespace qec
} // namespace catalyst
