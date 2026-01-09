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

#include <stim/mem/simd_word.h>
#include <stim/stabilizers/flex_pauli_string.h>
#include <stim/stabilizers/pauli_string.h>

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOps.h"
#include "QEC/Utils/PauliStringWrapper.h"

namespace catalyst {
namespace qec {

PauliStringWrapper::PauliStringWrapper(stim::FlexPauliString &&fps)
{
    pauliString = std::make_unique<stim::FlexPauliString>(std::move(fps));

    // allocated nullptr for correspondingQubits
    correspondingQubits.resize(pauliString->value.num_qubits, nullptr);
}

PauliStringWrapper::PauliStringWrapper(PauliStringWrapper &&other)
{
    pauliString.reset(other.pauliString.release());
    correspondingQubits = std::move(other.correspondingQubits);
    op = other.op;
}

PauliStringWrapper::~PauliStringWrapper() = default;

PauliStringWrapper PauliStringWrapper::from_pauli_word(const PauliWord &pauliWord)
{
    std::string pauliStringStr;
    for (auto pauli : pauliWord) {
        pauliStringStr += pauli;
    }
    return PauliStringWrapper(stim::FlexPauliString::from_text(pauliStringStr));
}

PauliStringWrapper PauliStringWrapper::from_qec_op(QECOpInterface op)
{
    std::string pauliStringStr;
    for (auto pauli : op.getPauliProduct()) {
        auto pauliStr = mlir::cast<mlir::StringAttr>(pauli).getValue();
        pauliStringStr += pauliStr;
    }
    return PauliStringWrapper(stim::FlexPauliString::from_text(pauliStringStr));
}

bool PauliStringWrapper::isNegative() const { return pauliString->value.sign; }
bool PauliStringWrapper::isImaginary() const { return pauliString->imag; }

void PauliStringWrapper::updateSign(bool sign) { pauliString->value.sign = sign; }

PauliWord PauliStringWrapper::get_pauli_word() const
{
    PauliWord pauliWord;

    for (char c : pauliString->value.str()) {
        if (c == 'i' || c == '-' || c == '+')
            continue;
        if (c == '_') {
            pauliWord.push_back("I");
            continue;
        }
        pauliWord.emplace_back(1, c);
    }

    return pauliWord;
}

bool PauliStringWrapper::commutes(const PauliStringWrapper &other) const
{
    return this->pauliString->value.ref().commutes(other.pauliString->value.ref());
}

PauliStringWrapper
PauliStringWrapper::computeCommutationRulesWith(const PauliStringWrapper &rhs) const
{
    stim::FlexPauliString result = *rhs.pauliString;
    assert(llvm::isa<PPRotationOp>(this->op) && "Clifford Operation is not PPRotationOp");
    auto this_op = llvm::cast<PPRotationOp>(this->op);
    if (this_op.hasPiOverTwoRotation()) {
        // -P'
        result.value.sign = !result.value.sign;
    }
    else if (this_op.hasPiOverFourRotation()) {
        // P * P' * i
        result = (*this->pauliString) * result * stim::FlexPauliString::from_text("i");
    }
    else {
        llvm_unreachable("Clifford rotation should be π/2 or π/4");
    }
    assert(!result.imag && "Resulting Pauli string should be real");
    return PauliStringWrapper(std::move(result));
}

template <typename T, typename U>
PauliWord expandPauliWord(const T &operands, const U &inOutOperands, QECOpInterface op)
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

// Emit explicit instantiation for the Value-based specialization used by QECLayer
template PauliWord expandPauliWord<llvm::SetVector<Value>, std::vector<Value>>(
    const llvm::SetVector<Value> &, const std::vector<Value> &, QECOpInterface);

PauliWordPair normalizePPROps(QECOpInterface lhs, QECOpInterface rhs)
{
    return normalizePPROps(lhs, rhs, lhs.getOutQubits(), rhs.getInQubits());
}

PauliWordPair normalizePPROps(QECOpInterface lhs, QECOpInterface rhs, ValueRange lhsQubits,
                              ValueRange rhsQubits)
{
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

    auto applySignFromOp = [](PauliStringWrapper &wrapper, QECOpInterface qecOp) {
        Operation *operation = qecOp.getOperation();

        if (auto pprOp = dyn_cast<PPRotationOp>(operation)) {
            wrapper.updateSign(static_cast<int16_t>(pprOp.getRotationKind()) < 0);
            return;
        }

        if (auto ppmOp = dyn_cast<PPMeasurementOp>(operation)) {
            wrapper.updateSign(static_cast<int16_t>(ppmOp.getRotationSign()) < 0);
        }
    };

    applySignFromOp(lhsPSWrapper, lhs);
    applySignFromOp(rhsPSWrapper, rhs);

    return std::make_pair(std::move(lhsPSWrapper), std::move(rhsPSWrapper));
}

SmallVector<StringRef> removeIdentityPauli(QECOpInterface op, SmallVector<Value> &qubits)
{
    assert(op.getPauliProduct().size() == qubits.size());

    auto pauliProduct = op.getPauliProduct();
    SmallVector<StringRef> pauliProductArrayRef;
    int erased = 0;

    for (auto [i, pauli] : llvm::enumerate(pauliProduct)) {
        auto pauliStr = mlir::cast<mlir::StringAttr>(pauli).getValue();
        if (pauliStr == "I" || pauliStr == "_") {
            qubits.erase(qubits.begin() + i - erased);
            erased++;
            continue;
        }
        pauliProductArrayRef.push_back(pauliStr);
    }

    return pauliProductArrayRef;
}

SmallVector<Value> replaceValueWithOperands(const PauliStringWrapper &lhsPauliWrapper,
                                            const PauliStringWrapper &rhsPauliWrapper)
{
    auto lhs = lhsPauliWrapper.op;
    auto rhs = rhsPauliWrapper.op;
    auto rhsPauliSize = rhsPauliWrapper.correspondingQubits.size();

    // lhsPauli.correspondingQubits consists of:
    //    - OutQubit: initialized qubit
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

void updatePauliWord(QECOpInterface op, const PauliWord &newPauliWord, PatternRewriter &rewriter)
{
    SmallVector<StringRef> pauliProductArrayAttr(newPauliWord.begin(), newPauliWord.end());
    auto pauliProduct = rewriter.getStrArrayAttr(pauliProductArrayAttr);
    op.setPauliProductAttr(pauliProduct);
}

void updatePauliWordSign(QECOpInterface op, bool isNegated, PatternRewriter &rewriter)
{
    if (auto pprOp = dyn_cast<PPRotationOp>(op.getOperation())) {
        int16_t rotationKind = static_cast<int16_t>(pprOp.getRotationKind());
        int16_t sign = isNegated ? -1 : 1;
        rotationKind = (rotationKind < 0 ? -rotationKind : rotationKind) * sign;
        pprOp.setRotationKind(rotationKind);
    }
    else if (auto ppmOp = dyn_cast<PPMeasurementOp>(op.getOperation())) {
        int16_t rotationSign = static_cast<int16_t>(ppmOp.getRotationSign());
        rotationSign = (rotationSign < 0 ? -rotationSign : rotationSign) * (isNegated ? -1 : 1);
        ppmOp.setRotationSign(rotationSign);
    }
}

SmallVector<StringRef> extractPauliString(QECOpInterface op)
{
    SmallVector<StringRef> pauliWord;
    for (auto pauli : op.getPauliProduct()) {
        pauliWord.emplace_back(mlir::cast<mlir::StringAttr>(pauli).getValue());
    }
    return pauliWord;
}

bool isNoSizeLimit(size_t MaxPauliSize) { return MaxPauliSize == 0; }

bool exceedPauliSizeLimit(size_t pauliSize, size_t MaxPauliSize)
{
    // No size limit
    if (isNoSizeLimit(MaxPauliSize)) {
        return false;
    }
    return pauliSize > MaxPauliSize;
}

bool operator==(const PauliWord &lhs, const PauliWord &rhs) { return llvm::equal(lhs, rhs); }
bool operator!=(const PauliWord &lhs, const PauliWord &rhs) { return !(lhs == rhs); }

} // namespace qec
} // namespace catalyst
