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

#pragma once

#include "stim.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "llvm/ADT/SetVector.h"

using namespace stim;
using namespace mlir;

namespace catalyst {
namespace qec {

constexpr static size_t MAX_BITWORD = stim::MAX_BITWORD_WIDTH;

/// A PauliWord is a vector of strings representing Pauli string ("I", "X", "Y", "Z").
using PauliWord = llvm::SmallVector<std::string>;

/// A PauliStringWrapper provides a convenient interface for manipulating Pauli strings,
/// tracking corresponding qubits, and handling operations. It wraps the stim::PauliString
/// class and adds additional functionality specific to the QEC dialect.
///
/// Key components:
/// - pauliString: The underlying stim::PauliString representation
/// - correspondingQubits: Maps each Pauli operator to its associated MLIR Value
/// - op: The QEC operation this Pauli string is associated with
/// - imaginary: Tracks whether the Pauli string has an imaginary component
///
/// The wrapper provides methods for:
/// - Construction from text representations and PauliWords
/// - Computing commutation rules between Pauli strings
/// - Converting between different Pauli string formats
/// - Tracking sign and imaginary components
struct PauliStringWrapper {
    stim::PauliString<MAX_BITWORD> pauliString;
    std::vector<Value> correspondingQubits;
    QECOpInterface op;
    bool imaginary; // 0 for real, 1 for imaginary

    PauliStringWrapper(std::string text, bool imag, bool sign)
        : pauliString(stim::PauliString<MAX_BITWORD>::from_str(text.c_str())), imaginary(imag)
    {
        // allocated nullptr for correspondingQubits
        correspondingQubits.resize(pauliString.num_qubits, nullptr);
        pauliString.sign = sign;
    }
    PauliStringWrapper(stim::PauliString<MAX_BITWORD> &pauliString) : pauliString(pauliString) {}
    PauliStringWrapper(stim::PauliString<MAX_BITWORD> &pauliString, bool imag, bool sign)
        : pauliString(pauliString), imaginary(imag)
    {
        // allocated nullptr for correspondingQubits
        correspondingQubits.resize(pauliString.num_qubits, nullptr);
        pauliString.sign = sign;
    }
    PauliStringWrapper(stim::FlexPauliString &fps) : pauliString(fps.str()), imaginary(fps.imag)
    {
        // allocated nullptr for correspondingQubits
        correspondingQubits.resize(pauliString.num_qubits, nullptr);
    }

    static PauliStringWrapper from_text(std::string_view text)
    {
        // check if pauliString contain + or -
        bool negated = 0;
        bool imaginary = 0;
        if (text.starts_with("-")) {
            negated = true;
            text = text.substr(1);
        }
        else if (text.starts_with("+")) {
            text = text.substr(1);
        }
        if (text.starts_with("i")) {
            imaginary = true;
            text = text.substr(1);
        }
        auto str = std::string(text);
        return PauliStringWrapper(std::string(text), imaginary, negated);
    }

    static PauliStringWrapper from_pauli_words(PauliWord &pauliWord)
    {
        std::string pauliStringStr;
        for (auto pauli : pauliWord) {
            pauliStringStr += pauli;
        }
        return PauliStringWrapper::from_text(pauliStringStr);
    }

    stim::PauliStringRef<MAX_BITWORD> ref() { return pauliString.ref(); }

    std::string str() const { return pauliString.str(); }

    stim::FlexPauliString flex() { return stim::FlexPauliString(pauliString.ref(), imaginary); }

    PauliWord get_pauli_words()
    {
        PauliWord pauliWord;
        auto str = pauliString.str();
        for (char c : str) {
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

    bool commutes(PauliStringWrapper &other) { return ref().commutes(other.ref()); }

    // Commutation Rules
    // P is Clifford, P' is non-Clifford
    // if P commutes with P' then PP' = P'P
    // if P anti-commutes with P' then PP' = -iPP' P
    // In here, P and P' are lhs and rhs, respectively.
    PauliStringWrapper computeCommutationRulesWith(PauliStringWrapper &rhs);
};

using PauliWordPair = std::pair<PauliStringWrapper, PauliStringWrapper>;

/**
 * @brief Expland the op's operands to the set of operands.
 *        - Initialize the new pauliWord with "I" for each qubit.
 *        - Find location of inOutOperands in set of operands
 *        - Assign the inOutOperands' pauli word to new pauliWord in that location
 *        e.g: qubits = {q0, q1, q2}, inQubitOut = [q1, q2], op.getPauliProduct() = ["X", "Y"]
 *        -> ["I", "X", "Y"]
 *
 * @tparam T either mlir::Operation::operand_range or mlir::Operation::result_range
 * @param qubits set of combination of qubit operands
 * @param inOutOperands either value from.inQubits or .outQubits from op
 * @param op QECOpInterface
 * @return PauliWord of the expanded pauliWord
 */
template <typename T>
PauliWord expandPauliWord(const llvm::SetVector<Value> &operands, const T &inOutOperands,
                          QECOpInterface &op);

/**
 * @brief Normalize the qubits of the two operations.
 *        The goal is to normalize the operations of the two operaitons to the same order and number
 of qubits.
 *        - Find the common qubits between the two operations.
 *        - Assign the common qubits to the corresponding qubits of the other operation.
 *
 *        e.g: lhs.getOutQubits() = [q0, q1] with PauliProduct = ["X", "Y"],
 *             rhs.getInQubits() = [q1, q2] with PauliProduct = ["Y", "Z"]
 *        -> lhs.correspondingQubits = [q0, q1, nullptr], rhs.correspondingQubits = [nullptr, q1,
 q2]
 *        -> lhs.pauliString = ["X", "Y", "I"], rhs.pauliString = ["I", "Y", "Z"]

 * @param lhs QECOpInterface of the left hand side
 * @param rhs QECOpInterface of the right hand side
 * @return PauliWordPair of the normalized pair of PauliStringWrapper
 */
PauliWordPair normalizePPROps(QECOpInterface lhs, QECOpInterface rhs);

// Remove the value of newRHSOperands that based on index of 'rhs' where the value is 'I' or '_'
SmallVector<StringRef> removeIdentityPauli(QECOpInterface rhs, SmallVector<Value> &newRHSOperands);

/**
 * @brief Replace the value with the corresponding qubits.
 *        Corresponding qubits are the qubits that are the same in both operations.
 *        Those qubits are commbination of OutQubit and InQubit.
 *        The goal is to replace the value with OutQubit with InQubit.
 *
 * @param lhsPauliWrapper PauliStringWrapper of the left hand side
 * @param rhsPauliWrapper PauliStringWrapper of the right hand side
 * @return SmallVector<Value> of the replaced qubits
 */
SmallVector<Value> replaceValueWithOperands(PauliStringWrapper lhsPauliWrapper,
                                            PauliStringWrapper rhsPauliWrapper);

/**
 * @brief Update the pauliWord of the right hand side operation.
 *
 * @param rhsOp QECOpInterface of the right hand side
 * @param newPauliWord PauliWord of the new pauliWord
 * @param rewriter PatternRewriter
 */
void updatePauliWord(QECOpInterface op, PauliWord newPauliWord, PatternRewriter &rewriter);

// Update the sign of the operation.
// TODO: Using QECOpInterface instead of PPRotationOp
void updatePauliWordSign(PPRotationOp op, bool isNegated, PatternRewriter &rewriter);

} // namespace qec
} // namespace catalyst
