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

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

// forward declare stim member classes to encapsulate dependencies
namespace stim {
struct FlexPauliString;
} // namespace stim

namespace catalyst {
namespace qec {

/// A PauliWord is a vector of strings representing Pauli string ("I", "X", "Y", "Z").
using PauliWord = llvm::SmallVector<std::string>;

/// A PauliStringWrapper provides a convenient interface for manipulating Pauli strings,
/// tracking corresponding qubits, and handling operations. It wraps the stim::FlexPauliString
/// class and adds additional functionality specific to the QEC dialect.
///
/// Key components:
/// - pauliString: The underlying stim::FlexPauliString representation
/// - correspondingQubits: Maps each Pauli operator to its associated MLIR Value
/// - op: The QEC operation this Pauli string is associated with
///
/// The wrapper provides methods for:
/// - Construction from text representations and PauliWords
/// - Computing commutation rules between Pauli strings
/// - Converting between different Pauli string formats
/// - Tracking sign and imaginary components
struct PauliStringWrapper {
  public:
    std::vector<Value> correspondingQubits;
    QECOpInterface op;

  private:
    std::unique_ptr<stim::FlexPauliString> pauliString;

  public:
    PauliStringWrapper(stim::FlexPauliString &&fps);

    // needed for use in std::pair (copy or move)
    PauliStringWrapper(PauliStringWrapper &&other);

    // If we omit the destructor declaration the default one will be "part of the header",
    // which doesn't have the concrete member types for Stim classes, and thus fails to compile.
    // So we declare it here and define the default destructor in the source file.
    ~PauliStringWrapper();

    // delete everything we don't need, in the future just define more as needed
    PauliStringWrapper(const PauliStringWrapper &other) = delete;
    PauliStringWrapper &operator=(const PauliStringWrapper &data) = delete;
    PauliStringWrapper &operator=(PauliStringWrapper &&data) = delete;

    static PauliStringWrapper from_pauli_word(const PauliWord &pauliWord);

    bool isNegative() const;
    bool isImaginary() const;

    void updateSign(bool sign);

    PauliWord get_pauli_word() const;

    bool commutes(const PauliStringWrapper &other) const;

    // Commutation Rules
    // P is Clifford, P' is non-Clifford
    // if P commutes with P' then PP' = P'P
    // if P anti-commutes with P' then PP' = -iPP' P
    // In here, P and P' are lhs and rhs, respectively.
    PauliStringWrapper computeCommutationRulesWith(const PauliStringWrapper &rhs) const;
};

////////////////////////////////////////////////////////////
//                  QEC Helper functions
////////////////////////////////////////////////////////////

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
                          QECOpInterface op);

/**
 * @brief Normalize the qubits of the two operations.
 *        The goal is to normalize the operations of the two operations to the same order and
 *        number of qubits.
 *        - Find the common qubits between the two operations.
 *        - Assign the common qubits to the corresponding qubits of the other operation.
 *
 *        e.g: lhs.getOutQubits() = [q0, q1] with PauliProduct = ["X", "Y"],
 *             rhs.getInQubits() = [q1, q2] with PauliProduct = ["Y", "Z"]
 *        -> lhs.correspondingQubits = [q0, q1, nullptr], rhs.correspondingQubits = [nullptr,
 *           q1, q2]
 *        -> lhs.pauliString = ["X", "Y", "I"], rhs.pauliString = ["I", "Y", "Z"]

 * @param lhs QECOpInterface of the left hand side
 * @param rhs QECOpInterface of the right hand side
 * @return PauliWordPair of the normalized pair of PauliStringWrapper
 */
PauliWordPair normalizePPROps(QECOpInterface lhs, QECOpInterface rhs);

// Remove Identity from the op's Pauli product and corresponding qubits from the list/
// The size of op.pauliProduct and qubits is assumed to be the same.
SmallVector<StringRef> removeIdentityPauli(QECOpInterface op, SmallVector<Value> &qubits);

/**
 * @brief Replace the value with the corresponding qubits.
 *        Corresponding qubits are the qubits that are the same in both operations.
 *        Those qubits are combination of OutQubit and InQubit.
 *        The goal is to replace the value with OutQubit with InQubit.
 *
 * @param lhsPauliWrapper PauliStringWrapper of the left hand side
 * @param rhsPauliWrapper PauliStringWrapper of the right hand side
 * @return SmallVector<Value> of the replaced qubits
 */
SmallVector<Value> replaceValueWithOperands(const PauliStringWrapper &lhsPauliWrapper,
                                            const PauliStringWrapper &rhsPauliWrapper);

/**
 * @brief Update the pauliWord of the right hand side operation.
 *
 * @param rhsOp QECOpInterface of the right hand side
 * @param newPauliWord PauliWord of the new pauliWord
 * @param rewriter PatternRewriter
 */
void updatePauliWord(QECOpInterface op, const PauliWord &newPauliWord, PatternRewriter &rewriter);

// Update the sign of the operation.
void updatePauliWordSign(QECOpInterface op, bool isNegated, PatternRewriter &rewriter);

// Extract the pauli string from the operation.
SmallVector<StringRef> extractPauliString(QECOpInterface op);

// No size limit when MaxPauliSize is 0
bool isNoSizeLimit(size_t MaxPauliSize);

// Combine the size check logic in one place
bool exceedPauliSizeLimit(size_t pauliSize, size_t MaxPauliSize);

} // namespace qec
} // namespace catalyst
