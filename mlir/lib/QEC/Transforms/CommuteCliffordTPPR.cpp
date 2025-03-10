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

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ScopedPrinter.h"
#include <cstdint>
#define DEBUG_TYPE "commute-clifford-t-ppr"

#include "stim.h"

#include "llvm/ADT/DenseSet.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"

using namespace stim;
using namespace mlir;
using namespace catalyst::qec;

namespace {

constexpr static size_t MAX_BITWORD = stim::MAX_BITWORD_WIDTH;

using PauliWords = llvm::SmallVector<std::string>;

struct PauliStringWrapper {
    stim::PauliString<MAX_BITWORD> pauliString;
    std::vector<Value> correspondingQubits;
    PPRotationOp *op;
    bool imaginary; // 0 for real, 1 for imaginary
    bool negated;   // 0 for positive, 1 for negative

    PauliStringWrapper(std::string text, bool imag, bool sign)
        : pauliString(stim::PauliString<MAX_BITWORD>::from_str(text.c_str())), imaginary(imag),
          negated(sign)
    {
        // allocated nullptr for correspondingQubits
        correspondingQubits.resize(pauliString.num_qubits, nullptr);
    }
    PauliStringWrapper(stim::PauliString<MAX_BITWORD> &pauliString) : pauliString(pauliString) {}
    PauliStringWrapper(stim::PauliString<MAX_BITWORD> &pauliString, bool imag, bool sign)
        : pauliString(pauliString), imaginary(imag), negated(sign)
    {
        // allocated nullptr for correspondingQubits
        correspondingQubits.resize(pauliString.num_qubits, nullptr);
    }
    PauliStringWrapper(stim::FlexPauliString &fps)
        : pauliString(fps.str()), imaginary(fps.imag), negated(fps.value.sign)
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

    static PauliStringWrapper from_pauli_words(PauliWords &pauliWords)
    {
        std::string pauliStringStr;
        for (auto pauli : pauliWords) {
            pauliStringStr += pauli;
        }
        return PauliStringWrapper::from_text(pauliStringStr);
    }

    stim::PauliStringRef<MAX_BITWORD> ref() { return pauliString.ref(); }

    std::string str() { return pauliString.str(); }

    stim::FlexPauliString flex() { return stim::FlexPauliString(pauliString.ref(), imaginary); }

    PauliWords get_pauli_words()
    {
        PauliWords pauliWords;
        auto str = pauliString.str();
        for (char c : str) {
            if (c == 'i')
                continue;
            if (c == '-')
                continue;
            if (c == '+')
                continue;
            if (c == '_') {
                pauliWords.push_back("I");
                continue;
            }
            pauliWords.push_back(std::string(1, c));
        }
        return pauliWords;
    }

    bool commutes(PauliStringWrapper &other) { return ref().commutes(other.ref()); }
};

// Track the Pauli words of the PPRotationOps
// The key is the Pauli word, the value is the qubits
// e.g:
// %qa:3 = qec.ppr ["X","Z","Y"](8) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
// %qb = qec.ppr ["Z","Y"](8) %qa#0, %qa#2 : !quantum.bit, !quantum.bit, !quantum.bit
// TrackedPauliWord of %qb is:
// [("Z", %qa#0), ("I", %qa#1), ("Y", %qa#2)]
using TrackedPauliWord = llvm::SmallVector<std::pair<StringRef, Value>>;

template <typename T>
PauliStringWrapper computePauliWords(const llvm::SetVector<Value> &qubits, const T &inOutQubits,
                                     PPRotationOp &op)
{
    // PauliNames initializes with "I" for each qubit
    PauliWords pauliWords(qubits.size(), "I");
    std::vector<Value> correspondingQubits;
    correspondingQubits.resize(qubits.size(), nullptr);
    for (auto [position, value] : llvm::enumerate(llvm::zip(inOutQubits, op.getPauliProduct()))) {
        auto qubit = std::get<0>(value);
        // Find the position of the qubit in array of qubits
        auto it = std::find(qubits.begin(), qubits.end(), qubit);
        if (it != qubits.end()) {
            position = std::distance(qubits.begin(), it);
            auto pauli = mlir::cast<mlir::StringAttr>(std::get<1>(value)).getValue();
            pauliWords[position] = pauli;
            correspondingQubits[position] = qubit;
        }
    }
    auto pauliStringWrapper = PauliStringWrapper::from_pauli_words(pauliWords);
    pauliStringWrapper.correspondingQubits = correspondingQubits;
    pauliStringWrapper.op = &op;
    return pauliStringWrapper;
}

using PauliWordsPair = std::pair<PauliStringWrapper, PauliStringWrapper>;

// normalize the Pauli product of two PPRotationOps
PauliWordsPair normalizePPROps(PPRotationOp &lhs, PPRotationOp &rhs)
{
    auto lhsQubits = lhs.getOutQubits();
    auto rhsQubits = rhs.getInQubits();

    llvm::SetVector<Value> qubits;
    qubits.insert(lhsQubits.begin(), lhsQubits.end());
    qubits.insert(rhsQubits.begin(), rhsQubits.end());

    PauliStringWrapper lhsPauliStringWrapper = computePauliWords(qubits, lhsQubits, lhs);
    PauliStringWrapper rhsPauliStringWrapper = computePauliWords(qubits, rhsQubits, rhs);

    // normalize the qubits
    for (auto [lhs, rhs] : llvm::zip(lhsPauliStringWrapper.correspondingQubits,
                                     rhsPauliStringWrapper.correspondingQubits)) {
        if (lhs == rhs)
            continue;

        if (lhs == nullptr) {
            lhs = rhs;
        }
        else {
            rhs = lhs;
        }
    }

    return std::make_pair(lhsPauliStringWrapper, rhsPauliStringWrapper);
}

// check if two Pauli words commute or anti-commute

bool isCommute(PauliStringWrapper &lhs, PauliStringWrapper &rhs) { return lhs.commutes(rhs); }

bool isNonClifford(PPRotationOp op)
{
    // check the attribuate if it is 8
    int64_t rotationKind = static_cast<int16_t>(op.getRotationKind());
    return rotationKind == 8 || rotationKind == -8;
}

// Commutation Rules
// P is Clifford, P' is non-Clifford
// if P commutes with P' then PP' = P'P
// if P anti-commutes with P' then PP' = -iPP' P
PauliStringWrapper computeCommutationRules(PauliWords &lhs, PauliWords &rhs)
{
    auto lhsPauliStringFlex = PauliStringWrapper::from_pauli_words(lhs).flex();
    auto rhsPauliStringFlex = PauliStringWrapper::from_pauli_words(rhs).flex();
    auto imagFlex = FlexPauliString::from_text("i");

    // P * P' * i
    FlexPauliString result = lhsPauliStringFlex * rhsPauliStringFlex * imagFlex;
    return PauliStringWrapper(result);
}

LogicalResult visitPPRotationOp(PPRotationOp op,
                                std::function<LogicalResult(PPRotationOp)> callback)
{
    if (isNonClifford(op)) {
        return failure();
    }

    // TODO: Consider arbitrary gate
    Operation *nextOp = op->getNextNode();
    if (PPRotationOp nextPPROp = dyn_cast_or_null<PPRotationOp>(nextOp)) {
        if (isNonClifford(nextPPROp)) {
            return callback(nextPPROp);
        }
    }

    return failure();
}

void printPauliWords(const PauliWords &pauliWords)
{
    for (auto pauli : pauliWords) {
        llvm::outs() << pauli << " ";
    }
    llvm::outs() << "\n";
}

void updatePauliProduct(PPRotationOp op, PauliStringWrapper pauli, PatternRewriter &rewriter)
{
    auto pauliProductArray = pauli.get_pauli_words();
    SmallVector<StringRef> pauliProductArrayAttr(pauliProductArray.begin(),
                                                 pauliProductArray.end());
    auto pauliProduct = rewriter.getStrArrayAttr(pauliProductArrayAttr);
    op.setPauliProductAttr(pauliProduct);
}

void moveCliffordPastNonClifford(PPRotationOp lhs, PPRotationOp rhs, PauliStringWrapper lhsPauli,
                                 PauliStringWrapper rhsPauli, PauliStringWrapper *result,
                                 PatternRewriter &rewriter)
{
    assert(!isNonClifford(lhs) && "LHS Operation is not Clifford");
    assert(isNonClifford(rhs) && "RHS Operation is not non-Clifford");

    // Update Pauli words of RHS
    if (result != nullptr) {
        updatePauliProduct(rhs, *result, rewriter);
    }
    else {
        updatePauliProduct(rhs, rhsPauli, rewriter);
    }

    // TODO: Update sign of LHS
    if (result != nullptr) {
        uint16_t negated = result->negated ? -1 : 1;
        lhs.setRotationKindAttr(rewriter.getI16IntegerAttr(lhs.getRotationKind() * negated));
    }

    // Update Operands of RHS
    // lhsPauli.correspondingQubits consists of:
    //    - OutQubit: initilized qubit
    //    - InQubit: new qubits from RHS op
    //
    SmallVector<Value> newRHSOperands(rhsPauli.correspondingQubits.size(), nullptr);
    for (unsigned i = 0; i < rhsPauli.correspondingQubits.size(); i++) {
        for (unsigned j = 0; j < rhs.getInQubits().size(); j++) {
            if (rhs.getInQubits()[j] == rhsPauli.correspondingQubits[i]) {
                newRHSOperands[i] = rhs.getInQubits()[j];
            }
        }
        for (unsigned j = 0; j < lhs.getOutQubits().size(); j++) {
            if (lhs.getOutQubits()[j] == rhsPauli.correspondingQubits[i]) {
                newRHSOperands[i] = lhs.getInQubits()[j];
            }
        }
    }

    SmallVector<Type> outQubitsTypesList;
    for (auto qubit : newRHSOperands) {
        outQubitsTypesList.push_back(qubit.getType());
    }

    auto nonCliffordOp =
        rewriter.create<PPRotationOp>(rhs->getLoc(), outQubitsTypesList, rhs.getPauliProduct(),
                                      rhs.getRotationKindAttr(), newRHSOperands);

    // update the use of value in newRHSOperands
    for (unsigned i = 0; i < newRHSOperands.size(); i++) {
        newRHSOperands[i].replaceAllUsesExcept(nonCliffordOp.getOutQubits()[i], nonCliffordOp);
    }

    rewriter.moveOpBefore(nonCliffordOp, lhs);

    for (auto [outQubit, inQubit] : llvm::zip(rhs.getOutQubits(), rhs.getInQubits())) {
        rewriter.replaceAllUsesWith(outQubit, inQubit);
    }
    rewriter.eraseOp(rhs);
}

struct CommuteCliffordTPPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        return visitPPRotationOp(op, [&](PPRotationOp nextPPROp) {
            PauliWordsPair normOps = normalizePPROps(op, nextPPROp);

            auto lhsPauliWords = normOps.first.get_pauli_words();
            auto rhsPauliWords = normOps.second.get_pauli_words();

            // DEBUG: print the Pauli words
            printPauliWords(lhsPauliWords);
            printPauliWords(rhsPauliWords);

            if (isCommute(normOps.first, normOps.second)) {
                moveCliffordPastNonClifford(op, nextPPROp, normOps.first, normOps.second, nullptr,
                                            rewriter);
                return success();
            }
            else {
                auto resultStr = computeCommutationRules(lhsPauliWords, rhsPauliWords);
                llvm::outs() << resultStr.str() << "\n";
                moveCliffordPastNonClifford(op, nextPPROp, normOps.first, normOps.second,
                                            &resultStr, rewriter);
                llvm::outs() << resultStr.str() << "\n";
                return success();
            }
        });
    }
};
} // namespace

namespace catalyst {
namespace qec {

void populateCommuteCliffordTPPRPatterns(mlir::RewritePatternSet &patterns)
{
    patterns.add<CommuteCliffordTPPR>(patterns.getContext(), 1);
}
} // namespace qec

} // namespace catalyst
