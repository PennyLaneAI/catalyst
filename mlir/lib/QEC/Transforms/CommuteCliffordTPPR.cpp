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

#define DEBUG_TYPE "commute-clifford-t-ppr"

#include "stim.h"

#include "llvm/Support/Debug.h"

#include "mlir/Transforms/TopologicalSortUtils.h"

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

    std::string str() const { return pauliString.str(); }

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
    pauliStringWrapper.negated = (int16_t)op.getRotationKind() < 0;
    pauliStringWrapper.pauliString.sign = pauliStringWrapper.negated;
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

    PauliStringWrapper lhsPSWrapper = computePauliWords(qubits, lhsQubits, lhs);
    PauliStringWrapper rhsPSWrapper = computePauliWords(qubits, rhsQubits, rhs);

    // normalize the qubits
    for (auto [lhs, rhs] :
         llvm::zip(lhsPSWrapper.correspondingQubits, rhsPSWrapper.correspondingQubits)) {
        if (lhs != rhs) {
            rhs = (lhs != nullptr) ? lhs : rhs;
            lhs = rhs;
        }
    }

    return std::make_pair(lhsPSWrapper, rhsPSWrapper);
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
// In here, P and P' are lhs and rhs, respectively.
PauliStringWrapper computeCommutationRules(PauliStringWrapper &lhs, PauliStringWrapper &rhs)
{
    // P * P' * i
    FlexPauliString result = lhs.flex() * rhs.flex() * FlexPauliString::from_text("i");
    return PauliStringWrapper(result);
}

bool verifyPrevNonClifford(PPRotationOp op, Operation *prevOp)
{
    if (prevOp == nullptr)
        return true;

    if (prevOp == op)
        return false;

    if (prevOp->isBeforeInBlock(op))
        return true;

    for (auto userOp : prevOp->getOperands()) {
        if (!verifyPrevNonClifford(op, userOp.getDefiningOp()))
            return false;
    }
    return true;
}

bool verifyNextNonClifford(PPRotationOp op, PPRotationOp nextOp)
{
    if (!isNonClifford(nextOp))
        return false;

    if (nextOp == nullptr)
        return false;

    for (auto userOp : nextOp.getOperands()) {
        auto defOp = userOp.getDefiningOp();

        if (defOp == op)
            continue;

        if (!verifyPrevNonClifford(op, defOp))
            return false;
    }

    return true;
}

LogicalResult visitPPRotationOp(PPRotationOp op,
                                std::function<LogicalResult(PPRotationOp)> callback)
{
    if (isNonClifford(op))
        return failure();

    for (auto userOp : op->getUsers()) {
        if (auto pprOp = llvm::dyn_cast<PPRotationOp>(userOp)) {
            if (verifyNextNonClifford(op, pprOp)) {
                return callback(pprOp);
            }
        }
    }

    return failure();
}

void updatePauliProduct(PPRotationOp rhsOp, PauliStringWrapper pauli, PatternRewriter &rewriter)
{
    auto pauliProductArray = pauli.get_pauli_words();
    SmallVector<StringRef> pauliProductArrayAttr(pauliProductArray.begin(),
                                                 pauliProductArray.end());
    auto pauliProduct = rewriter.getStrArrayAttr(pauliProductArrayAttr);
    rhsOp.setPauliProductAttr(pauliProduct);

    // Handle rotation kind separately
    // Instead, use the negated flag directly
    int16_t rotationKind = static_cast<int16_t>(rhsOp.getRotationKind());
    int16_t sign = pauli.negated ? -1 : 1;

    // Preserve the absolute value but apply the correct sign
    rotationKind = (rotationKind < 0 ? -rotationKind : rotationKind) * sign;
    rhsOp.setRotationKindAttr(rewriter.getI16IntegerAttr(rotationKind));
}

void replaceIdentityPauli(PauliStringWrapper &rhsPauli, PauliStringWrapper &lhsPauli)
{
    auto rhsPauliStr = rhsPauli.str();
    auto lhsPauliStr = lhsPauli.str();
    for (unsigned i = 0; i < rhsPauliStr.size(); i++) {
        if (rhsPauliStr[i] == 'I' || rhsPauliStr[i] == '_') {
            rhsPauliStr[i] = lhsPauliStr[i];
        }
    }
    rhsPauli.pauliString = stim::PauliString<MAX_BITWORD>::from_str(rhsPauliStr.c_str());
}

SmallVector<Value> fullfillOperands(PauliStringWrapper lhsPauliWrapper,
                                    PauliStringWrapper rhsPauliWrapper)
{
    auto lhs = *lhsPauliWrapper.op;
    auto rhs = *rhsPauliWrapper.op;
    auto rhsPauliSize = rhsPauliWrapper.correspondingQubits.size();

    // Fullfill Operands of RHS
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

SmallVector<StringRef> removeIdentityPauli(PPRotationOp rhs, SmallVector<Value> &newRHSOperands)
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

void moveCliffordPastNonClifford(PauliStringWrapper lhsPauli, PauliStringWrapper rhsPauli,
                                 PauliStringWrapper *result, PatternRewriter &rewriter)
{
    assert(lhsPauli.op != nullptr && "LHS Operation is not found");
    assert(rhsPauli.op != nullptr && "RHS Operation is not found");

    auto lhs = *lhsPauli.op;
    auto rhs = *rhsPauli.op;

    assert(!isNonClifford(lhs) && "LHS Operation is not Clifford");
    assert(isNonClifford(rhs) && "RHS Operation is not non-Clifford");
    assert(lhs.getPauliProduct().size() == lhs.getOutQubits().size() &&
           "LHS Pauli product size mismatch before commutation.");
    assert(rhs.getPauliProduct().size() == rhs.getInQubits().size() &&
           "RHS Pauli product size mismatch before commutation.");

    // Update Pauli words of RHS
    if (result != nullptr) {
        updatePauliProduct(rhs, *result, rewriter);
    }
    else {
        replaceIdentityPauli(rhsPauli, lhsPauli);
        updatePauliProduct(rhs, rhsPauli, rewriter);
    }

    // Fullfill Operands of RHS
    SmallVector<Value> newRHSOperands = fullfillOperands(lhsPauli, rhsPauli);

    // Remove the Identity gate in the Pauli product
    SmallVector<StringRef> pauliProductArrayRef = removeIdentityPauli(rhs, newRHSOperands);
    mlir::ArrayAttr pauliProduct = rewriter.getStrArrayAttr(pauliProductArrayRef);

    // Get the type list from new RHS
    SmallVector<Type> newOutQubitsTypesList;
    for (auto qubit : newRHSOperands) {
        newOutQubitsTypesList.push_back(qubit.getType());
    }

    // Create the new PPR
    auto nonCliffordOp =
        rewriter.create<PPRotationOp>(rhs->getLoc(), newOutQubitsTypesList, pauliProduct,
                                      rhs.getRotationKindAttr(), newRHSOperands);
    rewriter.moveOpBefore(nonCliffordOp, rhs);

    // Update the use of value in newRHSOperands
    for (unsigned i = 0; i < newRHSOperands.size(); i++) {
        newRHSOperands[i].replaceAllUsesExcept(nonCliffordOp.getOutQubits()[i], nonCliffordOp);
    }

    rewriter.replaceOp(rhs, rhs.getInQubits());
}

struct CommuteCliffordTPPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        return visitPPRotationOp(op, [&](PPRotationOp nextPPROp) {
            PauliWordsPair normOps = normalizePPROps(op, nextPPROp);

            if (isCommute(normOps.first, normOps.second)) {
                moveCliffordPastNonClifford(normOps.first, normOps.second, nullptr, rewriter);
            }else {
                auto resultStr = computeCommutationRules(normOps.first, normOps.second);
                moveCliffordPastNonClifford(normOps.first, normOps.second, &resultStr, rewriter);
            }

            sortTopologically(op->getBlock());
            return success();
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
