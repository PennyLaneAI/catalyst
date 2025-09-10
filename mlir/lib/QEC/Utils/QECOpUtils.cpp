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

#include "QEC/Utils/QECOpUtils.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/Utils/PauliStringWrapper.h"

namespace catalyst {
namespace qec {

mlir::Value getReachingValueAt(mlir::Value qubit, QECOpInterface op)
{
    // We want to find the qubit that is used by op.
    // e.g., op is first PPR, while qubit can be the operands from the third PPR.
    // %0_q0, %0_q1 = qec.ppr ["X", "X"](8) %arg0, %arg1 // <- op
    // %1_q0, %1_q1 = qec.ppr ["Z", "X"](8) %0_q0, %0_q1
    // %2_q0, %2_q1 = qec.ppr ["X", "X"](8) %1_q0, %1_q1
    //                                        ^- qubit that is used by op
    // So if qubit is %1_q0, we want to return %arg0.

    assert(qubit != nullptr && "Qubit should not be nullptr");

    auto defOp = qubit.getDefiningOp();

    if (!defOp || defOp->isBeforeInBlock(op)) {
        return qubit;
    }

    auto qecOp = llvm::dyn_cast<QECOpInterface>(defOp);

    if (!qecOp) {
        return nullptr;
    }

    auto outQubits = qecOp.getOutQubits();
    auto inQubits = qecOp.getInQubits();
    assert((inQubits.size() == outQubits.size()) &&
           "PPR op should have the same number of input and output qubits");

    auto pos = std::distance(outQubits.begin(), llvm::find(outQubits, qubit));
    mlir::Value inQubit = inQubits[pos];

    if (qecOp == op) {
        return inQubit;
    }

    return getReachingValueAt(inQubit, op);
}

std::vector<mlir::Value> getInQubitReachingValuesAt(QECOpInterface srcOp, QECOpInterface dstOp)
{
    std::vector<mlir::Value> dominanceQubits;
    dominanceQubits.reserve(srcOp.getInQubits().size());
    for (auto inQubit : srcOp.getInQubits()) {
        mlir::Value v = getReachingValueAt(inQubit, dstOp);
        dominanceQubits.emplace_back(v);
    }
    return dominanceQubits;
}

bool commutes(QECOpInterface rhsOp, QECOpInterface lhsOp)
{
    if (lhsOp->getBlock() != rhsOp->getBlock()) {
        return false;
    }

    assert(lhsOp != rhsOp && "lshOp and rhsOp should not be equal");
    assert(lhsOp->isBeforeInBlock(rhsOp) && "lhsOp should be before rhsOp");

    // Reaching in-qubit values of `rhsOp` at the program point of `lhsOp`.
    std::vector<mlir::Value> rhsOpInQubitsFromLhs = getInQubitReachingValuesAt(rhsOp, lhsOp);

    // If any of the in-qubit values are nullptr, the ops do not commute.
    if (llvm::any_of(rhsOpInQubitsFromLhs, [](mlir::Value v) { return v == nullptr; })) {
        return false;
    }

    // Normalize the ops to the same Pauli string.
    auto normalizedOps = normalizePPROps(lhsOp, rhsOp, lhsOp.getInQubits(), rhsOpInQubitsFromLhs);

    // If the normalized ops do not commute, the original ops do not commute.
    if (!normalizedOps.first.commutes(normalizedOps.second)) {
        return false;
    }

    return true;
}

} // namespace qec
} // namespace catalyst
