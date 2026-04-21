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

#include "PBC/Utils/PBCOpUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/Utils/PauliStringWrapper.h"

namespace catalyst {
namespace pbc {

mlir::Value getReachingValueAt(mlir::Value qubit, PBCOpInterface op)
{
    // We want to find the qubit that is used by op.
    // e.g., op is first PPR, while qubit can be the operands from the third PPR.
    // %0_q0, %0_q1 = pbc.ppr ["X", "X"](8) %arg0, %arg1 // <- op
    // %1_q0, %1_q1 = pbc.ppr ["Z", "X"](8) %0_q0, %0_q1
    // %2_q0, %2_q1 = pbc.ppr ["X", "X"](8) %1_q0, %1_q1
    //                                        ^- qubit that is used by op
    // So if qubit is %1_q0, we want to return %arg0.

    assert(qubit != nullptr && "Qubit should not be nullptr");

    auto defOp = qubit.getDefiningOp();

    if (!defOp || defOp->isBeforeInBlock(op)) {
        return qubit;
    }

    auto pbcOp = llvm::dyn_cast<PBCOpInterface>(defOp);

    if (!pbcOp) {
        return nullptr;
    }

    auto outQubits = pbcOp.getOutQubits();
    auto inQubits = pbcOp.getInQubits();
    assert((inQubits.size() == outQubits.size()) &&
           "PPR op should have the same number of input and output qubits");

    auto pos = std::distance(outQubits.begin(), llvm::find(outQubits, qubit));
    mlir::Value inQubit = inQubits[pos];

    if (pbcOp == op) {
        return inQubit;
    }

    return getReachingValueAt(inQubit, op);
}

std::vector<mlir::Value> getInQubitReachingValuesAt(PBCOpInterface srcOp, PBCOpInterface dstOp)
{
    std::vector<mlir::Value> dominanceQubits;
    dominanceQubits.reserve(srcOp.getInQubits().size());
    for (auto inQubit : srcOp.getInQubits()) {
        mlir::Value v = getReachingValueAt(inQubit, dstOp);
        dominanceQubits.emplace_back(v);
    }
    return dominanceQubits;
}

bool commutes(PBCOpInterface rhsOp, PBCOpInterface lhsOp)
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

} // namespace pbc
} // namespace catalyst
