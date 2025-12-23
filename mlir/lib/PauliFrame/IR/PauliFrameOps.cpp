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

#include "mlir/IR/Builders.h"

#include "PauliFrame/IR/PauliFrameDialect.h"
#include "PauliFrame/IR/PauliFrameOps.h"

using namespace mlir;
using namespace catalyst::pauli_frame;

//===----------------------------------------------------------------------===//
// PauliFrame op definitions.
//===----------------------------------------------------------------------===//

#include "PauliFrame/IR/PauliFrameEnums.cpp.inc"

#define GET_OP_CLASSES
#include "PauliFrame/IR/PauliFrameOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PauliFrame op verifiers.
//===----------------------------------------------------------------------===//

namespace catalyst::pauli_frame {
/**
 * @brief Verifies the number of input and output qubits for PauliFrame operations
 *
 * @param op The PauliFrame op to verify
 * @param inQubits The OperandRange containing the input qubits of the op
 * @param outQubits The ResultRange containing the output qubits of the op
 * @return LogicalResult Success if verified, opError otherwise
 */
LogicalResult verifyQubitCounts(Operation *op, const OperandRange &inQubits,
                                const ResultRange &outQubits)
{
    if (inQubits.size() == 0) {
        return op->emitOpError("expected to have at least one qubit");
    }
    if (inQubits.size() != outQubits.size()) {
        return op->emitOpError("expected to consume and return the same number of qubits");
    }
    return success();
}
} // namespace catalyst::pauli_frame

LogicalResult InitOp::verify() { return verifyQubitCounts(*this, getInQubits(), getOutQubits()); }

LogicalResult UpdateOp::verify() { return verifyQubitCounts(*this, getInQubits(), getOutQubits()); }

LogicalResult UpdateWithCliffordOp::verify()
{
    LogicalResult result = verifyQubitCounts(*this, getInQubits(), getOutQubits());

    if (!result.succeeded()) {
        return result;
    }

    switch (getCliffordGate()) {
    case CliffordGate::Hadamard:
        if (getInQubits().size() != 1) {
            return emitOpError("expected exactly one input qubit for Clifford gate 'Hadamard'");
        }
        break;

    case CliffordGate::S:
        if (getInQubits().size() != 1) {
            return emitOpError("expected exactly one input qubit for Clifford gate 'S'");
        }
        break;

    case CliffordGate::CNOT:
        if (getInQubits().size() != 2) {
            return emitOpError("expected exactly two input qubits for Clifford gate 'CNOT'");
        }
        break;
    }
    return success();
}

LogicalResult SetOp::verify() { return verifyQubitCounts(*this, getInQubits(), getOutQubits()); }
