// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This analysis checks if a gate operation and its parent gate operation
// are correctly matched for the purposes of peephole optimizations like
// merge rotations and cancel inverses.
// Gates passing this analysis are considered valid candidates for merge
// rotation and cancel inverses.

// Specifically, we check the following conditions in VerifyParentGateAnalysis:
//  1. Both gates must be of the same type, i.e. a quantum.custom can
//     only be cancelled with a quantum.custom, not a quantum.unitary
//  2. The results of the parent gate must map one-to-one, in order,
//     to the operands of the second gate
//     For example, the pair
//        %0:2 = quantum.custom "CNOT"() %.., %..
//        %1:2 = quantum.custom "CNOT"() %0#0, %0#1
//     is considered to be a successful match, but the pair
//        %0:2 = quantum.custom "CNOT"() %.., %..
//        %1:2 = quantum.custom "CNOT"() %0#1, %0#0
//     is not.
//  3. If the gates are controlled, both gates' control wires and values
//     must be the same. The control wires must be in the same order
//
//  On top of the above, we also provide a VerifyParentGateAndNameAnalysis,
//  which also checks:
//  4. If the gates are quantum.custom, then both gates have the same name.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {

template <typename OpType> class VerifyParentGateAnalysis {
  public:
    VerifyParentGateAnalysis(OpType gate)
    {
        ValueRange inQubits = gate.getInQubits();
        auto parentGate = dyn_cast_or_null<OpType>(inQubits[0].getDefiningOp());

        if (!verifyParentGateType(gate, parentGate)) {
            setVerifierResult(false);
            return;
        }

        if (!verifyAllInQubits(gate, parentGate)) {
            setVerifierResult(false);
            return;
        }
    }

    bool getVerifierResult() { return succeeded; }

    void setVerifierResult(bool b) { succeeded = b; }

  private:
    bool succeeded = true;

    bool verifyParentGateType(OpType op, OpType parentOp) const
    {
        // Verify that the parent gate is of the same type.
        // If OpType is quantum.custom, also verify that parent gate has the
        // same gate name.

        if (!parentOp || !isa<OpType>(parentOp)) {
            return false;
        }

        return true;
    }

    bool verifyAllInQubits(OpType op, OpType parentOp) const
    {
        // Verify that parent's results and current gate's inputs are in the same order
        // If the gates are controlled, both gates' control wires and values
        // must be the same. The control wires must be in the same order.

        ValueRange inNonCtrlQubits = op.getNonCtrlQubitOperands();
        ValueRange inCtrlQubits = op.getCtrlQubitOperands();
        ValueRange parentOutNonCtrlQubits = parentOp.getNonCtrlQubitResults();
        ValueRange parentOutCtrlQubits = parentOp.getCtrlQubitResults();

        if ((inNonCtrlQubits.size() != parentOutNonCtrlQubits.size()) ||
            (inCtrlQubits.size() != parentOutCtrlQubits.size())) {
            return false;
        }

        for (const auto &[idx, qubit] : llvm::enumerate(inNonCtrlQubits)) {
            if (qubit.getDefiningOp() != parentOp || qubit != parentOutNonCtrlQubits[idx]) {
                return false;
            }
        }

        for (const auto &[idx, qubit] : llvm::enumerate(inCtrlQubits)) {
            if (qubit.getDefiningOp() != parentOp || qubit != parentOutCtrlQubits[idx]) {
                return false;
            }
        }

        ValueRange opCtrlValues = op.getCtrlValueOperands();
        ValueRange parentCtrlValues = parentOp.getCtrlValueOperands();
        if (opCtrlValues.size() != parentCtrlValues.size()) {
            return false;
        }

        for (const auto &[idx, v] : llvm::enumerate(opCtrlValues)) {
            // We assume CSE is already run before this analysis.
            if (v != parentCtrlValues[idx]) {
                return false;
            }
        }

        return true;
    }
};

class VerifyParentGateAndNameAnalysis : public VerifyParentGateAnalysis<quantum::CustomOp> {
    // If OpType is quantum.custom, also verify that parent gate has the
    // same gate name.
  public:
    VerifyParentGateAndNameAnalysis(quantum::CustomOp gate)
        : VerifyParentGateAnalysis<quantum::CustomOp>(gate)
    {
        ValueRange inQubits = gate.getInQubits();
        auto parentGate = dyn_cast_or_null<quantum::CustomOp>(inQubits[0].getDefiningOp());

        if (!parentGate) {
            this->setVerifierResult(false);
            return;
        }

        if (!verifyParentGateName(gate, parentGate)) {
            this->setVerifierResult(false);
            return;
        }
    }

  private:
    bool verifyParentGateName(quantum::CustomOp op, quantum::CustomOp parentOp) const
    {
        StringRef opGateName = op.getGateName();
        StringRef parentGateName = parentOp.getGateName();
        return opGateName == parentGateName;
    }
};
} // namespace catalyst
