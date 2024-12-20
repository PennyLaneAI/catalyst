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

// This algorithm is taken from https://arxiv.org/pdf/2012.07711, figure 5

#pragma once

#include <map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {

// The six Pauli eigenstates
enum class QubitState {
    ZERO,
    ONE,
    PLUS,
    MINUS,
    LEFT,
    RIGHT,
    NOT_A_BASIS,
};

// {input state : {gate, output state}}
static std::map<QubitState, std::map<StringRef, QubitState>> QubitTransitions = {
    {QubitState::ZERO,
     {
         {"Hadamard", QubitState::PLUS},
         {"PauliX", QubitState::ONE},
         {"PauliY", QubitState::ONE},
         {"PauliZ", QubitState::ZERO},
     }},

    {QubitState::ONE,
     {
         {"Hadamard", QubitState::MINUS},
         {"PauliX", QubitState::ZERO},
         {"PauliY", QubitState::ZERO},
         {"PauliZ", QubitState::ONE},
     }},

    {QubitState::PLUS,
     {
         {"Hadamard", QubitState::ZERO},
         {"PauliX", QubitState::PLUS},
         {"PauliY", QubitState::MINUS},
         {"PauliZ", QubitState::MINUS},
         {"S", QubitState::LEFT},
     }},

    {QubitState::MINUS,
     {
         {"Hadamard", QubitState::ONE},
         {"PauliX", QubitState::MINUS},
         {"PauliY", QubitState::PLUS},
         {"PauliZ", QubitState::PLUS},
         {"S", QubitState::RIGHT},
     }},

    {QubitState::LEFT,
     {
         {"Hadamard", QubitState::RIGHT},
         {"PauliX", QubitState::RIGHT},
         {"PauliY", QubitState::LEFT},
         {"PauliZ", QubitState::RIGHT},
         // We leave in S+ to indicate the FSM structure
         // The actual implementation is `quantum.custom "S"() %in {adjoint}`
         //{"S+", QubitState::PLUS},
     }},

    {QubitState::RIGHT,
     {
         {"Hadamard", QubitState::LEFT},
         {"PauliX", QubitState::LEFT},
         {"PauliY", QubitState::RIGHT},
         {"PauliZ", QubitState::LEFT},
         // We leave in S+ to indicate the FSM structure
         // The actual implementation is `quantum.custom "S"() %in {adjoint}`
         //{"S+", QubitState::MINUS},
     }},
};

class PropagateSimpleStatesAnalysis {
  public:
    PropagateSimpleStatesAnalysis(Operation *target)
    {
        // `target` is a qnode function
        // We restrict the analysis to gates at the top-level body of the function
        // This is so that gates inside nested regions, like control flows, are not valid targets
        // e.g.
        //   func.func circuit() {
        //      %0 = |0>
        //      %1 = scf.if (condition) {
        //          true:  Hadamard
        //          false: PauliZ
        //      }
        //   }
        // since depending on the control flow path, %1 can be in multiple different states

        // Two kinds of operations produce qubit values: extract ops and custom ops
        // For extract ops, if its operand is a alloc op directly, then it's a starting qubit and is
        // in |0>. Then the FSM propagates the states through the custom op gates

        target->walk([&](quantum::ExtractOp op) {
            // Do not analyze any operations in invalid ops or regions.
            // The only valid ops are the custom/extract ops whose immediate parent is the qnode
            // function. With this, we skip over regions like control flow.
            if (!isImmediateChild(op, target)) {
                return;
            }

            // starting qubits are in |0>
            // We restrict "starting qubits" to those who are extracted immediately from alloc ops
            if (isa<quantum::AllocOp>(op.getQreg().getDefiningOp())) {
                qubitValues[op.getQubit()] = QubitState::ZERO;
                return;
            }
        });

        target->walk([&](quantum::CustomOp op) {
            if (!isImmediateChild(op, target)) {
                return;
            }

            if (op->getNumResults() != 1) {
                // restrict to single-qubit gates
                return;
            }

            Value res = op->getResult(0);

            // takes in parameters other than the parent qubit
            // e.g. the rotation angle
            // must be NOT_A_BASIS!
            if (op->getNumOperands() != 1) {
                qubitValues[res] = QubitState::NOT_A_BASIS;
                return;
            }

            // get state from parent and gate
            StringRef gate = cast<quantum::CustomOp>(op).getGateName();
            Value parent = op->getOperand(0);

            // Unknown parent state, child state is thus also unknown
            if (!qubitValues.contains(parent) || isOther(qubitValues[parent])) {
                qubitValues[res] = QubitState::NOT_A_BASIS;
                return;
            }

            // Identity preserves parent state
            if (gate == "Identity") {
                qubitValues[res] = qubitValues[parent];
                return;
            }

            // A valid FSM transition gate
            // Special treatment for S+ gate from |L> and |R>
            if ((isLeft(qubitValues[parent]) || isRight(qubitValues[parent])) && gate == "S") {
                if (op->hasAttr("adjoint")) {
                    switch (qubitValues[parent]) {
                    case QubitState::LEFT:
                        qubitValues[res] = QubitState::PLUS;
                        break;
                    case QubitState::RIGHT:
                        qubitValues[res] = QubitState::MINUS;
                        break;
                    default:
                        // this will never trigger as the switch is inside an if
                        break;
                    }
                }
                else {
                    qubitValues[res] = QubitState::NOT_A_BASIS;
                }
                return;
            }

            // A valid FSM transition gate
            if (QubitTransitions[qubitValues[parent]].count(gate) == 1) {
                qubitValues[res] = QubitTransitions[qubitValues[parent]][gate];
            }
            // Not a valid FSM transition gate
            else {
                qubitValues[res] = QubitState::NOT_A_BASIS;
            }
            return;
        });
    }

    llvm::DenseMap<Value, QubitState> getQubitValues() { return qubitValues; }

    // Function to convert enum values to strings
    static std::string QubitState2String(QubitState state)
    {
        switch (state) {
        case QubitState::ZERO:
            return "ZERO";
        case QubitState::ONE:
            return "ONE";
        case QubitState::PLUS:
            return "PLUS";
        case QubitState::MINUS:
            return "MINUS";
        case QubitState::LEFT:
            return "LEFT";
        case QubitState::RIGHT:
            return "RIGHT";
        case QubitState::NOT_A_BASIS:
            return "NOT_A_BASIS";
        }
    }

    bool isZero(QubitState qs) { return qs == QubitState::ZERO; }

    bool isOne(QubitState qs) { return qs == QubitState::ONE; }

    bool isPlus(QubitState qs) { return qs == QubitState::PLUS; }

    bool isMinus(QubitState qs) { return qs == QubitState::MINUS; }

    bool isLeft(QubitState qs) { return qs == QubitState::LEFT; }

    bool isRight(QubitState qs) { return qs == QubitState::RIGHT; }

    bool isOther(QubitState qs) { return qs == QubitState::NOT_A_BASIS; }

  private:
    // The object `qubitValues` contains all the analysis results
    // It is a map of the form
    // <mlir Value representing a qubit, its abstract QubitState>
    llvm::DenseMap<Value, QubitState> qubitValues;

    bool isImmediateChild(Operation *op, Operation *ancestor)
    {
        // returns true if op is an immediate child of ancestor,
        // with no extra operations in between
        return op->getParentOp() == ancestor;
    }
};

} // namespace catalyst
