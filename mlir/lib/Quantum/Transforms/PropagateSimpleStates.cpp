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

#define DEBUG_TYPE "propagatesimplestates"

#include <map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_PROPAGATESIMPLESTATESPASS
#define GEN_PASS_DECL_PROPAGATESIMPLESTATESPASS
#include "Quantum/Transforms/Passes.h.inc"

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

struct PropagateSimpleStatesPass
    : public impl::PropagateSimpleStatesPassBase<PropagateSimpleStatesPass> {
    using impl::PropagateSimpleStatesPassBase<
        PropagateSimpleStatesPass>::PropagateSimpleStatesPassBase;

    void runOnOperation() override
    {
        LLVM_DEBUG(dbgs() << "propagate simple states pass"
                          << "\n");

        Operation *module = getOperation();
        Operation *targetfunc;

        WalkResult result = module->walk([&](func::FuncOp op) {
            StringRef funcName = op.getSymName();

            if (funcName != FuncNameOpt) {
                // not the function to run the pass on, visit the next function
                return WalkResult::advance();
            }
            targetfunc = op;
            return WalkResult::interrupt();
        });

        if (!result.wasInterrupted()) {
            // Never met a target function
            // Do nothing and exit!
            return;
        }

        ///////////////////////////

        llvm::DenseMap<Value, QubitState> qubitValues;
        targetfunc->walk([&](Operation *op) {
            if (op->getNumResults() != 1) {
                // restrict to single-qubit gates
                return;
            }

            Value res = op->getResult(0);
            if (!isa<quantum::QubitType>(res.getType())) {
                // not a qubit value
                return;
            }

            // starting qubits are in |0>
            if (isa<quantum::ExtractOp>(op)) {
                qubitValues[res] = QubitState::ZERO;
                return;
            }

            assert(isa<quantum::CustomOp>(op));

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
            assert(qubitValues.contains(parent));

            // non basis states stay as non basis states
            if (qubitValues[parent] == QubitState::NOT_A_BASIS) {
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
            if (((qubitValues[parent] == QubitState::LEFT) ||
                 (qubitValues[parent] == QubitState::RIGHT)) &&
                (gate == "S")) {
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

        // The object `qubitValues` contains all the analysis results
        // We emit them as operation remarks for testing
        for (auto it = qubitValues.begin(); it != qubitValues.end(); ++it) {
            it->first.getDefiningOp()->emitRemark(QubitState2String(it->second));
        }
    }
};

std::unique_ptr<Pass> createPropagateSimpleStatesPass()
{
    return std::make_unique<PropagateSimpleStatesPass>();
}

} // namespace catalyst
