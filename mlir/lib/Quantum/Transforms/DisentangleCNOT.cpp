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

// This algorithm is taken from https://arxiv.org/pdf/2012.07711, table 1

#define DEBUG_TYPE "disentanglecnot"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

#include "PropagateSimpleStatesAnalysis.hpp"

using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_DISENTANGLECNOTPASS
#define GEN_PASS_DECL_DISENTANGLECNOTPASS
#include "Quantum/Transforms/Passes.h.inc"

struct DisentangleCNOTPass : public impl::DisentangleCNOTPassBase<DisentangleCNOTPass> {
    using impl::DisentangleCNOTPassBase<DisentangleCNOTPass>::DisentangleCNOTPassBase;

    void runOnOperation() override
    {
        auto op = getOperation();
        SmallVector<Operation *> targets;
        op->walk([&](FunctionOpInterface func) { targets.push_back(func); });

        for (auto func : targets) {
            mlir::IRRewriter builder(func->getContext());
            Location loc = func->getLoc();

            PropagateSimpleStatesAnalysis pssa(func);
            llvm::DenseMap<Value, QubitState> qubitValues = pssa.getQubitValues();

            if (EmitFSMStateRemark) {
                for (auto it = qubitValues.begin(); it != qubitValues.end(); ++it) {
                    it->first.getDefiningOp()->emitRemark(pssa.QubitState2String(it->second));
                }
            }

            func->walk([&](quantum::CustomOp op) {
                StringRef gate = op.getGateName();
                if (gate != "CNOT") {
                    return;
                }

                Value controlIn = op->getOperand(0);
                Value targetIn = op->getOperand(1);
                Value controlOut = op->getResult(0);
                Value targetOut = op->getResult(1);

                // Do nothing if the inputs states are not tracked
                if (!qubitValues.contains(controlIn) || !qubitValues.contains(targetIn)) {
                    return;
                }

                // |0> control, always do nothing
                if (pssa.isZero(qubitValues[controlIn])) {
                    controlOut.replaceAllUsesWith(controlIn);
                    targetOut.replaceAllUsesWith(targetIn);
                    op->erase();
                    return;
                }

                // |1> control, insert PauliX gate on target
                if (pssa.isOne(qubitValues[controlIn])) {
                    controlOut.replaceAllUsesWith(controlIn);

                    // PauliX on |+-> is unnecessary: they are eigenstates!
                    if ((pssa.isPlus(qubitValues[targetIn])) ||
                        (pssa.isMinus(qubitValues[targetIn]))) {
                        targetOut.replaceAllUsesWith(targetIn);
                        op->erase();
                        return;
                    }
                    else {
                        builder.setInsertionPoint(op);
                        quantum::CustomOp xgate = builder.create<quantum::CustomOp>(
                            loc, /*gate_name=*/"PauliX",
                            /*in_qubits=*/mlir::ValueRange({targetIn}));
                        targetOut.replaceAllUsesWith(xgate->getResult(0));
                        op->erase();
                        return;
                    }
                }

                // |+> target, always do nothing
                if (pssa.isPlus(qubitValues[targetIn])) {
                    controlOut.replaceAllUsesWith(controlIn);
                    targetOut.replaceAllUsesWith(targetIn);
                    op->erase();
                    return;
                }

                // |-> target, insert PauliZ on control
                if (pssa.isMinus(qubitValues[targetIn])) {
                    targetOut.replaceAllUsesWith(targetIn);

                    // PauliZ on |01> is unnecessary: they are eigenstates!
                    if ((pssa.isZero(qubitValues[controlIn])) ||
                        (pssa.isOne(qubitValues[controlIn]))) {
                        controlOut.replaceAllUsesWith(controlIn);
                        op->erase();
                        return;
                    }
                    else {
                        builder.setInsertionPoint(op);
                        quantum::CustomOp zgate = builder.create<quantum::CustomOp>(
                            loc, /*gate_name=*/"PauliZ",
                            /*in_qubits=*/mlir::ValueRange({controlIn}));
                        controlOut.replaceAllUsesWith(zgate->getResult(0));
                        op->erase();
                        return;
                    }
                }
            });
        }
    }
};

std::unique_ptr<Pass> createDisentangleCNOTPass()
{
    return std::make_unique<DisentangleCNOTPass>();
}

} // namespace catalyst
