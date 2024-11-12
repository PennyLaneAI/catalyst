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

// This algorithm is taken from https://arxiv.org/pdf/2012.07711, table 6 (Equivalences for basis-states in SWAP gate)

#define DEBUG_TYPE "disentangleswap"

#include "PropagateSimpleStatesAnalysis.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_DISENTANGLESWAPPASS
#define GEN_PASS_DECL_DISENTANGLESWAPPASS
#include "Quantum/Transforms/Passes.h.inc"

struct DisentangleSWAPPass : public impl::DisentangleSWAPPassBase<DisentangleSWAPPass> {
    using impl::DisentangleSWAPPassBase<DisentangleSWAPPass>::DisentangleSWAPPassBase;

    quantum::CustomOp createSimpleOneBitGate(StringRef gateName, const Value &inQubit,
                                             const Value &outQubit, mlir::IRRewriter &builder,
                                             Location &loc, const quantum::CustomOp &insert_after_gate)
    {
        OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPointAfter(insert_after_gate);
        quantum::CustomOp newGate =
            builder.create<quantum::CustomOp>(loc,
                                              /*out_qubits=*/mlir::TypeRange({outQubit.getType()}),
                                              /*out_ctrl_qubits=*/mlir::TypeRange(),
                                              /*params=*/mlir::ValueRange(),
                                              /*in_qubits=*/mlir::ValueRange({inQubit}),
                                              /*gate_name=*/gateName,
                                              /*adjoint=*/nullptr,
                                              /*in_ctrl_qubits=*/mlir::ValueRange(),
                                              /*in_ctrl_values=*/mlir::ValueRange());
    
        return newGate;
    }


    // quantum::CustomOp createSimpleTwoBitGate(StringRef gateName, 
    //                                          const Value &controlIn, const Value &TargetIn, 
    //                                          const Value &controlOut, const Value &targetOut, 
    //                                          mlir::IRRewriter &builder, Location &loc, 
    //                                          const quantum::CustomOp &insert_after_gate)
    // {
    //     OpBuilder::InsertionGuard insertionGuard(builder);
    //     builder.setInsertionPointAfter(insert_after_gate);
    //     quantum::CustomOp newGate =
    //         builder.create<quantum::CustomOp>(loc,
    //                                           /*out_qubits=*/mlir::TypeRange({targetOut.getType()}),
    //                                           /*out_ctrl_qubits=*/mlir::TypeRange({controlOut.getType()}),
    //                                           /*params=*/mlir::ValueRange(),
    //                                           /*in_qubits=*/mlir::ValueRange({TargetIn}),
    //                                           /*gate_name=*/gateName,
    //                                           /*adjoint=*/nullptr,
    //                                           /*in_ctrl_qubits=*/mlir::ValueRange({controlIn}),
    //                                           /*in_ctrl_values=*/mlir::ValueRange());

    //     return newGate;
    // }

    bool canScheduleOn(RegisteredOperationName opInfo) const override
    {
        return opInfo.hasInterface<FunctionOpInterface>();
    }

    void runOnOperation() override
    {
        LLVM_DEBUG(dbgs() << "disentangle SWAP pass\n");

        func::FuncOp func = cast<func::FuncOp>(getOperation());
        mlir::IRRewriter builder(func->getContext());
        Location loc = func->getLoc();

        if (func.getSymName() != FuncNameOpt) {
            // not the function to run the pass on
            return;
        }

        ///////////////////////////

        PropagateSimpleStatesAnalysis &pssa = getAnalysis<PropagateSimpleStatesAnalysis>();
        llvm::DenseMap<Value, QubitState> qubitValues = pssa.getQubitValues();

        func->walk([&](quantum::CustomOp op) {

            // PropagateSimpleStatesAnalysis pssa = PropagateSimpleStatesAnalysis(op);
            // llvm::DenseMap<Value, QubitState> qubitValues = pssa.getQubitValues();

            StringRef gate = op.getGateName();
            if (gate != "SWAP") {
                return;
            }

            Value SwapQubit_0_In = op->getOperand(0);
            Value SwapQubit_1_In = op->getOperand(1);
            Value SwapQubit_0_Out = op->getResult(0);
            Value SwapQubit_1_Out = op->getResult(1);

            // first qubit in |0> 
            if (pssa.isZero(qubitValues[SwapQubit_0_In])) {

                // second qubit in |0>
                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP 0,0\n"; 
                    SwapQubit_0_Out.replaceAllUsesWith(SwapQubit_0_In);
                    SwapQubit_1_Out.replaceAllUsesWith(SwapQubit_1_In);
                    op->erase();
                    return;
                }

                // second qubit in |1>
                else if (pssa.isOne(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP 0,1\n";
                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("PauliX", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, xgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));
                    op->erase();
                    return;
                }

                // second qubit in |+>
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP 0,+\n";
                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }

                // second qubit in |->
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP 0,-\n";
                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", xgate_on_0->getOperand(0), xgate_on_0->getResult(0), builder, loc, xgate_on_0);
                    (xgate_on_0->getResult(0)).replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, op);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));

                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("Pauli", hgate_on_1->getOperand(0), hgate_on_1->getResult(0), builder, loc, hgate_on_1);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));

                    op->erase();
                    return;
                }
            }

            // first qubit in |1> 
            else if (pssa.isOne(qubitValues[SwapQubit_0_In])) {
                // second qubit in |1>
                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP 1,0\n";
                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("PauliX", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, xgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                else if (pssa.isOne(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP 1,1\n";
                    SwapQubit_0_Out.replaceAllUsesWith(SwapQubit_0_In);
                    SwapQubit_1_Out.replaceAllUsesWith(SwapQubit_1_In);
                    op->erase();
                    return;
                }
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP 1,+\n";
                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", xgate_on_0->getOperand(0), xgate_on_0->getResult(0), builder, loc, xgate_on_0);
                    (xgate_on_0->getResult(0)).replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));

                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("Pauli", hgate_on_1->getOperand(0), hgate_on_1->getResult(0), builder, loc, hgate_on_1);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));

                    op->erase();
                    return;
                }
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP 1,-\n";
                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
            }

            // first qubit in |+> 
            else if (pssa.isPlus(qubitValues[SwapQubit_0_In])) {

                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP +,0\n";
                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                else if (pssa.isOne(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP +,1\n";
                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", hgate_on_0->getOperand(0), hgate_on_0->getResult(0), builder, loc, hgate_on_0);
                    (xgate_on_0->getResult(0)).replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("PauliX", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));

                    quantum::CustomOp hgate_on_1 =
                        createSimpleOneBitGate("Hadamard", xgate_on_1->getOperand(0), xgate_on_1->getResult(0), builder, loc, xgate_on_1);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));

                    op->erase();
                    return;
                }
                // second qubit in |+>
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP +,+\n";
                    SwapQubit_0_Out.replaceAllUsesWith(SwapQubit_0_In);
                    SwapQubit_1_Out.replaceAllUsesWith(SwapQubit_1_In);
                    op->erase();
                    return;
                }
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP +,-\n";
                    quantum::CustomOp zgate_on_0 =
                        createSimpleOneBitGate("PauliZ", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(zgate_on_0->getResult(0));

                    quantum::CustomOp zgate_on_1 =
                        createSimpleOneBitGate("PauliZ", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, zgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(zgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
            }

            // first qubit in |-> 
            else if (pssa.isMinus(qubitValues[SwapQubit_0_In])) {
                // second qubit in |->
                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP -,0\n";
                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", hgate_on_0->getOperand(0), hgate_on_0->getResult(0), builder, loc, hgate_on_0);
                    (xgate_on_0->getResult(0)).replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("PauliX", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));

                    quantum::CustomOp hgate_on_1 =
                        createSimpleOneBitGate("Hadamard", xgate_on_1->getOperand(0), xgate_on_1->getResult(0), builder, loc, xgate_on_1);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));

                    op->erase();
                    return;
                }
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP -,1\n";
                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP -,+\n";
                    quantum::CustomOp zgate_on_0 =
                        createSimpleOneBitGate("PauliZ", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(zgate_on_0->getResult(0));

                    quantum::CustomOp zgate_on_1 =
                        createSimpleOneBitGate("PauliZ", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, zgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(zgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    llvm::errs() << "SWAP -,-\n";
                    SwapQubit_0_Out.replaceAllUsesWith(SwapQubit_0_In);
                    SwapQubit_1_Out.replaceAllUsesWith(SwapQubit_1_In);
                    op->erase();
                    return;
                }
            }
        });
    }
};

std::unique_ptr<Pass> createDisentangleSWAPPass()
{
    return std::make_unique<DisentangleSWAPPass>();
}

} // namespace catalyst
