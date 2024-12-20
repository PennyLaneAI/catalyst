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

// This algorithm is taken from https://arxiv.org/pdf/2012.07711, table 6 (Equivalences for
// basis-states in SWAP gate)

#define DEBUG_TYPE "disentangleswap"

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
#define GEN_PASS_DEF_DISENTANGLESWAPPASS
#define GEN_PASS_DECL_DISENTANGLESWAPPASS
#include "Quantum/Transforms/Passes.h.inc"

struct DisentangleSWAPPass : public impl::DisentangleSWAPPassBase<DisentangleSWAPPass> {
    using impl::DisentangleSWAPPassBase<DisentangleSWAPPass>::DisentangleSWAPPassBase;

    // function to create a single qubit gate with a given name
    // right after the SWAP is to be erased
    quantum::CustomOp createSimpleOneBitGate(StringRef gateName, const Value &inQubit,
                                             const Value &outQubit, mlir::IRRewriter &builder,
                                             Location &loc,
                                             const quantum::CustomOp &insert_after_gate)
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

    // above function overloaded to create a single qubit gate with a given name
    // if multiple gates are to be inserted after SWAP transformation
    quantum::CustomOp createSimpleOneBitGate(StringRef gateName, const Value &inQubit,
                                             mlir::IRRewriter &builder, Location &loc,
                                             const quantum::CustomOp &insert_after_gate)
    {
        OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPointAfter(insert_after_gate);
        quantum::CustomOp newGate =
            builder.create<quantum::CustomOp>(loc,
                                              /*out_qubits=*/mlir::TypeRange({inQubit.getType()}),
                                              /*out_ctrl_qubits=*/mlir::TypeRange(),
                                              /*params=*/mlir::ValueRange(),
                                              /*in_qubits=*/mlir::ValueRange({inQubit}),
                                              /*gate_name=*/gateName,
                                              /*adjoint=*/nullptr,
                                              /*in_ctrl_qubits=*/mlir::ValueRange(),
                                              /*in_ctrl_values=*/mlir::ValueRange());

        return newGate;
    }

    // function to create a two qubit gate with a given name
    // right after the SWAP is to be erased
    quantum::CustomOp createSimpleTwoBitGate(StringRef gateName, const Value &controlIn,
                                             const Value &targetIn, const Value &controlOut,
                                             const Value &targetOut, mlir::IRRewriter &builder,
                                             Location &loc,
                                             const quantum::CustomOp &insert_after_gate)
    {
        OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPointAfter(insert_after_gate);
        quantum::CustomOp newGate = builder.create<quantum::CustomOp>(
            loc,
            /*out_qubits=*/mlir::TypeRange({controlOut.getType(), targetOut.getType()}),
            /*out_ctrl_qubits=*/mlir::TypeRange({}),
            /*params=*/mlir::ValueRange(),
            /*in_qubits=*/mlir::ValueRange({controlIn, targetIn}),
            /*gate_name=*/gateName,
            /*adjoint=*/nullptr,
            /*in_ctrl_qubits=*/mlir::ValueRange({}),
            /*in_ctrl_values=*/mlir::ValueRange());

        return newGate;
    }

    // above function overloaded to create a two qubit gate with a given name
    // if multiple gates are to be inserted after SWAP transformation
    quantum::CustomOp createSimpleTwoBitGate(StringRef gateName, const Value &controlIn,
                                             const Value &targetIn, mlir::IRRewriter &builder,
                                             Location &loc,
                                             const quantum::CustomOp &insert_after_gate)
    {
        OpBuilder::InsertionGuard insertionGuard(builder);
        builder.setInsertionPointAfter(insert_after_gate);
        quantum::CustomOp newGate = builder.create<quantum::CustomOp>(
            loc,
            /*out_qubits=*/mlir::TypeRange({controlIn.getType(), targetIn.getType()}),
            /*out_ctrl_qubits=*/mlir::TypeRange({}),
            /*params=*/mlir::ValueRange(),
            /*in_qubits=*/mlir::ValueRange({controlIn, targetIn}),
            /*gate_name=*/gateName,
            /*adjoint=*/nullptr,
            /*in_ctrl_qubits=*/mlir::ValueRange({}),
            /*in_ctrl_values=*/mlir::ValueRange());

        return newGate;
    }

    bool canScheduleOn(RegisteredOperationName opInfo) const override
    {
        return opInfo.hasInterface<FunctionOpInterface>();
    }

    void runOnOperation() override
    {
        FunctionOpInterface func = cast<FunctionOpInterface>(getOperation());
        mlir::IRRewriter builder(func->getContext());
        Location loc = func->getLoc();

        PropagateSimpleStatesAnalysis &pssa = getAnalysis<PropagateSimpleStatesAnalysis>();
        llvm::DenseMap<Value, QubitState> qubitValues = pssa.getQubitValues();

        func->walk([&](quantum::CustomOp op) {
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
                // second qubit in |0>: SWAP(|0>,|0>)
                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    SwapQubit_0_Out.replaceAllUsesWith(SwapQubit_0_In);
                    SwapQubit_1_Out.replaceAllUsesWith(SwapQubit_1_In);
                    op->erase();
                    return;
                }
                // second qubit in |1>: SWAP(|0>,|1>)
                else if (pssa.isOne(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp xgate_on_0 = createSimpleOneBitGate(
                        "PauliX", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_1 = createSimpleOneBitGate(
                        "PauliX", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, xgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |+>: SWAP(|0>,|+>)
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp hgate_on_0 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |->: SWAP(|0>,|->)
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", SwapQubit_0_In, builder, loc, op);
                    xgate_on_0->getOperand(0) = SwapQubit_0_Out;

                    quantum::CustomOp hgate_on_0 = createSimpleOneBitGate(
                        "Hadamard", xgate_on_0->getResult(0), builder, loc, xgate_on_0);
                    (hgate_on_0->getOperand(0)) = (xgate_on_0->getResult(0));
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_1_In, builder, loc, hgate_on_0);
                    hgate_on_1->getOperand(0) = SwapQubit_1_Out;

                    quantum::CustomOp xgate_on_1 = createSimpleOneBitGate(
                        "PauliX", hgate_on_1->getResult(0), builder, loc, hgate_on_1);
                    xgate_on_1->getOperand(0) = hgate_on_1->getResult(0);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in NON_BASIS: SWAP(|0>,|NON_BASIS>)
                else if (pssa.isOther(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp cnot_on_1_0 = createSimpleTwoBitGate(
                        "CNOT", SwapQubit_1_In, SwapQubit_0_In, builder, loc, op);

                    cnot_on_1_0->getOperand(0) = SwapQubit_0_Out;
                    cnot_on_1_0->getOperand(1) = SwapQubit_1_Out;

                    quantum::CustomOp cnot_on_0_1 = createSimpleTwoBitGate(
                        "CNOT", cnot_on_1_0->getResult(0), cnot_on_1_0->getResult(1), builder, loc,
                        cnot_on_1_0);
                    cnot_on_0_1->getOperand(0) = cnot_on_1_0->getResult(0);
                    cnot_on_0_1->getOperand(1) = cnot_on_1_0->getResult(1);

                    SwapQubit_0_Out.replaceAllUsesWith(cnot_on_0_1->getResult(0));
                    SwapQubit_1_Out.replaceAllUsesWith(cnot_on_0_1->getResult(1));
                    op->erase();
                    return;
                }
            }

            // first qubit in |1>
            else if (pssa.isOne(qubitValues[SwapQubit_0_In])) {
                // second qubit in |0>: SWAP(|1>,|0>)
                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp xgate_on_0 = createSimpleOneBitGate(
                        "PauliX", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_1 = createSimpleOneBitGate(
                        "PauliX", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, xgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |1>: SWAP(|1>,|1>)
                else if (pssa.isOne(qubitValues[SwapQubit_1_In])) {
                    SwapQubit_0_Out.replaceAllUsesWith(SwapQubit_0_In);
                    SwapQubit_1_Out.replaceAllUsesWith(SwapQubit_1_In);
                    op->erase();
                    return;
                }
                // second qubit in |+>: SWAP(|1>,|+>)
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", SwapQubit_0_In, builder, loc, op);
                    xgate_on_0->getOperand(0) = SwapQubit_0_Out;

                    quantum::CustomOp hgate_on_0 = createSimpleOneBitGate(
                        "Hadamard", xgate_on_0->getResult(0), builder, loc, xgate_on_0);
                    (hgate_on_0->getOperand(0)) = (xgate_on_0->getResult(0));
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_1_In, builder, loc, hgate_on_0);
                    hgate_on_1->getOperand(0) = SwapQubit_1_Out;

                    quantum::CustomOp xgate_on_1 = createSimpleOneBitGate(
                        "PauliX", hgate_on_1->getResult(0), builder, loc, hgate_on_1);
                    xgate_on_1->getOperand(0) = hgate_on_1->getResult(0);
                    SwapQubit_1_Out.replaceAllUsesWith(xgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |->: SWAP(|1>,|->)
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp hgate_on_0 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |NON_BASIS>: SWAP(|1>,|NON_BASIS>)
                else if (pssa.isOther(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("PauliX", SwapQubit_1_In, builder, loc, op);
                    xgate_on_1->getOperand(0) = SwapQubit_1_Out;

                    quantum::CustomOp cnot_on_1_0 = createSimpleTwoBitGate(
                        "CNOT", xgate_on_1->getResult(0), SwapQubit_0_In, builder, loc, xgate_on_1);
                    cnot_on_1_0->getOperand(0) = SwapQubit_0_Out;
                    cnot_on_1_0->getOperand(1) = xgate_on_1->getResult(0);

                    quantum::CustomOp cnot_on_0_1 = createSimpleTwoBitGate(
                        "CNOT", cnot_on_1_0->getResult(0), cnot_on_1_0->getResult(1), builder, loc,
                        cnot_on_1_0);
                    cnot_on_0_1->getOperand(0) = cnot_on_1_0->getResult(0);
                    cnot_on_0_1->getOperand(1) = cnot_on_1_0->getResult(1);

                    SwapQubit_0_Out.replaceAllUsesWith(cnot_on_0_1->getResult(0));
                    SwapQubit_1_Out.replaceAllUsesWith(cnot_on_0_1->getResult(1));
                    op->erase();
                    return;
                }
            }

            // first qubit in |+>
            else if (pssa.isPlus(qubitValues[SwapQubit_0_In])) {
                // second qubit in |0>: SWAP(|+>,|0>)
                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp hgate_on_0 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |01>: SWAP(|+>,|1>)
                else if (pssa.isOne(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_0_In, builder, loc, op);
                    hgate_on_0->getOperand(0) = SwapQubit_0_Out;

                    quantum::CustomOp xgate_on_0 = createSimpleOneBitGate(
                        "PauliX", hgate_on_0->getResult(0), builder, loc, hgate_on_0);
                    (xgate_on_0->getOperand(0)) = (hgate_on_0->getResult(0));
                    SwapQubit_0_Out.replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("PauliX", SwapQubit_1_In, builder, loc, xgate_on_0);
                    xgate_on_1->getOperand(0) = SwapQubit_1_Out;

                    quantum::CustomOp hgate_on_1 = createSimpleOneBitGate(
                        "Hadamard", xgate_on_1->getResult(0), builder, loc, xgate_on_1);
                    hgate_on_1->getOperand(0) = xgate_on_1->getResult(0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |+>: SWAP(|+>,|+>)
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    SwapQubit_0_Out.replaceAllUsesWith(SwapQubit_0_In);
                    SwapQubit_1_Out.replaceAllUsesWith(SwapQubit_1_In);
                    op->erase();
                    return;
                }
                // second qubit in |->: SWAP(|+>,|->)
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp zgate_on_0 = createSimpleOneBitGate(
                        "PauliZ", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(zgate_on_0->getResult(0));

                    quantum::CustomOp zgate_on_1 = createSimpleOneBitGate(
                        "PauliZ", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, zgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(zgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |NON_BASIS>: SWAP(|+>,|NON_BASIS>)
                else if (pssa.isOther(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp cnot_on_0_1 = createSimpleTwoBitGate(
                        "CNOT", SwapQubit_0_In, SwapQubit_1_In, builder, loc, op);
                    cnot_on_0_1->getOperand(0) = SwapQubit_0_Out;
                    cnot_on_0_1->getOperand(1) = SwapQubit_1_Out;

                    quantum::CustomOp cnot_on_1_0 = createSimpleTwoBitGate(
                        "CNOT", cnot_on_0_1->getResult(1), cnot_on_0_1->getResult(0), builder, loc,
                        cnot_on_0_1);
                    cnot_on_1_0->getOperand(0) = cnot_on_0_1->getResult(0);
                    cnot_on_1_0->getOperand(1) = cnot_on_0_1->getResult(1);

                    SwapQubit_0_Out.replaceAllUsesWith(cnot_on_1_0->getResult(0));
                    SwapQubit_1_Out.replaceAllUsesWith(cnot_on_1_0->getResult(1));
                    op->erase();
                    return;
                }
            }

            // first qubit in |->
            else if (pssa.isMinus(qubitValues[SwapQubit_0_In])) {
                // second qubit in |0>: SWAP(|->,|0>)
                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp hgate_on_0 =
                        createSimpleOneBitGate("Hadamard", SwapQubit_0_In, builder, loc, op);
                    hgate_on_0->getOperand(0) = SwapQubit_0_Out;

                    quantum::CustomOp xgate_on_0 = createSimpleOneBitGate(
                        "PauliX", hgate_on_0->getResult(0), builder, loc, hgate_on_0);
                    (xgate_on_0->getOperand(0)) = (hgate_on_0->getResult(0));
                    SwapQubit_0_Out.replaceAllUsesWith(xgate_on_0->getResult(0));

                    quantum::CustomOp xgate_on_1 =
                        createSimpleOneBitGate("PauliX", SwapQubit_1_In, builder, loc, xgate_on_0);
                    xgate_on_1->getOperand(0) = SwapQubit_1_Out;

                    quantum::CustomOp hgate_on_1 = createSimpleOneBitGate(
                        "Hadamard", xgate_on_1->getResult(0), builder, loc, xgate_on_1);
                    hgate_on_1->getOperand(0) = xgate_on_1->getResult(0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |1>: SWAP(|->,|1>)
                else if (pssa.isOne(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp hgate_on_0 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(hgate_on_0->getResult(0));

                    quantum::CustomOp hgate_on_1 = createSimpleOneBitGate(
                        "Hadamard", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, hgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(hgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |+>: SWAP(|->,|+>)
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp zgate_on_0 = createSimpleOneBitGate(
                        "PauliZ", SwapQubit_0_In, SwapQubit_0_Out, builder, loc, op);
                    SwapQubit_0_Out.replaceAllUsesWith(zgate_on_0->getResult(0));

                    quantum::CustomOp zgate_on_1 = createSimpleOneBitGate(
                        "PauliZ", SwapQubit_1_In, SwapQubit_1_Out, builder, loc, zgate_on_0);
                    SwapQubit_1_Out.replaceAllUsesWith(zgate_on_1->getResult(0));
                    op->erase();
                    return;
                }
                // second qubit in |->: SWAP(|->,|->)
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    SwapQubit_0_Out.replaceAllUsesWith(SwapQubit_0_In);
                    SwapQubit_1_Out.replaceAllUsesWith(SwapQubit_1_In);
                    op->erase();
                    return;
                }
                // second qubit in |NON_BASIS>: SWAP(|->,|NON_BASIS>)
                else if (pssa.isOther(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp zgate_on_1 =
                        createSimpleOneBitGate("PauliZ", SwapQubit_1_In, builder, loc, op);
                    zgate_on_1->getOperand(0) = SwapQubit_1_Out;

                    quantum::CustomOp cnot_on_0_1 = createSimpleTwoBitGate(
                        "CNOT", SwapQubit_0_In, zgate_on_1->getResult(0), builder, loc, zgate_on_1);
                    cnot_on_0_1->getOperand(1) = SwapQubit_0_Out;
                    cnot_on_0_1->getOperand(1) = zgate_on_1->getResult(0);

                    quantum::CustomOp cnot_on_1_0 = createSimpleTwoBitGate(
                        "CNOT", cnot_on_0_1->getResult(1), cnot_on_0_1->getResult(0), builder, loc,
                        cnot_on_0_1);
                    cnot_on_1_0->getOperand(0) = cnot_on_0_1->getResult(0);
                    cnot_on_1_0->getOperand(1) = cnot_on_0_1->getResult(1);

                    SwapQubit_0_Out.replaceAllUsesWith(cnot_on_1_0->getResult(0));
                    SwapQubit_1_Out.replaceAllUsesWith(cnot_on_1_0->getResult(1));
                    op->erase();
                    return;
                }
            }

            // first qubit in |NON_BASIS>
            else if (pssa.isOther(qubitValues[SwapQubit_0_In])) {
                // second qubit in |0>: SWAP(|NON_BASIS>,|0>)
                if (pssa.isZero(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp cnot_on_0_1 = createSimpleTwoBitGate(
                        "CNOT", SwapQubit_0_In, SwapQubit_1_In, builder, loc, op);
                    cnot_on_0_1->getOperand(0) = SwapQubit_0_Out;
                    cnot_on_0_1->getOperand(1) = SwapQubit_1_Out;

                    quantum::CustomOp cnot_on_1_0 = createSimpleTwoBitGate(
                        "CNOT", cnot_on_0_1->getResult(1), cnot_on_0_1->getResult(0), builder, loc,
                        cnot_on_0_1);
                    cnot_on_1_0->getOperand(0) = cnot_on_0_1->getResult(0);
                    cnot_on_1_0->getOperand(1) = cnot_on_0_1->getResult(1);

                    SwapQubit_0_Out.replaceAllUsesWith(cnot_on_1_0->getResult(0));
                    SwapQubit_1_Out.replaceAllUsesWith(cnot_on_1_0->getResult(1));
                    op->erase();
                    return;
                }
                // second qubit in |1>: SWAP(|NON_BASIS>,|1>)
                else if (pssa.isOne(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp xgate_on_0 =
                        createSimpleOneBitGate("PauliX", SwapQubit_0_In, builder, loc, op);

                    xgate_on_0->getOperand(0) = SwapQubit_0_Out;

                    quantum::CustomOp cnot_on_0_1 = createSimpleTwoBitGate(
                        "CNOT", xgate_on_0->getResult(0), SwapQubit_1_In, builder, loc, xgate_on_0);

                    cnot_on_0_1->getOperand(0) = xgate_on_0->getResult(0);
                    cnot_on_0_1->getOperand(1) = SwapQubit_1_Out;

                    quantum::CustomOp cnot_on_1_0 = createSimpleTwoBitGate(
                        "CNOT", cnot_on_0_1->getResult(1), cnot_on_0_1->getResult(0), builder, loc,
                        cnot_on_0_1);
                    cnot_on_1_0->getOperand(0) = cnot_on_0_1->getResult(0);
                    cnot_on_1_0->getOperand(1) = cnot_on_0_1->getResult(1);

                    SwapQubit_0_Out.replaceAllUsesWith(cnot_on_1_0->getResult(0));
                    SwapQubit_1_Out.replaceAllUsesWith(cnot_on_1_0->getResult(1));
                    op->erase();
                    return;
                }
                // second qubit in |+>: SWAP(|NON_BASIS>,|+>)
                else if (pssa.isPlus(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp cnot_on_1_0 = createSimpleTwoBitGate(
                        "CNOT", SwapQubit_1_In, SwapQubit_0_In, builder, loc, op);

                    cnot_on_1_0->getOperand(0) = SwapQubit_0_Out;
                    cnot_on_1_0->getOperand(1) = SwapQubit_1_Out;

                    quantum::CustomOp cnot_on_0_1 = createSimpleTwoBitGate(
                        "CNOT", cnot_on_1_0->getResult(0), cnot_on_1_0->getResult(1), builder, loc,
                        cnot_on_1_0);
                    cnot_on_0_1->getOperand(0) = cnot_on_1_0->getResult(0);
                    cnot_on_0_1->getOperand(1) = cnot_on_1_0->getResult(1);

                    SwapQubit_0_Out.replaceAllUsesWith(cnot_on_0_1->getResult(0));
                    SwapQubit_1_Out.replaceAllUsesWith(cnot_on_0_1->getResult(1));
                    op->erase();
                    return;
                }
                // second qubit in |->: SWAP(|NON_BASIS>,|->)
                else if (pssa.isMinus(qubitValues[SwapQubit_1_In])) {
                    quantum::CustomOp zgate_on_0 =
                        createSimpleOneBitGate("PauliZ", SwapQubit_0_In, builder, loc, op);
                    zgate_on_0->getOperand(0) = SwapQubit_0_Out;

                    quantum::CustomOp cnot_on_1_0 = createSimpleTwoBitGate(
                        "CNOT", SwapQubit_1_In, zgate_on_0->getResult(0), builder, loc, zgate_on_0);

                    cnot_on_1_0->getOperand(0) = zgate_on_0->getResult(0);
                    cnot_on_1_0->getOperand(1) = SwapQubit_1_Out;

                    quantum::CustomOp cnot_on_0_1 = createSimpleTwoBitGate(
                        "CNOT", cnot_on_1_0->getResult(0), cnot_on_1_0->getResult(1), builder, loc,
                        cnot_on_1_0);
                    cnot_on_0_1->getOperand(0) = cnot_on_1_0->getResult(0);
                    cnot_on_0_1->getOperand(1) = cnot_on_1_0->getResult(1);

                    SwapQubit_0_Out.replaceAllUsesWith(cnot_on_0_1->getResult(0));
                    SwapQubit_1_Out.replaceAllUsesWith(cnot_on_0_1->getResult(1));
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
