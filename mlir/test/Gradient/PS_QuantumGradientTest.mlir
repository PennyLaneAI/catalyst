// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --lower-gradients --split-input-file | FileCheck %s

// CHECK-LABEL: @simple_circuit.qgrad(%arg0: tensor<3xf64>, %arg1: index) -> tensor<?xf64>
func.func @simple_circuit(%arg0: tensor<3xf64>) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %f0 = tensor.extract %arg0[%c0] : tensor<3xf64>
    %f1 = tensor.extract %arg0[%c1] : tensor<3xf64>
    %f2 = tensor.extract %arg0[%c2] : tensor<3xf64>

    // CHECK-DAG: [[c0:%[a-zA-Z0-9_]+]] = index.constant 0
    // CHECK-DAG: [[c1:%[a-zA-Z0-9_]+]] = index.constant 1
    // CHECK-DAG: [[divisor:%[a-zA-Z0-9_]+]] = arith.constant 2.0
    // CHECK-DAG: [[shift0pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<0, 1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift0neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<0, -1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift1pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<1, 1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift1neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<1, -1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift2pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<2, 1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift2neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<2, -1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift3pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<3, 1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift3neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<3, -1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[selBuff:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<0xindex>
    // CHECK-DAG: [[gradIdx:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<index>
    // CHECK-DAG: memref.store [[c0]], [[gradIdx]]
    // CHECK-DAG: [[grad:%[a-zA-Z0-9_]+]] = memref.alloc(%arg1) : memref<?xf64>

    %idx = arith.index_cast %c0 : index to i64
    // CHECK-NOT: quantum.qalloc
    %r = quantum.alloc(1) : !quantum.reg
    // CHECK-NOT: quantum.extract
    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit

    // CHECK-NOT: quantum.custom
    %q_1 = quantum.custom "h"() %q_0 : !quantum.bit

    // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
    // CHECK: [[epos:%[a-zA-Z0-9_]+]] = call @simple_circuit.shifted(%arg0, [[shift0pos]], [[sel]])
    // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = call @simple_circuit.shifted(%arg0, [[shift0neg]], [[sel]])
    // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
    // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
    // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
    // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
    // CHECK: memref.store [[newIdx]], [[gradIdx]]
    //
    // CHECK-NOT: quantum.custom
    %q_2 = quantum.custom "rz"(%f0) %q_1 : !quantum.bit

    // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
    // CHECK: [[epos:%[a-zA-Z0-9_]+]] = call @simple_circuit.shifted(%arg0, [[shift1pos]], [[sel]])
    // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = call @simple_circuit.shifted(%arg0, [[shift1neg]], [[sel]])
    // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
    // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
    // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
    // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
    // CHECK: memref.store [[newIdx]], [[gradIdx]]
    //
    // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
    // CHECK: [[epos:%[a-zA-Z0-9_]+]] = call @simple_circuit.shifted(%arg0, [[shift2pos]], [[sel]])
    // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = call @simple_circuit.shifted(%arg0, [[shift2neg]], [[sel]])
    // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
    // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
    // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
    // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
    // CHECK: memref.store [[newIdx]], [[gradIdx]]
    //
    // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
    // CHECK: [[epos:%[a-zA-Z0-9_]+]] = call @simple_circuit.shifted(%arg0, [[shift3pos]], [[sel]])
    // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = call @simple_circuit.shifted(%arg0, [[shift3neg]], [[sel]])
    // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
    // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
    // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
    // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
    // CHECK: memref.store [[newIdx]], [[gradIdx]]
    //
    // CHECK-NOT: quantum.custom
    %q_3 = quantum.custom "u3"(%f0, %f1, %f2) %q_2 : !quantum.bit
    %obs = quantum.namedobs %q_3[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[ret:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[grad]]
    // CHECK: return [[ret]] : tensor<?xf64>
    func.return %expval : f64
}

func.func @gradCall0(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = gradient.grad "auto" @simple_circuit(%arg0) : (tensor<3xf64>) -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}

// -----

// CHECK-LABEL: @structured_circuit.qgrad(%arg0: f64, %arg1: i1, %arg2: i1, %arg3: index) -> tensor<?xf64>
func.func @structured_circuit(%arg0: f64, %arg1: i1, %arg2: i1) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    // CHECK-DAG: [[c0:%[a-zA-Z0-9_]+]] = index.constant 0
    // CHECK-DAG: [[c1:%[a-zA-Z0-9_]+]] = index.constant 1
    // CHECK-DAG: [[divisor:%[a-zA-Z0-9_]+]] = arith.constant 2.0
    // CHECK-DAG: [[shift0pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<0, 1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift0neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<0, -1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift1pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<1, 1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift1neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<1, -1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift2pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<2, 1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift2neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<2, -1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift3pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<3, 1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift3neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<3, -1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift4pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<4, 1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift4neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<4, -1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift5pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<5, 1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[shift5neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<5, -1.57{{[0-9]*}}> : tensor<6xf64>
    // CHECK-DAG: [[selBuff:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<0xindex>
    // CHECK-DAG: [[gradIdx:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<index>
    // CHECK-DAG: memref.store [[c0]], [[gradIdx]]
    // CHECK-DAG: [[grad:%[a-zA-Z0-9_]+]] = memref.alloc(%arg3) : memref<?xf64>

    %idx = arith.constant 0 : i64
    // CHECK-NOT: quantum.qalloc
    %r = quantum.alloc(1) : !quantum.reg
    // CHECK-NOT: quantum.extract
    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit

    // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
    // CHECK: [[epos:%[a-zA-Z0-9_]+]] = call @structured_circuit.shifted(%arg0, %arg1, %arg2, [[shift0pos]], [[sel]])
    // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = call @structured_circuit.shifted(%arg0, %arg1, %arg2, [[shift0neg]], [[sel]])
    // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
    // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
    // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
    // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
    // CHECK: memref.store [[newIdx]], [[gradIdx]]
    //
    // CHECK-NOT: quantum.custom
    %q_1 = quantum.custom "rx"(%arg0) %q_0 : !quantum.bit

    // CHECK: scf.if %arg1
    %q_2 = scf.if %arg1 -> !quantum.bit {
        // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
        // CHECK: [[epos:%[a-zA-Z0-9_]+]] = func.call @structured_circuit.shifted(%arg0, %true, %arg2, [[shift1pos]], [[sel]])
        // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = func.call @structured_circuit.shifted(%arg0, %true, %arg2, [[shift1neg]], [[sel]])
        // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
        // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
        // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
        // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
        // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
        // CHECK: memref.store [[newIdx]], [[gradIdx]]
        //
        // CHECK-NOT: quantum.custom
        %q_1_0 = quantum.custom "ry"(%arg0) %q_1 : !quantum.bit

        // CHECK: scf.if %arg2
        %q_1_1 = scf.if %arg2 -> !quantum.bit {
            // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
            // CHECK: [[epos:%[a-zA-Z0-9_]+]] = func.call @structured_circuit.shifted(%arg0, %true, %true, [[shift2pos]], [[sel]])
            // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = func.call @structured_circuit.shifted(%arg0, %true, %true, [[shift2neg]], [[sel]])
            // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
            // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
            // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
            // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
            // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
            // CHECK: memref.store [[newIdx]], [[gradIdx]]
            //
            // CHECK-NOT: quantum.custom
            %q_1_0_0 = quantum.custom "rz"(%arg0) %q_1_0 : !quantum.bit
            scf.yield %q_1_0_0 : !quantum.bit
        // CHECK: else
        } else {
            // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
            // CHECK: [[epos:%[a-zA-Z0-9_]+]] = func.call @structured_circuit.shifted(%arg0, %true, %false, [[shift3pos]], [[sel]])
            // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = func.call @structured_circuit.shifted(%arg0, %true, %false, [[shift3neg]], [[sel]])
            // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
            // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
            // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
            // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
            // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
            // CHECK: memref.store [[newIdx]], [[gradIdx]]
            //
            // CHECK-NOT: quantum.custom
            %q_1_0_1 = quantum.custom "rz"(%arg0) %q_1_0 : !quantum.bit
            // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
            // CHECK: [[epos:%[a-zA-Z0-9_]+]] = func.call @structured_circuit.shifted(%arg0, %true, %false, [[shift4pos]], [[sel]])
            // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = func.call @structured_circuit.shifted(%arg0, %true, %false, [[shift4neg]], [[sel]])
            // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
            // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
            // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
            // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
            // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
            // CHECK: memref.store [[newIdx]], [[gradIdx]]
            //
            // CHECK-NOT: quantum.custom
            %q_1_0_2 = quantum.custom "rz"(%arg0) %q_1_0_1 : !quantum.bit
            scf.yield %q_1_0_2 : !quantum.bit
        }
        scf.yield %q_1_1 : !quantum.bit
    } else {
        scf.yield %q_1 : !quantum.bit
    }

    // Branching is only valid as long as each block is traversed at most once!
    cf.br ^exit

  ^exit:
    // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
    // CHECK: [[epos:%[a-zA-Z0-9_]+]] = call @structured_circuit.shifted(%arg0, %arg1, %arg2, [[shift5pos]], [[sel]])
    // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = call @structured_circuit.shifted(%arg0, %arg1, %arg2, [[shift5neg]], [[sel]])
    // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
    // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
    // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
    // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
    // CHECK: memref.store [[newIdx]], [[gradIdx]]
    //
    // CHECK-NOT: quantum.custom
    %q_3 = quantum.custom "rx"(%arg0) %q_2 : !quantum.bit
    %obs = quantum.namedobs %q_3[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[ret:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[grad]]
    // CHECK: return [[ret]] : tensor<?xf64>
    func.return %expval : f64
}

func.func @gradCall1(%arg0: f64, %b0: i1, %b1: i1) -> f64 {
    %0 = gradient.grad "auto" @structured_circuit(%arg0, %b0, %b1) : (f64, i1, i1) -> f64
    func.return %0 : f64
}

// -----

// CHECK-LABEL: @loop_circuit.qgrad(%arg0: f64, %arg1: index) -> tensor<?xf64>
func.func @loop_circuit(%arg0: f64) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    // CHECK-DAG: [[c0:%[a-zA-Z0-9_]+]] = index.constant 0
    // CHECK-DAG: [[c1:%[a-zA-Z0-9_]+]] = index.constant 1
    // CHECK-DAG: [[divisor:%[a-zA-Z0-9_]+]] = arith.constant 2.0
    // CHECK-DAG: [[shift0pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<0, 1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift0neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<0, -1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift1pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<1, 1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift1neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<1, -1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift2pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<2, 1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift2neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<2, -1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift3pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<3, 1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[shift3neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<3, -1.57{{[0-9]*}}> : tensor<4xf64>
    // CHECK-DAG: [[selBuff:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<2xindex>
    // CHECK-DAG: [[gradIdx:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<index>
    // CHECK-DAG: memref.store [[c0]], [[gradIdx]]
    // CHECK-DAG: [[grad:%[a-zA-Z0-9_]+]] = memref.alloc(%arg1) : memref<?xf64>

    %idx = arith.constant 0 : i64
    // CHECK-NOT: quantum.qalloc
    %r = quantum.alloc(1) : !quantum.reg
    // CHECK-NOT: quantum.extract
    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit

    // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
    // CHECK: [[epos:%[a-zA-Z0-9_]+]] = call @loop_circuit.shifted(%arg0, [[shift0pos]], [[sel]])
    // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = call @loop_circuit.shifted(%arg0, [[shift0neg]], [[sel]])
    // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
    // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
    // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
    // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
    // CHECK: memref.store [[newIdx]], [[gradIdx]]
    //
    // CHECK-NOT: quantum.custom
    %q_1 = quantum.custom "rx"(%arg0) %q_0 : !quantum.bit

    %lb = arith.constant 0 : index
    %ub = arith.constant 10: index
    %st = arith.constant 1 : index

    // CHECK: scf.for [[i:%[a-zA-Z0-9_]+]] =
    %q_2 = scf.for %i = %lb to %ub step %st iter_args(%q_1_0 = %q_1) -> !quantum.bit {
        // CHECK: memref.store [[i]], [[selBuff]][[[c0]]]

        // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
        // CHECK: [[epos:%[a-zA-Z0-9_]+]] = func.call @loop_circuit.shifted(%arg0, [[shift1pos]], [[sel]])
        // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = func.call @loop_circuit.shifted(%arg0, [[shift1neg]], [[sel]])
        // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
        // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
        // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
        // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
        // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
        // CHECK: memref.store [[newIdx]], [[gradIdx]]
        //
        // CHECK-NOT: quantum.custom
        %q_1_1 = quantum.custom "ry"(%arg0) %q_1_0 : !quantum.bit

        scf.yield %q_1_1 : !quantum.bit
    }

    // CHECK: scf.for [[j:%[a-zA-Z0-9_]+]] =
    %q_3 = scf.for %j = %lb to %ub step %st iter_args(%q_2_0 = %q_2) -> !quantum.bit {
        // CHECK: memref.store [[j]], [[selBuff]][[[c0]]]

        // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
        // CHECK: [[epos:%[a-zA-Z0-9_]+]] = func.call @loop_circuit.shifted(%arg0, [[shift2pos]], [[sel]])
        // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = func.call @loop_circuit.shifted(%arg0, [[shift2neg]], [[sel]])
        // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
        // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
        // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
        // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
        // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
        // CHECK: memref.store [[newIdx]], [[gradIdx]]
        //
        // CHECK-NOT: quantum.custom
        %q_2_1 = quantum.custom "ry"(%arg0) %q_2_0 : !quantum.bit

        // CHECK: scf.for [[k:%[a-zA-Z0-9_]+]] =
        %q_1_1 = scf.for %k = %j to %ub step %st iter_args(%q_2_1_0 = %q_2_1) -> !quantum.bit {
            // CHECK: memref.store [[k]], [[selBuff]][[[c1]]]

            // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
            // CHECK: [[epos:%[a-zA-Z0-9_]+]] = func.call @loop_circuit.shifted(%arg0, [[shift3pos]], [[sel]])
            // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = func.call @loop_circuit.shifted(%arg0, [[shift3neg]], [[sel]])
            // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
            // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
            // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
            // CHECK: memref.store [[deriv]], [[grad]][[[idx]]]
            // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
            // CHECK: memref.store [[newIdx]], [[gradIdx]]
            //
            // CHECK-NOT: quantum.custom
            %q_2_1_1 = quantum.custom "rz"(%arg0) %q_2_1_0 : !quantum.bit

            scf.yield %q_2_1_1 : !quantum.bit
        }

        scf.yield %q_1_1 : !quantum.bit
    }
    %obs = quantum.namedobs %q_3[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[ret:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[grad]]
    // CHECK: return [[ret]] : tensor<?xf64>
    func.return %expval : f64
}

func.func @gradCall2(%arg0: f64) -> f64 {
    %0 = gradient.grad "auto" @loop_circuit(%arg0) : (f64) -> f64
    func.return %0 : f64
}

// -----

// CHECK-LABEL: @tensor_circuit.qgrad(%arg0: f64, %arg1: index) -> tensor<?x2x3xf64>
func.func @tensor_circuit(%arg0: f64) -> tensor<2x3xf64> attributes {qnode, diff_method = "parameter-shift"} {
    // CHECK-DAG: [[c0:%[a-zA-Z0-9_]+]] = index.constant 0
    // CHECK-DAG: [[c1:%[a-zA-Z0-9_]+]] = index.constant 1
    // CHECK-DAG: [[divisor:%[a-zA-Z0-9_]+]] = arith.constant dense<2.0{{[0e+]*}}> : tensor<2x3xf64>
    // CHECK-DAG: [[shift0pos:%[a-zA-Z0-9_]+]] = arith.constant sparse<0, 1.57{{[0-9]*}}> : tensor<1xf64>
    // CHECK-DAG: [[shift0neg:%[a-zA-Z0-9_]+]] = arith.constant sparse<0, -1.57{{[0-9]*}}> : tensor<1xf64>
    // CHECK-DAG: [[selBuff:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<0xindex>
    // CHECK-DAG: [[gradIdx:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<index>
    // CHECK-DAG: memref.store [[c0]], [[gradIdx]]
    // CHECK-DAG: [[grad:%[a-zA-Z0-9_]+]] = memref.alloc(%arg1) : memref<?x2x3xf64>

    %idx = arith.constant 0 : i64
    // CHECK-NOT: quantum.qalloc
    %r = quantum.alloc(1) : !quantum.reg
    // CHECK-NOT: quantum.extract
    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit

    // CHECK: [[sel:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[selBuff]]
    // CHECK: [[epos:%[a-zA-Z0-9_]+]] = call @tensor_circuit.shifted(%arg0, [[shift0pos]], [[sel]])
    // CHECK: [[eneg:%[a-zA-Z0-9_]+]] = call @tensor_circuit.shifted(%arg0, [[shift0neg]], [[sel]])
    // CHECK: [[diff:%[a-zA-Z0-9_]+]] = arith.subf [[epos]], [[eneg]]
    // CHECK: [[deriv:%[a-zA-Z0-9_]+]] = arith.divf [[diff]], [[divisor]]
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[gradIdx]]
    // CHECK: [[view:%[a-zA-Z0-9_]+]] = memref.subview [[grad]][[[idx]], 0, 0] [1, 2, 3] [1, 1, 1]
    // CHECK: bufferization.materialize_in_destination [[deriv]] in writable [[view]]
    // CHECK: [[newIdx:%[a-zA-Z0-9_]+]] = index.add [[idx]], [[c1]]
    // CHECK: memref.store [[newIdx]], [[gradIdx]]
    //
    // CHECK-NOT: quantum.custom
    %q_1 = quantum.custom "rx"(%arg0) %q_0 : !quantum.bit
    %obs = quantum.namedobs %q_1[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[ret:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[grad]]
    // CHECK: return [[ret]] : tensor<?x2x3xf64>
    %res = tensor.from_elements %expval, %expval, %expval, %expval, %expval, %expval : tensor<2x3xf64>
    func.return %res : tensor<2x3xf64>
}

func.func @gradCall3(%arg0: f64) -> tensor<2x3xf64> {
    %0 = gradient.grad "auto" @tensor_circuit(%arg0) : (f64) -> tensor<2x3xf64>
    func.return %0 : tensor<2x3xf64>
}

// -----

// CHECK-LABEL: @multi_res_circuit.qgrad(%arg0: f64, %arg1: index) -> (tensor<?xf64>, tensor<?x2xf64>)
func.func @multi_res_circuit(%arg0: f64) -> (f64, tensor<2xf64>) attributes {qnode, diff_method = "parameter-shift"} {
    // CHECK-DAG:     [[C0:%.+]] = index.constant 0
    // CHECK-DAG:     [[C1:%.+]] = index.constant 1
    // CHECK-DAG:     [[DIVISOR0:%.+]] = arith.constant 2.0{{.*}} : f64
    // CHECK-DAG:     [[DIVISOR1:%.+]] = arith.constant dense<2.0{{.*}}> : tensor<2xf64>
    // CHECK-DAG:     [[SHIFTPOS:%.+]] = arith.constant sparse<0, 1.57{{.*}}> : tensor<1xf64>
    // CHECK-DAG:     [[SHIFTNEG:%.+]] = arith.constant sparse<0, -1.57{{.*}}> : tensor<1xf64>
    // CHECK-DAG:     [[SELBUFF:%.+]] = memref.alloca() : memref<0xindex>
    // CHECK-DAG:     [[GRAD0:%.+]] = memref.alloc(%arg1) : memref<?xf64>
    // CHECK-DAG:     [[GRAD1:%.+]] = memref.alloc(%arg1) : memref<?x2xf64>
    // CHECK-DAG:     [[GRADIDX:%.+]] = memref.alloca() : memref<index>
    // CHECK-DAG:     memref.store [[C0]], [[GRADIDX]]

    %idx = arith.constant 0 : i64

    // CHECK-NOT: quantum.
    %r = quantum.alloc(1) : !quantum.reg
    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit

    // CHECK:         [[SEL:%.+]] = bufferization.to_tensor [[SELBUFF]] restrict : memref<0xindex>
    // CHECK:         [[EVALPOS:%.+]]:2 = call @multi_res_circuit.shifted(%arg0, [[SHIFTPOS]], [[SEL]]) : {{.+}} -> (f64, tensor<2xf64>)
    // CHECK:         [[EVALNEG:%.+]]:2 = call @multi_res_circuit.shifted(%arg0, [[SHIFTNEG]], [[SEL]]) : {{.+}} -> (f64, tensor<2xf64>)
    // CHECK:         [[DIFF0:%.+]] = arith.subf [[EVALPOS]]#0, [[EVALNEG]]#0
    // CHECK:         [[DERIV0:%.+]] = arith.divf [[DIFF0]], [[DIVISOR0]]
    // CHECK:         [[DIFF1:%.+]] = arith.subf [[EVALPOS]]#1, [[EVALNEG]]#1
    // CHECK:         [[DERIV1:%.+]] = arith.divf [[DIFF1]], [[DIVISOR1]]
    // CHECK:         [[IDX:%.+]] = memref.load [[GRADIDX]]
    // CHECK:         memref.store [[DERIV0]], [[GRAD0]][[[IDX]]]
    // CHECK:         [[VIEW:%.+]] = memref.subview [[GRAD1]][[[IDX]], 0] [1, 2] [1, 1]
    // CHECK:         bufferization.materialize_in_destination [[DERIV1]] in writable [[VIEW]]
    // CHECK:         [[NEWIDX:%.+]] = index.add [[IDX]], [[C1]]
    // CHECK:         memref.store [[NEWIDX]], [[GRADIDX]]
    // CHECK-NOT: quantum.
    %q_1 = quantum.custom "rx"(%arg0) %q_0 : !quantum.bit
    %obs = quantum.namedobs %q_1[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK:         [[RES0:%.+]] = bufferization.to_tensor [[GRAD0]]
    // CHECK:         [[RES1:%.+]] = bufferization.to_tensor [[GRAD1]]
    // CHECK:         return [[RES0]], [[RES1]] : tensor<?xf64>, tensor<?x2xf64>
    %res = tensor.from_elements %expval, %expval : tensor<2xf64>
    func.return %arg0, %res : f64, tensor<2xf64>
}

func.func @gradCall4(%arg0: f64) -> (f64, tensor<2xf64>)  {
    %0:2 = gradient.grad "auto" @multi_res_circuit(%arg0) : (f64) -> (f64, tensor<2xf64>)
    func.return %0#0, %0#1 : f64, tensor<2xf64>
}

// -----

// Check multiple grad calls to same function
func.func private @funcMultiCall(%arg0: f64) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0 : i64
    %r = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %r[%c0] : !quantum.reg -> !quantum.bit

    %q_1 = quantum.custom "rz"(%arg0) %q : !quantum.bit
    %obs = quantum.namedobs %q_1[PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    func.return %expval : f64
}

// CHECK-LABEL: @funcMultiCall.shifted(%arg0: f64, %arg1: tensor<1xf64>, %arg2: tensor<0xindex>) -> f64

// CHECK-LABEL: @funcMultiCall.qgrad(%arg0: f64, %arg1: index) -> tensor<?xf64>
    // CHECK:   call @funcMultiCall.shifted(%arg0, {{%.+}}, {{%.+}}) : (f64, tensor<1xf64>, tensor<0xindex>) -> f64
    // CHECK:   call @funcMultiCall.shifted(%arg0, {{%.+}}, {{%.+}}) : (f64, tensor<1xf64>, tensor<0xindex>) -> f64
// }

// CHECK-LABEL: @gradCallMultiCall
func.func @gradCallMultiCall(%arg0: f64) -> (f64, f64) {
    %0 = gradient.grad "auto" @funcMultiCall(%arg0) : (f64) -> f64
    %1 = gradient.grad "auto" @funcMultiCall(%arg0) : (f64) -> f64
    func.return %0, %1 : f64, f64
}
