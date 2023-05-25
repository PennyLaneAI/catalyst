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

// RUN: quantum-opt %s --lower-gradients="only=ps lower-vectors=false" --split-input-file | FileCheck %s

// CHECK-LABEL: @simple_circuit.argmap(%arg0: tensor<3xf64>) ->  tensor<?xf64>
func.func @simple_circuit(%arg0: tensor<3xf64>) -> f64 {
    // CHECK: [[c0:%[a-zA-Z0-9_]+]] = index.constant 0
    // CHECK: [[buff:%[a-zA-Z0-9_]+]] = gradient.vector_init : <f64>
    // CHECK: [[count:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<index>
    // CHECK: memref.store [[c0]], [[count]]

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    // CHECK: [[a0:%[a-zA-Z0-9_]+]] = tensor.extract %arg0
    %f0 = tensor.extract %arg0[%c0] : tensor<3xf64>
    // CHECK: [[a1:%[a-zA-Z0-9_]+]] = tensor.extract %arg0
    %f1 = tensor.extract %arg0[%c1] : tensor<3xf64>
    // CHECK: [[a2:%[a-zA-Z0-9_]+]] = tensor.extract %arg0
    %f2 = tensor.extract %arg0[%c2] : tensor<3xf64>

    // CHECK-NOT: quantum.
    %r = quantum.alloc(1) : !quantum.reg
    %idx = arith.index_cast %c0 : index to i64
    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit
    %q_1 = quantum.custom "h"() %q_0 : !quantum.bit
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
    // CHECK-NEXT: gradient.vector_push [[a0]], [[buff]]
    // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
    // CHECK-NEXT: memref.store [[ip1]], [[count]]
    // CHECK-NOT: quantum.
    %q_2 = quantum.custom "rz"(%f0) %q_1 : !quantum.bit
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
    // CHECK-NEXT: gradient.vector_push [[a0]], [[buff]]
    // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
    // CHECK-NEXT: gradient.vector_push [[a1]], [[buff]]
    // CHECK-NEXT: [[ip2:%[a-zA-Z0-9_]+]] = index.add [[ip1]]
    // CHECK-NEXT: gradient.vector_push [[a2]], [[buff]]
    // CHECK-NEXT: [[ip3:%[a-zA-Z0-9_]+]] = index.add [[ip2]]
    // CHECK-NEXT: memref.store [[ip3]], [[count]]
    // CHECK-NOT: quantum.
    %q_3 = quantum.custom "u3"(%f0, %f1, %f2) %q_2 : !quantum.bit

    // CHECK: [[vec:%[a-zA-Z0-9_]+]] = gradient.vector_load_data [[buff]]
    // CHECK: [[tensor:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[vec]] : memref<?xf64>
    // CHECK: return [[tensor]]
    func.return %f0 : f64
}

func.func @gradCall(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = gradient.grad "ps" @simple_circuit(%arg0) : (tensor<3xf64>) -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}

// -----

// CHECK-LABEL: @structured_circuit.argmap(%arg0: f64, %arg1: i1, %arg2: i1) ->  tensor<?xf64>
func.func @structured_circuit(%arg0: f64, %arg1: i1, %arg2: i1) -> f64 {
    // CHECK: [[c0:%[a-zA-Z0-9_]+]] = index.constant 0
    // CHECK: [[buff:%[a-zA-Z0-9_]+]] = gradient.vector_init : <f64>
    // CHECK: [[count:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<index>
    // CHECK: memref.store [[c0]], [[count]]

    %c0 = arith.constant 0 : i64

    // CHECK-NOT: quantum.
    %r = quantum.alloc(1) : !quantum.reg
    %q_0 = quantum.extract %r[%c0] : !quantum.reg -> !quantum.bit

    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
    // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
    // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
    // CHECK-NEXT: memref.store [[ip1]], [[count]]
    // CHECK-NOT: quantum.
    %q_1 = quantum.custom "rx"(%arg0) %q_0 : !quantum.bit

    // CHECK: scf.if %arg1
    %q_2 = scf.if %arg1 -> !quantum.bit {
        // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
        // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
        // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
        // CHECK-NEXT: memref.store [[ip1]], [[count]]
        // CHECK-NOT: quantum.
        %q_1_0 = quantum.custom "ry"(%arg0) %q_1 : !quantum.bit

        // CHECK:  scf.if %arg2
        %q_1_1 = scf.if %arg2 -> !quantum.bit {
            // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
            // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
            // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
            // CHECK-NEXT: memref.store [[ip1]], [[count]]
            // CHECK-NOT: quantum.
            %q_1_0_0 = quantum.custom "rz"(%arg0) %q_1_0 : !quantum.bit
            // CHECK-NOT: scf.yield
            scf.yield %q_1_0_0 : !quantum.bit
        // CHECK: else
        } else {
            // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
            // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
            // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
            // CHECK-NEXT: memref.store [[ip1]], [[count]]
            // CHECK-NOT: quantum.
            %q_1_0_1 = quantum.custom "rz"(%arg0) %q_1_0 : !quantum.bit
            // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
            // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
            // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
            // CHECK-NEXT: memref.store [[ip1]], [[count]]
            // CHECK-NOT: quantum.
            %q_1_0_2 = quantum.custom "rz"(%arg0) %q_1_0_1 : !quantum.bit
            // CHECK-NOT: scf.yield
            scf.yield %q_1_0_2 : !quantum.bit
        }
        // CHECK-NOT: scf.yield
        scf.yield %q_1_1 : !quantum.bit
    // CHECK-NOT: else
    } else {
        scf.yield %q_1 : !quantum.bit
    }

    // Branching is only valid as long as each block is traversed at most once!
    cf.br ^exit

  ^exit:
    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
    // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
    // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
    // CHECK-NEXT: memref.store [[ip1]], [[count]]
    // CHECK-NOT: quantum.
    %q_3 = quantum.custom "rx"(%arg0) %q_2 : !quantum.bit

    // CHECK: [[vec:%[a-zA-Z0-9_]+]] = gradient.vector_load_data [[buff]]
    // CHECK: [[tensor:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[vec]] : memref<?xf64>
    // CHECK: return [[tensor]]
    func.return %arg0 : f64
}

func.func @gradCall(%arg0: f64, %b0: i1, %b1: i1) -> f64 {
    %0 = gradient.grad "ps" @structured_circuit(%arg0, %b0, %b1) : (f64, i1, i1) -> f64
    func.return %0 : f64
}

// -----

// CHECK-LABEL: @loop_circuit.argmap(%arg0: f64) ->  tensor<?xf64>
func.func @loop_circuit(%arg0: f64) -> f64 {
    // CHECK: [[c0:%[a-zA-Z0-9_]+]] = index.constant 0
    // CHECK: [[buff:%[a-zA-Z0-9_]+]] = gradient.vector_init : <f64>
    // CHECK: [[count:%[a-zA-Z0-9_]+]] = memref.alloca() : memref<index>
    // CHECK: memref.store [[c0]], [[count]]

    %idx = arith.constant 0 : i64

    // CHECK-NOT: quantum.
    %r = quantum.alloc(1) : !quantum.reg
    %q_0 = quantum.extract %r[%idx] : !quantum.reg -> !quantum.bit

    // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
    // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
    // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
    // CHECK-NEXT: memref.store [[ip1]], [[count]]
    // CHECK-NOT: quantum.
    %q_1 = quantum.custom "rx"(%arg0) %q_0 : !quantum.bit

    %lb = arith.constant 0 : index
    %ub = arith.constant 10: index
    %st = arith.constant 1 : index

    // CHECK: scf.for
    %q_2 = scf.for %i = %lb to %ub step %st iter_args(%q_1_0 = %q_1) -> !quantum.bit {
        // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
        // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
        // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
        // CHECK-NEXT: memref.store [[ip1]], [[count]]
        // CHECK-NOT: quantum.
        %q_1_1 = quantum.custom "ry"(%arg0) %q_1_0 : !quantum.bit

        // CHECK-NOT: scf.yield
        scf.yield %q_1_1 : !quantum.bit
    }

    // CHECK: scf.for
    %q_3 = scf.for %j = %lb to %ub step %st iter_args(%q_2_0 = %q_2) -> !quantum.bit {
        // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
        // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
        // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
        // CHECK-NEXT: memref.store [[ip1]], [[count]]
        // CHECK-NOT: quantum.
        %q_2_1 = quantum.custom "ry"(%arg0) %q_2_0 : !quantum.bit

        // CHECK: scf.for
        %q_1_1 = scf.for %k = %j to %ub step %st iter_args(%q_2_1_0 = %q_2_1) -> !quantum.bit {
            // CHECK: [[idx:%[a-zA-Z0-9_]+]] = memref.load [[count]]
            // CHECK-NEXT: gradient.vector_push %arg0, [[buff]]
            // CHECK-NEXT: [[ip1:%[a-zA-Z0-9_]+]] = index.add [[idx]]
            // CHECK-NEXT: memref.store [[ip1]], [[count]]
            // CHECK-NOT: quantum.
            %q_2_1_1 = quantum.custom "rz"(%arg0) %q_2_1_0 : !quantum.bit

            // CHECK-NOT: scf.yield
            scf.yield %q_2_1_1 : !quantum.bit
        }

        // CHECK-NOT: scf.yield
        scf.yield %q_1_1 : !quantum.bit
    }

    // CHECK: [[vec:%[a-zA-Z0-9_]+]] = gradient.vector_load_data [[buff]]
    // CHECK: [[tensor:%[a-zA-Z0-9_]+]] = bufferization.to_tensor [[vec]] : memref<?xf64>
    // CHECK: return [[tensor]]
    func.return %arg0 : f64
}

func.func @gradCall(%arg0: f64) -> f64 {
    %0 = gradient.grad "ps" @loop_circuit(%arg0) : (f64) -> f64
    func.return %0 : f64
}
