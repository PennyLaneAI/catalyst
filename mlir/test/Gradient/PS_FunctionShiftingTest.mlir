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

// RUN: quantum-opt %s --lower-gradients | FileCheck %s

// CHECK-LABEL: @simple_circuit.shifted(%arg0: tensor<3xf64>, %arg1: tensor<4xf64>, %arg2: tensor<0xindex>) -> f64
func.func @simple_circuit(%arg0: tensor<3xf64>) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    // CHECK: [[a0:%.+]] = tensor.extract %arg0[%c0]
    %f0 = tensor.extract %arg0[%c0] : tensor<3xf64>
    // CHECK: [[a1:%.+]] = tensor.extract %arg0[%c1]
    %f1 = tensor.extract %arg0[%c1] : tensor<3xf64>
    // CHECK: [[a2:%.+]] = tensor.extract %arg0[%c2]
    %f2 = tensor.extract %arg0[%c2] : tensor<3xf64>

    %r_0 = quantum.alloc(1) : !quantum.reg
    %idx = arith.index_cast %c0 : index to i64
    %q_0 = quantum.extract %r_0[%idx] : !quantum.reg -> !quantum.bit
    // CHECK: [[q0:%.+]] = quantum.custom "h"
    %q_1 = quantum.custom "h"() %q_0 : !quantum.bit
    // CHECK-NEXT: [[s0:%.+]] = tensor.extract %arg1
    // CHECK-NEXT: [[r0:%.+]] = arith.addf [[s0]], [[a0]] : f64
    // CHECK-NEXT: [[q1:%.+]] = quantum.custom "rz"([[r0]]) [[q0]]
    %q_2 = quantum.custom "rz"(%f0) %q_1 : !quantum.bit
    // CHECK-NEXT: [[s1:%.+]] = tensor.extract %arg1
    // CHECK-NEXT: [[r1:%.+]] = arith.addf [[s1]], [[a0]] : f64
    // CHECK-NEXT: [[s2:%.+]] = tensor.extract %arg1
    // CHECK-NEXT: [[r2:%.+]] = arith.addf [[s2]], [[a1]] : f64
    // CHECK-NEXT: [[s3:%.+]] = tensor.extract %arg1
    // CHECK-NEXT: [[r3:%.+]] = arith.addf [[s3]], [[a2]] : f64
    // CHECK-NEXT: {{%.+}} = quantum.custom "u3"([[r1]], [[r2]], [[r3]]) [[q1]]
    %q_3 = quantum.custom "u3"(%f0, %f1, %f2) %q_2 : !quantum.bit

    %r_1 = quantum.insert %r_0[%idx], %q_3 : !quantum.reg, !quantum.bit
    quantum.dealloc %r_1 : !quantum.reg
    func.return %f0 : f64
}

func.func @gradCall0(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %0 = gradient.grad "auto" @simple_circuit(%arg0) : (tensor<3xf64>) -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}

// -----

// CHECK-LABEL: @structured_circuit.shifted(%arg0: tensor<1xf64>, %arg1: i1, %arg2: i1, %arg3: tensor<6xf64>, %arg4: tensor<0xindex>) -> f64
func.func @structured_circuit(%arg0: tensor<1xf64>, %arg1: i1, %arg2: i1) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    %c0 = arith.constant 0 : index
    // CHECK: [[e0:%.+]] = tensor.extract %arg0[%c0] : tensor<1xf64>
    %f0 = tensor.extract %arg0[%c0] : tensor<1xf64>

    %idx = arith.constant 0 : i64
    %r_0 = quantum.alloc(1) : !quantum.reg
    // CHECK: [[q0:%.+]] = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %q_0 = quantum.extract %r_0[%idx] : !quantum.reg -> !quantum.bit

    // CHECK-NEXT: [[s0:%.+]] = tensor.extract %arg3
    // CHECK-NEXT: [[r0:%.+]] = arith.addf [[s0]], [[e0]] : f64
    // CHECK-NEXT: [[q1:%.+]] = quantum.custom "rx"([[r0]]) [[q0]]
    %q_1 = quantum.custom "rx"(%f0) %q_0 : !quantum.bit

    // CHECK: [[q2:%.+]] = scf.if
    %q_2 = scf.if %arg1 -> !quantum.bit {
        // CHECK-NEXT: [[s1:%.+]] = tensor.extract %arg3
        // CHECK-NEXT: [[r1:%.+]] = arith.addf [[s1]], [[e0]] : f64
        // CHECK-NEXT: [[q10:%.+]] = quantum.custom "ry"([[r1]]) [[q1]]
        %q_1_0 = quantum.custom "ry"(%f0) %q_1 : !quantum.bit

        // CHECK: [[q11:%.+]] = scf.if
        %q_1_1 = scf.if %arg2 -> !quantum.bit {
            // CHECK-NEXT: [[s2:%.+]] = tensor.extract %arg3
            // CHECK-NEXT: [[r2:%.+]] = arith.addf [[s2]], [[e0]] : f64
            // CHECK-NEXT: [[q100:%.+]] = quantum.custom "rz"([[r2]]) [[q10]]
            %q_1_0_0 = quantum.custom "rz"(%f0) %q_1_0 : !quantum.bit
            // CHECK: scf.yield [[q100]]
            scf.yield %q_1_0_0 : !quantum.bit
        } else {
            // CHECK: [[s3:%.+]] = tensor.extract %arg3
            // CHECK-NEXT: [[r3:%.+]] = arith.addf [[s3]], [[e0]] : f64
            // CHECK-NEXT: [[q101:%.+]] = quantum.custom "rz"([[r3]]) [[q10]]
            %q_1_0_1 = quantum.custom "rz"(%f0) %q_1_0 : !quantum.bit
            // CHECK-NEXT: [[s4:%.+]] = tensor.extract %arg3
            // CHECK-NEXT: [[r4:%.+]] = arith.addf [[s4]], [[e0]] : f64
            // CHECK-NEXT: [[q102:%.+]] = quantum.custom "rz"([[r4]]) [[q101]]
            %q_1_0_2 = quantum.custom "rz"(%f0) %q_1_0_1 : !quantum.bit
            // CHECK: scf.yield [[q102]]
            scf.yield %q_1_0_2 : !quantum.bit
        }
        // CHECK: scf.yield [[q11]]
        scf.yield %q_1_1 : !quantum.bit
    } else {
        // CHECK: scf.yield [[q1]]
        scf.yield %q_1 : !quantum.bit
    }

    // Branching is only valid as long as each block is traversed at most once!
    cf.br ^exit

  ^exit:
    // CHECK: [[s5:%.+]] = tensor.extract %arg3
    // CHECK-NEXT: [[r5:%.+]] = arith.addf [[s5]], [[e0]] : f64
    // CHECK-NEXT: {{%.+}} = quantum.custom "rx"([[r5]]) [[q2]]
    %q_3 = quantum.custom "rx"(%f0) %q_2 : !quantum.bit

    %r_1 = quantum.insert %r_0[%idx], %q_3 : !quantum.reg, !quantum.bit
    quantum.dealloc %r_1 : !quantum.reg
    func.return %f0 : f64
}

func.func @gradCall1(%arg0: tensor<1xf64>, %b0: i1, %b1: i1) -> tensor<1xf64> {
    %0 = gradient.grad "auto" @structured_circuit(%arg0, %b0, %b1) : (tensor<1xf64>, i1, i1) -> tensor<1xf64>
    func.return %0 : tensor<1xf64>
}

// -----

// CHECK-LABEL: @loop_circuit.shifted(%arg0: tensor<1xf64>, %arg1: tensor<4xf64>, %arg2: tensor<2xindex>) -> f64
func.func @loop_circuit(%arg0: tensor<1xf64>) -> f64 attributes {qnode, diff_method = "parameter-shift"} {
    // CHECK-DAG: [[c0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[c1:%.+]] = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    // CHECK: [[e0:%.+]] = tensor.extract
    %f0 = tensor.extract %arg0[%c0] : tensor<1xf64>

    %idx = arith.constant 0 : i64
    %r_0 = quantum.alloc(1) : !quantum.reg
    // CHECK: [[q0:%.+]] = quantum.extract
    %q_0 = quantum.extract %r_0[%idx] : !quantum.reg -> !quantum.bit

    // CHECK-NEXT: [[s0:%.+]] = tensor.extract %arg1
    // CHECK-NEXT: [[r0:%.+]] = arith.addf [[s0]], [[e0]] : f64
    // CHECK-NEXT: [[q1:%.+]] = quantum.custom "rx"([[r0]]) [[q0]]
    %q_1 = quantum.custom "rx"(%f0) %q_0 : !quantum.bit

    %lb = arith.constant 0 : index
    %ub = arith.constant 10: index
    %st = arith.constant 1 : index

    // CHECK: [[q2:%.+]] = scf.for [[i:%.+]] = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[q10:%.+]] = [[q1]])
    %q_2 = scf.for %i = %lb to %ub step %st iter_args(%q_1_0 = %q_1) -> !quantum.bit {
        // CHECK: [[sel0:%.+]] = tensor.extract %arg2[[[c0]]]

        // CHECK: [[s1:%.+]] = tensor.extract %arg1
        // CHECK: [[cond1:%.+]] = arith.cmpi eq, [[i]], [[sel0]] : index
        // CHECK: [[r1:%.+]] = scf.if [[cond1]] -> (f64)
        // CHECK-NEXT: [[tmp:%.+]] = arith.addf [[s1]], [[e0]] : f64
        // CHECK-NEXT: scf.yield [[tmp]] : f64
        // CHECK: scf.yield [[e0]] : f64
        // CHECK: [[q11:%.+]] = quantum.custom "ry"([[r1]]) [[q10]]
        %q_1_1 = quantum.custom "ry"(%f0) %q_1_0 : !quantum.bit

        // CHECK: scf.yield [[q11]]
        scf.yield %q_1_1 : !quantum.bit
    }

    // CHECK: {{%.+}} = scf.for [[j:%.+]] = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[q20:%.+]] = [[q2]])
    %q_3 = scf.for %j = %lb to %ub step %st iter_args(%q_2_0 = %q_2) -> !quantum.bit {
        // CHECK: [[sel1:%.+]] = tensor.extract %arg2[[[c0]]]

        // CHECK: [[s2:%.+]] = tensor.extract %arg1
        // CHECK: [[cond2:%.+]] = arith.cmpi eq, [[j]], [[sel1]] : index
        // CHECK: [[r2:%.+]] = scf.if [[cond2]] -> (f64)
        // CHECK-NEXT: [[tmp:%.+]] = arith.addf [[s2]], [[e0]] : f64
        // CHECK-NEXT: scf.yield [[tmp]] : f64
        // CHECK: scf.yield [[e0]] : f64
        // CHECK: [[q21:%.+]] = quantum.custom "ry"([[r2]]) [[q20]]
        %q_2_1 = quantum.custom "ry"(%f0) %q_2_0 : !quantum.bit

        // CHECK: [[q11:%.+]] = scf.for [[k:%.+]] = [[j]] to {{%.+}} step {{%.+}} iter_args([[q210:%.+]] = [[q21]])
        %q_1_1 = scf.for %k = %j to %ub step %st iter_args(%q_2_1_0 = %q_2_1) -> !quantum.bit {
            // CHECK: [[sel2:%.+]] = tensor.extract %arg2[[[c1]]]

            // CHECK: [[s3:%.+]] = tensor.extract %arg1
            // CHECK: [[cond3:%.+]] = arith.cmpi eq, [[j]], [[sel1]] : index
            // CHECK: [[cond4:%.+]] = arith.cmpi eq, [[k]], [[sel2]] : index
            // CHECK: [[cond5:%.+]] = arith.andi [[cond3]], [[cond4]] : i1
            // CHECK: [[r3:%.+]] = scf.if [[cond5]] -> (f64)
            // CHECK-NEXT: [[tmp:%.+]] = arith.addf [[s3]], [[e0]] : f64
            // CHECK-NEXT: scf.yield [[tmp]] : f64
            // CHECK: scf.yield [[e0]] : f64
            // CHECK: [[q211:%.+]] = quantum.custom "rz"([[r3]]) [[q210]]
            %q_2_1_1 = quantum.custom "rz"(%f0) %q_2_1_0 : !quantum.bit

            // CHECK: scf.yield [[q211]]
            scf.yield %q_2_1_1 : !quantum.bit
        }

        // CHECK: scf.yield [[q11]]
        scf.yield %q_1_1 : !quantum.bit
    }

    %r_1 = quantum.insert %r_0[%idx], %q_3 : !quantum.reg, !quantum.bit
    quantum.dealloc %r_1 : !quantum.reg
    func.return %f0 : f64
}

func.func @gradCall2(%arg0: tensor<1xf64>) -> tensor<1xf64> {
    %0 = gradient.grad "auto" @loop_circuit(%arg0) : (tensor<1xf64>) -> tensor<1xf64>
    func.return %0 : tensor<1xf64>
}
