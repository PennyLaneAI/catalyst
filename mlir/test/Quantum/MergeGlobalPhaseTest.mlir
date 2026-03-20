// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --combine-global-phases %s | FileCheck %s

// CHECK-LABEL: func.func @merge_simple_global_phases(
func.func @merge_simple_global_phases(%arg0: f64, %arg1: f64) {
    // CHECK-SAME: [[A:%.+]]: f64, [[B:%.+]]: f64

    // CHECK-NOT: quantum.gphase
    // CHECK: [[SUM:%.+]] = arith.addf [[B]], [[A]]
    // CHECK: quantum.gphase([[SUM]])
    // CHECK-NOT: quantum.gphase
    quantum.gphase(%arg0)
    quantum.gphase(%arg1)

    return
}

// -----

// CHECK-LABEL: func.func @dont_rewrite_single_global_phase(
func.func @dont_rewrite_single_global_phase(%arg0: f64) {
    // CHECK-SAME: [[A:%.+]]: f64

    // CHECK-NOT: arith.addf
    // CHECK: quantum.gphase([[A]])
    quantum.gphase(%arg0)
    return
}

// -----

// CHECK-LABEL: func.func @keep_controlled_global_phases(
func.func @keep_controlled_global_phases(%arg0: f64, %arg1: f64, %arg2: f64, %q: !quantum.bit) {
    // CHECK-SAME: [[A:%.+]]: f64, [[B:%.+]]: f64, [[C:%.+]]: f64, [[Q:%.+]]: !quantum.bit

    // CHECK-NOT: quantum.gphase
    // CHECK: quantum.gphase([[B]]) ctrls([[Q]])
    // CHECK-NOT: quantum.gphase
    // CHECK: [[SUM:%.+]] = arith.addf [[C]], [[A]]
    // CHECK: quantum.gphase([[SUM]])
    // CHECK-NOT: quantum.gphase
    quantum.gphase(%arg0)
    %true = arith.constant 1 : i1
    quantum.gphase(%arg1) ctrls(%q) ctrlvals(%true) : ctrls !quantum.bit
    quantum.gphase(%arg2)

    return
}

// -----

// CHECK-LABEL: func.func @merge_global_phases_in_scf_if(
func.func @merge_global_phases_in_scf_if(%cond: i1, %arg0: f64, %arg1: f64) {
    // CHECK-SAME: [[COND:%.+]]: i1, [[A:%.+]]: f64, [[B:%.+]]: f64

    // CHECK: [[IF_RES:%.+]] = scf.if [[COND]] -> (f64) {
    %ret = scf.if %cond -> (f64) {
        // CHECK-NOT: quantum.gphase
        // CHECK: [[THEN_SUM:%.+]] = arith.addf [[A]], [[A]]
        // CHECK: quantum.gphase([[THEN_SUM]])
        // CHECK-NOT: quantum.gphase
        quantum.gphase(%arg0)
        quantum.gphase(%arg0)

        scf.yield %arg0 : f64
    // CHECK: else
    } else {
        // CHECK-NOT: quantum.gphase
        // CHECK: [[ELSE_SUM:%.+]] = arith.addf [[B]], [[B]]
        // CHECK: quantum.gphase([[ELSE_SUM]])
        // CHECK-NOT: quantum.gphase
        quantum.gphase(%arg1)
        quantum.gphase(%arg1)

        scf.yield %arg1 : f64
    }

    // CHECK-NOT: quantum.gphase
    // CHECK: [[OUT_SUM:%.+]] = arith.addf [[IF_RES]], [[A]]
    // CHECK: quantum.gphase([[OUT_SUM]])
    // CHECK-NOT: quantum.gphase
    quantum.gphase(%arg0)
    quantum.gphase(%ret)

    return
}

// -----

// CHECK-LABEL: func.func @merge_global_phases_data_dep(
func.func @merge_global_phases_data_dep(%arg0: f64, %arg1: f64) {
    // CHECK-SAME: [[A:%.+]]: f64, [[B:%.+]]: f64

    // CHECK-NOT: quantum.gphase
    // CHECK: [[CONST:%.+]] = arith.mulf
    // CHECK-NOT: quantum.gphase
    // CHECK: [[OUT_SUM:%.+]] = arith.addf [[CONST]], [[A]]
    // CHECK: quantum.gphase([[OUT_SUM]])
    // CHECK-NOT: quantum.gphase
    quantum.gphase(%arg0)
    %c1 = arith.constant 2.0 : f64
    %c2 = arith.mulf %arg1, %c1 : f64
    quantum.gphase(%c2)

    return
}

// -----

// CHECK-LABEL: func.func @merge_global_phases_adjoint(
func.func @merge_global_phases_adjoint(%arg0: f64, %arg1: f64, %arg2: f64) {
    // CHECK-SAME: [[A:%.+]]: f64, [[B:%.+]]: f64, [[C:%.+]]: f64

    // CHECK-NOT: quantum.gphase
    // CHECK: [[NEG:%.+]] = arith.negf [[C]]
    // CHECK-NOT: quantum.gphase
    // CHECK: [[SUM1:%.+]] = arith.subf [[NEG]], [[A]]
    // CHECK-NOT: quantum.gphase
    // CHECK: [[SUM2:%.+]] = arith.addf [[SUM1]], [[B]]
    // CHECK: quantum.gphase([[SUM2]])
    // CHECK-NOT: quantum.gphase
    quantum.gphase(%arg0) adj
    quantum.gphase(%arg1)
    quantum.gphase(%arg2) adj

    return
}
