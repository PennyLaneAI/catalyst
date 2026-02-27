// Copyright 2026 Xanadu qref Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --canonicalize --cse --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: test_alloc_dce
func.func @test_alloc_dce() {
    // CHECK-NOT: qref.alloc
    %r = qref.alloc(5) : !qref.reg<5>
    return
}

// -----

// CHECK-LABEL: test_alloc_no_cse
func.func @test_alloc_no_cse() -> (!qref.reg<4>, !qref.reg<4>){
    // CHECK: qref.alloc
    // CHECK-NEXT: qref.alloc
    %r1 = qref.alloc(4) : !qref.reg<4>
    %r2 = qref.alloc(4) : !qref.reg<4>
    return %r1, %r2 : !qref.reg<4>, !qref.reg<4>
}

// -----

// CHECK-LABEL: test_alloc_dealloc_fold
func.func @test_alloc_dealloc_fold() {
    // CHECK-NOT: qref.alloc
    // CHECK-NOT: qref.dealloc
    %r = qref.alloc(3) : !qref.reg<3>
    qref.dealloc %r : !qref.reg<3>
    return
}

// -----

// CHECK-LABEL: test_alloc_dealloc_no_fold
func.func @test_alloc_dealloc_no_fold() -> !qref.bit {
    // CHECK: qref.alloc
    // CHECK: qref.dealloc
    %r = qref.alloc(3) : !qref.reg<3>
    %q = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    qref.dealloc %r : !qref.reg<3>
    return %q : !qref.bit<3>
}

// -----

// CHECK-LABEL: test_get_no_user_fold
func.func @test_get_no_user_fold(%r: !qref.reg<3>) {
    // CHECK-NOT: qref.get
    %q = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    return
}

// -----

// CHECK-LABEL: test_get_cse
func.func @test_get_cse(%r: !qref.reg<3>, %i: i64) {
    // CHECK: [[q0:%.+]] = qref.get %arg0[ 0] : !qref.reg<3> -> !qref.bit
    // CHECK-NOT: qref.get
    // qref.custom "Hadamard"() [[q0]] : !qref.bit
    // qref.custom "Hadamard"() [[q0]] : !qref.bit
    %q0 = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    %q0_dup = qref.get %r[0] : !qref.reg<3> -> !qref.bit
    qref.custom "Hadamard"() %q0 : !qref.bit
    qref.custom "Hadamard"() %q0_dup : !qref.bit

    // CHECK: [[q1:%.+]] = qref.get %arg0[%arg1] : !qref.reg<3>, i64 -> !qref.bit
    // CHECK-NOT: qref.get
    // qref.custom "Hadamard"() [[q1]] : !qref.bit
    // qref.custom "Hadamard"() [[q1]] : !qref.bit
    %q1 = qref.get %r[%i] : !qref.reg<3>, i64 -> !qref.bit
    %q1_dup = qref.get %r[%i] : !qref.reg<3>, i64 -> !qref.bit
    qref.custom "Hadamard"() %q1 : !qref.bit
    qref.custom "Hadamard"() %q1_dup : !qref.bit

    return
}

// -----

// CHECK-LABEL: test_hermitian_adjoint_canonicalize
func.func @test_hermitian_adjoint_canonicalize(%q0: !qref.bit) {
    // CHECK:  qref.custom "Hadamard"() %arg0 : !qref.bit
    qref.custom "Hadamard"() %q0 adj: !qref.bit
    return
}

// -----

// CHECK-LABEL: test_rotation_adjoint_canonicalize
func.func @test_rotation_adjoint_canonicalize(%arg0: f64, %q0: !qref.bit) {
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK:  qref.custom "RX"([[arg0neg]]) %arg1 : !qref.bit
    qref.custom "RX"(%arg0) %q0 adj: !qref.bit
    return
}

// -----

// CHECK-LABEL: test_multirz_adjoint_canonicalize
func.func @test_multirz_adjoint_canonicalize(%arg0: f64, %q0: !qref.bit, %q1: !qref.bit) {
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: qref.multirz([[arg0neg]]) %arg1, %arg2 : !qref.bit, !qref.bit
    qref.multirz (%arg0) %q0, %q1 adj : !qref.bit, !qref.bit
    return
}

// -----

// CHECK-LABEL: test_pcphase_adjoint_canonicalize
func.func @test_pcphase_adjoint_canonicalize(%arg0: f64, %dim: f64, %q0: !qref.bit, %q1: !qref.bit) {
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: qref.pcphase([[arg0neg]], %arg1) %arg2, %arg3 : !qref.bit, !qref.bit
    qref.pcphase (%arg0, %dim) %q0, %q1 adj : !qref.bit, !qref.bit
    return
}

// -----

// Unlike the value semantics quantum dialect, in reference semantics, gates do not produce output
// qubit values, and will consequently have no users.
// We must make sure that they are not removed by DCE.

// CHECK-LABEL: test_canonicalize_no_dce
func.func @test_canonicalize_no_dce(%arg0: tensor<2xcomplex<f64>>, %arg1 : tensor<1xi1>, %arg2: f64,
     %arg3 : tensor<2x2xcomplex<f64>>, %q0: !qref.bit, %q1: !qref.bit, %r: !qref.reg<2>) {

    // CHECK: qref.set_state
    qref.set_state(%arg0) %q0 : tensor<2xcomplex<f64>>, !qref.bit

    // CHECK: qref.set_basis_state
    qref.set_basis_state(%arg1) %q0 : tensor<1xi1>, !qref.bit

    // CHECK: qref.custom "Hadamard"
    qref.custom "Hadamard"() %q0 : !qref.bit

    // CHECK: qref.paulirot
    qref.paulirot ["Z"](%arg2) %q0 : !qref.bit

    // CHECK: qref.gphase
    qref.gphase(%arg2) : f64

    // CHECK: qref.multirz
    qref.multirz (%arg2) %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: qref.pcphase
    qref.pcphase (%arg2, %arg2) %q0 : !qref.bit

    // CHECK: qref.unitary
    qref.unitary (%arg3 : tensor<2x2xcomplex<f64>>) %q0 : !qref.bit

    // CHECK: qref.adjoint
    // CHECK:   qref.get
    // CHECK:   qref.custom "Hadamard"
    qref.adjoint(%r) : !qref.reg<2> {
    ^bb0(%r0: !qref.reg<2>):
        %q = qref.get %r0[0] : !qref.reg<2> -> !qref.bit
        qref.custom "Hadamard"() %q : !qref.bit
    }

    return
}
