// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --resolve-gate-level-adjoint %s | FileCheck %s

// CHECK-LABEL: test_hermitian_adjoint_canonicalize
func.func @test_hermitian_adjoint_canonicalize() {
    %0 = qref.alloc( 1) : !qref.reg<1>
    %1 = qref.get %0[ 0] : !qref.reg<1> -> !qref.bit
    // CHECK: [[reg:%.+]] = qref.alloc( 1) : !qref.reg<1>
    // CHECK: [[qubit:%.+]] = qref.get [[reg]][ 0] : !qref.reg<1> -> !qref.bit
    qref.custom "Hadamard"() %1 adj : !qref.bit
    // CHECK:  qref.custom "Hadamard"() [[qubit]] : !qref.bit
    return
}

// -----

// CHECK-LABEL: test_rotation_adjoint_canonicalize
func.func @test_rotation_adjoint_canonicalize(%arg0: f64) {
    %0 = qref.alloc( 1) : !qref.reg<1>
    %1 = qref.get %0[ 0] : !qref.reg<1> -> !qref.bit
    // CHECK: [[reg:%.+]] = qref.alloc( 1) : !qref.reg<1>
    // CHECK: [[qubit:%.+]] = qref.get [[reg]][ 0] : !qref.reg<1> -> !qref.bit
    qref.custom "RX"(%arg0) %1 adj : !qref.bit
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK:  qref.custom "RX"([[arg0neg]]) [[qubit]] : !qref.bit
    return
}

// -----

// CHECK-LABEL: test_multirz_adjoint_canonicalize
func.func @test_multirz_adjoint_canonicalize(%arg0: f64) {
    // CHECK: [[reg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[qubit1:%.+]] = qref.get [[reg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[qubit2:%.+]] = qref.get [[reg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: qref.multirz([[arg0neg]]) [[qubit1]], [[qubit2]] : !qref.bit, !qref.bit
    qref.multirz (%arg0) %1, %2 adj : !qref.bit, !qref.bit
    return
}

// -----

// CHECK-LABEL: test_pcphase_adjoint_canonicalize
func.func @test_pcphase_adjoint_canonicalize(%arg0: f64) {
    // CHECK: [[reg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[qubit1:%.+]] = qref.get [[reg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[qubit2:%.+]] = qref.get [[reg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = qref.alloc( 2) : !qref.reg<2>
    %1 = qref.get %0[ 0] : !qref.reg<2> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: qref.pcphase([[arg0neg]], dim : 2) [[qubit1]], [[qubit2]] : !qref.bit, !qref.bit
    qref.pcphase(%arg0, dim : 2) %1, %2 adj : !qref.bit, !qref.bit
    return
}

// -----

// CHECK-LABEL: test_paulirot_adjoint_canonicalize
func.func @test_paulirot_adjoint_canonicalize(%arg0: f64) {
    // CHECK: [[reg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[qubit1:%.+]] = qref.get [[reg]][ 0] : !qref.reg<3> -> !qref.bit
    // CHECK: [[qubit2:%.+]] = qref.get [[reg]][ 1] : !qref.reg<3> -> !qref.bit
    // CHECK: [[qubit3:%.+]] = qref.get [[reg]][ 2] : !qref.reg<3> -> !qref.bit
    %0 = qref.alloc( 3) : !qref.reg<3>
    %1 = qref.get %0[ 0] : !qref.reg<3> -> !qref.bit
    %2 = qref.get %0[ 1] : !qref.reg<3> -> !qref.bit
    %3 = qref.get %0[ 2] : !qref.reg<3> -> !qref.bit

    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: qref.paulirot ["X", "Y", "Z"]([[arg0neg]]) [[qubit1]], [[qubit2]], [[qubit3]] : !qref.bit, !qref.bit, !qref.bit
    qref.paulirot ["X", "Y", "Z"](%arg0) %1, %2, %3 adj : !qref.bit, !qref.bit, !qref.bit
    return
}

// -----

// CHECK-LABEL: test_gphase_adjoint_canonicalize
func.func @test_gphase_adjoint_canonicalize(%arg0: f64) {
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: qref.gphase([[arg0neg]])
    qref.gphase(%arg0) adj
    return
}
