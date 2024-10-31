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

// RUN: quantum-opt --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL: test_alloc_dce
func.func @test_alloc_dce() {
    // CHECK-NOT: quantum.alloc
    %r = quantum.alloc(5) : !quantum.reg
    return
}

// CHECK-LABEL: test_alloc_cse
func.func @test_alloc_cse() -> (!quantum.reg, !quantum.reg){
    // CHECK: quantum.alloc
    // CHECK-NOT: quantum.alloc
    %r1 = quantum.alloc(4) : !quantum.reg
    %r2 = quantum.alloc(4) : !quantum.reg
    return %r1, %r2 : !quantum.reg, !quantum.reg
}

// CHECK-LABEL: test_alloc_dealloc_fold
func.func @test_alloc_dealloc_fold() {
    // CHECK-NOT: quantum.alloc
    // CHECK-NOT: quantum.dealloc
    %r = quantum.alloc(3) : !quantum.reg
    quantum.dealloc %r : !quantum.reg
    return
}

// CHECK-LABEL: test_alloc_dealloc_no_fold
func.func @test_alloc_dealloc_no_fold() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: quantum.dealloc
    %r = quantum.alloc(3) : !quantum.reg
    %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
    quantum.dealloc %r : !quantum.reg
    return %q : !quantum.bit
}

// CHECK-LABEL: test_extract_dce
func.func @test_extract_dce(%r: !quantum.reg) {
    // CHECK-NOT: quantum.extract
    %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
    return
}

// CHECK-LABEL: test_extract_insert_fold
func.func @test_extract_insert_fold(%r1: !quantum.reg, %i: i64) -> !quantum.reg {
    // CHECK-NOT: quantum.extract
    // CHECK-NOT: quantum.insert
    %q1 = quantum.extract %r1[0] : !quantum.reg -> !quantum.bit
    %r2 = quantum.insert %r1[0], %q1 : !quantum.reg, !quantum.bit

    // CHECK-NOT: quantum.extract
    // CHECK-NOT: quantum.insert
    %q2 = quantum.extract %r2[%i] : !quantum.reg -> !quantum.bit
    %r3 = quantum.insert %r2[%i], %q2 : !quantum.reg, !quantum.bit

    return %r3 : !quantum.reg
}

// CHECK-LABEL: test_extract_insert_no_fold_static
func.func @test_extract_insert_no_fold_static(%r1: !quantum.reg, %i1: i64, %i2: i64) -> !quantum.reg {
    // CHECK: quantum.extract
    // CHECK: quantum.insert
    %q1 = quantum.extract %r1[0] : !quantum.reg -> !quantum.bit
    %r2 = quantum.insert %r1[1], %q1 : !quantum.reg, !quantum.bit

    // CHECK: quantum.extract
    // CHECK: quantum.insert
    %q2 = quantum.extract %r2[0] : !quantum.reg -> !quantum.bit
    %r3 = quantum.insert %r2[%i1], %q2 : !quantum.reg, !quantum.bit

    // CHECK: quantum.extract
    // CHECK: quantum.insert
    %q3 = quantum.extract %r3[%i1] : !quantum.reg -> !quantum.bit
    %r4 = quantum.insert %r3[%i2], %q3 : !quantum.reg, !quantum.bit

    return %r4 : !quantum.reg
}

// CHECK-LABEL: test_extract_insert_constant
func.func @test_extract_insert_constant(%r1: !quantum.reg) -> !quantum.reg {
    // CHECK-NOT: arith.constant
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    // CHECK-NOT: arith.addi
    %add = arith.addi %c1, %c2 : i64

    // CHECK: quantum.extract %{{.*}}[ 3]
    %q1 = quantum.extract %r1[%add] : !quantum.reg -> !quantum.bit

    // CHECK: quantum.insert %{{.*}}[ 2]
    %r2 = quantum.insert %r1[%c2], %q1 : !quantum.reg, !quantum.bit
    return %r2 : !quantum.reg
}

// CHECK-LABEL: test_insert_canonicalize
func.func @test_insert_canonicalize(%r1: !quantum.reg, %i: i64) -> !quantum.bit {
    // CHECK:  quantum.extract
    %q1 = quantum.extract %r1[0] : !quantum.reg -> !quantum.bit
    // CHECK:  quantum.insert
    %r2 = quantum.insert %r1[0], %q1 : !quantum.reg, !quantum.bit
    %4 = quantum.custom "Hadamard"() %q1 : !quantum.bit
    // CHECK:  quantum.dealloc
    quantum.dealloc %r2 : !quantum.reg
    return %4 : !quantum.bit
}

// CHECK-LABEL: test_hermitian_adjoint_canonicalize
func.func @test_hermitian_adjoint_canonicalize() -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 {adjoint}: !quantum.bit
    // CHECK:  quantum.custom "Hadamard"() [[qubit]] : !quantum.bit
    return %2 : !quantum.bit
}

// CHECK-LABEL: test_rotation_adjoint_canonicalize
func.func @test_rotation_adjoint_canonicalize(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "RX"(%arg0) %1 {adjoint}: !quantum.bit
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK:  quantum.custom "RX"([[arg0neg]]) [[qubit]] : !quantum.bit
    return %2 : !quantum.bit
}

// CHECK-LABEL: test_multirz_adjoint_canonicalize
func.func @test_multirz_adjoint_canonicalize(%arg0: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[ret:%.+]]:2 = quantum.multirz([[arg0neg]]) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    %3:2 = quantum.multirz (%arg0) %1, %2 {adjoint} : !quantum.bit, !quantum.bit
    return %3#0, %3#1 : !quantum.bit, !quantum.bit
}

