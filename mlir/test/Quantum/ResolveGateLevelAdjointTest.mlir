// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --resolve-gate-level-adjoint --split-input-file %s | FileCheck %s

// CHECK-LABEL: test_hermitian_adjoint_canonicalize
func.func @test_hermitian_adjoint_canonicalize() -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 adj : !quantum.bit
    // CHECK:  quantum.custom "Hadamard"() [[qubit]] : !quantum.bit
    return %2 : !quantum.bit
}

// -----

// CHECK-LABEL: test_rotation_adjoint_canonicalize
func.func @test_rotation_adjoint_canonicalize(%arg0: f64) -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "RX"(%arg0) %1 adj : !quantum.bit
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK:  quantum.custom "RX"([[arg0neg]]) [[qubit]] : !quantum.bit
    return %2 : !quantum.bit
}

// -----

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
    %3:2 = quantum.multirz (%arg0) %1, %2 adj  : !quantum.bit, !quantum.bit
    return %3#0, %3#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_pcphase_adjoint_canonicalize
func.func @test_pcphase_adjoint_canonicalize(%arg0: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[ret:%.+]]:2 = quantum.pcphase([[arg0neg]], dim : 2) [[qubit1]], [[qubit2]] : !quantum.bit, !quantum.bit
    %3:2 = quantum.pcphase(%arg0, dim : 2) %1, %2 adj  : !quantum.bit, !quantum.bit
    return %3#0, %3#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_paulirot_adjoint_canonicalize
func.func @test_paulirot_adjoint_canonicalize(%arg0: f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    // CHECK: [[reg:%.+]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: [[qubit1:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit2:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit3:%.+]] = quantum.extract [[reg]][ 2] : !quantum.reg -> !quantum.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: [[ret:%.+]]:3 = quantum.paulirot ["X", "Y", "Z"]([[arg0neg]]) [[qubit1]], [[qubit2]], [[qubit3]] : !quantum.bit, !quantum.bit, !quantum.bit
    %4:3 = quantum.paulirot ["X", "Y", "Z"](%arg0) %1, %2, %3 adj  : !quantum.bit, !quantum.bit, !quantum.bit
    return %4#0, %4#1, %4#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_gphase_adjoint_canonicalize
func.func @test_gphase_adjoint_canonicalize(%arg0: f64) {
    // CHECK: [[arg0neg:%.+]] = arith.negf %arg0 : f64
    // CHECK: quantum.gphase([[arg0neg]])
    quantum.gphase(%arg0) adj 
    return
}
