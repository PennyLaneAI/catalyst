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

// RUN: quantum-opt --adjoint-lowering-canonicalization %s | FileCheck %s

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
