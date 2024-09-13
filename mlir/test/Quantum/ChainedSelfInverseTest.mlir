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

// RUN: quantum-opt --pass-pipeline="builtin.module(remove-chained-self-inverse{func-name=test_chained_self_inverse})" --split-input-file -verify-diagnostics %s | FileCheck %s

// test chained Hadamard
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %3 = quantum.custom "Hadamard"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

// test chained Hadamard from block arg
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse(%arg: !quantum.bit) -> !quantum.bit {
    // CHECK-NOT: quantum.custom
    %0 = quantum.custom "Hadamard"() %arg : !quantum.bit
    %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
    return %1 : !quantum.bit
}

// -----

// Test for chained PauliX
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "PauliX"() %1 : !quantum.bit
    %3 = quantum.custom "PauliX"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

// Test for chained PauliY
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "PauliY"() %1 : !quantum.bit
    %3 = quantum.custom "PauliY"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

// Test for chained PauliZ
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "PauliZ"() %1 : !quantum.bit
    %3 = quantum.custom "PauliZ"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

// Test for chained CNOT
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
    // CHECK: return %1, %2 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
}

// -----

// Test for chained CY
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %out_qubits:2 = quantum.custom "CY"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CY"() %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
    // CHECK: return %1, %2 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
}

// -----

// Test for chained CZ
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %out_qubits:2 = quantum.custom "CZ"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CZ"() %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
    // CHECK: return %1, %2 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
}

// -----

// Test for chained SWAP
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %out_qubits:2 = quantum.custom "SWAP"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "SWAP"() %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
    // CHECK: return %1, %2 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
}

// -----

// Test for chained Toffoli
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    %0 = quantum.alloc(3) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[2] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %out_qubits:3 = quantum.custom "Toffoli"() %1, %2, %3: !quantum.bit, !quantum.bit, !quantum.bit
    %out_qubits_0:3 = quantum.custom "Toffoli"() %out_qubits#0, %out_qubits#1, %out_qubits#2 : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: return %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit

    return %out_qubits_0#0, %out_qubits_0#1, %out_qubits_0#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// Test for chained CNOT with wrong order
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    // CHECK-NEXT: quantum.custom "CNOT"() %out_qubits#1, %out_qubits#0 : !quantum.bit, !quantum.bit
    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#1, %out_qubits#0 : !quantum.bit, !quantum.bit
    // CHECK: %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
}

// -----

// Test for chained CY with wrong order
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom "CY"() %1, %2 : !quantum.bit, !quantum.bit
    // CHECK-NEXT: quantum.custom "CY"() %out_qubits#1, %out_qubits#0 : !quantum.bit, !quantum.bit
    %out_qubits:2 = quantum.custom "CY"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CY"() %out_qubits#1, %out_qubits#0 : !quantum.bit, !quantum.bit
    // CHECK: return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
}

// -----

// Test for chained CZ with wrong order
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom "CZ"() %1, %2 : !quantum.bit, !quantum.bit
    // CHECK-NEXT: quantum.custom "CZ"() %out_qubits#1, %out_qubits#0 : !quantum.bit, !quantum.bit
    %out_qubits:2 = quantum.custom "CZ"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CZ"() %out_qubits#1, %out_qubits#0 : !quantum.bit, !quantum.bit
    // CHECK: return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
}

// -----

// Test for chained SWAP with wrong order
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom "SWAP"() %1, %2 : !quantum.bit, !quantum.bit
    // CHECK-NEXT: quantum.custom "SWAP"() %out_qubits#1, %out_qubits#0 : !quantum.bit, !quantum.bit
    %out_qubits:2 = quantum.custom "SWAP"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "SWAP"() %out_qubits#1, %out_qubits#0 : !quantum.bit, !quantum.bit
    // CHECK: return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
}

// -----

// Test for chained Toffoli with wrong order
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    %0 = quantum.alloc(3) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[2] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK-NEXT: quantum.custom "Toffoli"() %out_qubits#1, %out_qubits#2, %out_qubits#0 : !quantum.bit, !quantum.bit, !quantum.bit
    %out_qubits:3 = quantum.custom "Toffoli"() %1, %2, %3: !quantum.bit, !quantum.bit, !quantum.bit
    %out_qubits_0:3 = quantum.custom "Toffoli"() %out_qubits#1, %out_qubits#2, %out_qubits#0 : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: return %out_qubits_0#0, %out_qubits_0#1, %out_qubits_0#2 : !quantum.bit, !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1, %out_qubits_0#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// test nested self-inverse Gates with different names
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %3 = quantum.custom "PauliX"() %2 : !quantum.bit
    %4 = quantum.custom "PauliX"() %3 : !quantum.bit
    %5 = quantum.custom "Hadamard"() %4 : !quantum.bit
    return %5 : !quantum.bit
}