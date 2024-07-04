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

// RUN: quantum-opt --remove-chained-self-inverse --split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: test_chained_self_inverse_from_block_arg_Hadamard
func.func @test_chained_self_inverse_from_block_arg_Hadamard(%arg: !quantum.bit) -> !quantum.bit {
    // CHECK-NOT: quantum.custom
    %0 = quantum.custom "Hadamard"() %arg : !quantum.bit
    %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
    return %1 : !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_Hadamard
func.func @test_chained_self_inverse_Hadamard() -> !quantum.bit {
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

// CHECK-LABEL: test_chained_self_inverse_PauliX
func.func @test_chained_self_inverse_PauliX() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "PauliX"() %1 : !quantum.bit
    %3 = quantum.custom "PauliX"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_PauliY
func.func @test_chained_self_inverse_PauliY() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "PauliY"() %1 : !quantum.bit
    %3 = quantum.custom "PauliY"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_PauliZ
func.func @test_chained_self_inverse_PauliZ() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %2 = quantum.custom "PauliZ"() %1 : !quantum.bit
    %3 = quantum.custom "PauliZ"() %2 : !quantum.bit
    return %3 : !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_CNOT
func.func @test_chained_self_inverse_CNOT() -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %3:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CNOT"() %3#0, %3#1 : !quantum.bit, !quantum.bit
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_CNOT_wrong_order
func.func @test_chained_self_inverse_CNOT_wrong_order() -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %3:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CNOT"() %3#1, %3#0 : !quantum.bit, !quantum.bit
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_CY
func.func @test_chained_self_inverse_CY() -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %3:2 = quantum.custom "CY"() %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CY"() %3#0, %3#1 : !quantum.bit, !quantum.bit
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_CY_wrong_order
func.func @test_chained_self_inverse_CY_wrong_order() -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %3:2 = quantum.custom "CY"() %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CY"() %3#1, %3#0 : !quantum.bit, !quantum.bit
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_CZ
func.func @test_chained_self_inverse_CZ() -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %3:2 = quantum.custom "CZ"() %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CZ"() %3#0, %3#1 : !quantum.bit, !quantum.bit
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_CZ_wrong_order
func.func @test_chained_self_inverse_CZ_wrong_order() -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %3:2 = quantum.custom "CZ"() %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "CZ"() %3#1, %3#0 : !quantum.bit, !quantum.bit
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_SWAP
func.func @test_chained_self_inverse_SWAP() -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %3:2 = quantum.custom "SWAP"() %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "SWAP"() %3#0, %3#1 : !quantum.bit, !quantum.bit
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_SWAP_wrong_order
func.func @test_chained_self_inverse_SWAP_wrong_order() -> (!quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %3:2 = quantum.custom "SWAP"() %1, %2 : !quantum.bit, !quantum.bit
    %4:2 = quantum.custom "SWAP"() %3#1, %3#0 : !quantum.bit, !quantum.bit
    return %4#0, %4#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_Toffoli
func.func @test_chained_self_inverse_Toffoli() -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    // CHECK-NOT: quantum.custom
    %4:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %5:3 = quantum.custom "Toffoli"() %4#0, %4#1, %4#2 : !quantum.bit, !quantum.bit, !quantum.bit
    return %5#0, %5#1, %5#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_chained_self_inverse_Toffoli_wrong_order
func.func @test_chained_self_inverse_Toffoli_wrong_order() -> (!quantum.bit, !quantum.bit, !quantum.bit, 
!quantum.bit, !quantum.bit, !quantum.bit, 
!quantum.bit, !quantum.bit, !quantum.bit,
!quantum.bit, !quantum.bit, !quantum.bit,
!quantum.bit, !quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    // CHECK: quantum.extract
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %4:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %5:3 = quantum.custom "Toffoli"() %4#0, %4#2, %4#1 : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %6:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %7:3 = quantum.custom "Toffoli"() %6#1, %6#0, %6#2 : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %8:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %9:3 = quantum.custom "Toffoli"() %8#1, %8#2, %8#0 : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %10:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %11:3 = quantum.custom "Toffoli"() %10#2, %10#0, %10#1 : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: quantum.custom
    // CHECK: quantum.custom
    %12:3 = quantum.custom "Toffoli"() %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %13:3 = quantum.custom "Toffoli"() %12#2, %12#1, %12#0 : !quantum.bit, !quantum.bit, !quantum.bit

    // return all the results so quantum-opt does not fold them away
    return %5#0, %5#1, %5#2, %7#0, %7#1, %7#2, %9#0, %9#1, %9#2, 
    %11#0, %11#1, %11#2, %13#0, %13#1, %13#2 : 
    !quantum.bit, !quantum.bit, !quantum.bit, 
    !quantum.bit, !quantum.bit, !quantum.bit, 
    !quantum.bit, !quantum.bit, !quantum.bit,
    !quantum.bit, !quantum.bit, !quantum.bit,
    !quantum.bit, !quantum.bit, !quantum.bit
}