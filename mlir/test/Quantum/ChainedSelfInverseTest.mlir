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

// RUN: quantum-opt --pass-pipeline="builtin.module(remove-chained-self-inverse)" --split-input-file -verify-diagnostics %s | FileCheck %s

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
// -----

// test non-consecutive self-inverse Gates are not canceled out
// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[VAL1:%.*]] = quantum.custom "Hadamard"() %1 : !quantum.bit
    // CHECK: [[VAL2:%.*]] = quantum.custom "PauliX"() [[VAL1:%.*]] : !quantum.bit
    // CHECK: quantum.custom "Hadamard"() [[VAL2:%.*]] : !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %3 = quantum.custom "PauliX"() %2 : !quantum.bit
    %4 = quantum.custom "Hadamard"() %3 : !quantum.bit
    return %4 : !quantum.bit
}


// -----


// test quantum.unitary labeled with adjoint attribute

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse(%arg0: tensor<2x2xf64>, %arg1: tensor<f64>) -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: [[IN:%.+]] = quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    %2 = stablehlo.convert %arg0 : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>
    %out_qubits = quantum.unitary(%2 : tensor<2x2xcomplex<f64>>) %1 : !quantum.bit
    %3 = stablehlo.convert %arg0 : (tensor<2x2xf64>) -> tensor<2x2xcomplex<f64>>
    %out_qubits_1 = quantum.unitary(%3 : tensor<2x2xcomplex<f64>>) %out_qubits adj : !quantum.bit

    // CHECK-NOT: quantum.unitary
    // CHECK: return [[IN]]
    return %out_qubits_1 : !quantum.bit
}


// -----


// test quantum.custom labeled with adjoint attribute

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse(%arg0: tensor<f64>) -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: [[IN:%.+]] = quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted_0) %1 : !quantum.bit
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RX"(%extracted_1) %out_qubits adj : !quantum.bit


    %out_qubits_2 = quantum.custom "RX"(%extracted_0) %out_qubits_1 adj : !quantum.bit
    %out_qubits_3 = quantum.custom "RX"(%extracted_1) %out_qubits_2 : !quantum.bit

    // CHECK-NOT: quantum.custom
    // CHECK: return [[IN]]
    return %out_qubits_3 : !quantum.bit
}


// -----


// test quantum.custom labeled both with adjoints

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse(%arg0: tensor<f64>) -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: [[IN:%.+]] = quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted_0) %1 adj : !quantum.bit
    %extracted_1 = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RX"(%extracted_1) %out_qubits adj : !quantum.bit

    // CHECK: quantum.custom
    return %out_qubits_1 : !quantum.bit
}


// -----


// test with explicit rotation angles

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    // CHECK: quantum.alloc
    // CHECK: [[IN:%.+]] = quantum.extract
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    %cst_0 = stablehlo.constant dense<1.234000e+01> : tensor<f64>
    %extracted_0 = tensor.extract %cst_0[] : tensor<f64>
    %out_qubits_0 = quantum.custom "RY"(%extracted_0) %1 adj : !quantum.bit

    %cst_1 = stablehlo.constant dense<1.234000e+01> : tensor<f64>
    %extracted_1 = tensor.extract %cst_1[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RY"(%extracted_1) %out_qubits_0 : !quantum.bit

    // CHECK-NOT: quantum.custom
    // CHECK: return [[IN]]
    return %out_qubits_1 : !quantum.bit
}


// -----


// test with unmatched explicit rotation angles

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> !quantum.bit {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    %cst_0 = stablehlo.constant dense<1.234000e+01> : tensor<f64>
    %extracted_0 = tensor.extract %cst_0[] : tensor<f64>
    %out_qubits_0 = quantum.custom "RY"(%extracted_0) %1 adj : !quantum.bit

    %cst_1 = stablehlo.constant dense<5.678000e+01> : tensor<f64>
    %extracted_1 = tensor.extract %cst_1[] : tensor<f64>
    %out_qubits_1 = quantum.custom "RY"(%extracted_1) %out_qubits_0 : !quantum.bit

    return %out_qubits_1 : !quantum.bit
}

// CHECK: quantum.custom "RY"{{.+}}adj
// CHECK: quantum.custom "RY"


// -----


// test with matched control wires

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %cst_0 = llvm.mlir.constant (9.000000e-01 : f64) : f64
    %cst_1 = llvm.mlir.constant (3.000000e-01 : f64) : f64

    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract {{.+}}[ 0]
    // CHECK: [[IN1:%.+]] = quantum.extract {{.+}}[ 1]
    // CHECK: [[IN2:%.+]] = quantum.extract {{.+}}[ 2]
    // CHECK: [[IN3:%.+]] = quantum.extract {{.+}}[ 3]
    %reg = quantum.alloc( 4) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %reg[ 3] : !quantum.reg -> !quantum.bit

    %out_qubits:2, %out_ctrl_qubits:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %0, %1 ctrls(%2, %3) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits_1:2, %out_ctrl_qubits_1:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %out_qubits#0, %out_qubits#1 adj ctrls(%out_ctrl_qubits#0, %out_ctrl_qubits#1) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.custom
    // CHECK: return [[IN0]], [[IN1]], [[IN2]], [[IN3]]
    return %out_qubits_1#0, %out_qubits_1#1, %out_ctrl_qubits_1#0, %out_ctrl_qubits_1#1 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}


// -----


// test with unmatched operation wires

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %cst_0 = llvm.mlir.constant (9.000000e-01 : f64) : f64
    %cst_1 = llvm.mlir.constant (3.000000e-01 : f64) : f64

    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract {{.+}}[ 0]
    // CHECK: [[IN1:%.+]] = quantum.extract {{.+}}[ 1]
    // CHECK: [[IN2:%.+]] = quantum.extract {{.+}}[ 2]
    // CHECK: [[IN3:%.+]] = quantum.extract {{.+}}[ 3]
    %reg = quantum.alloc( 4) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %reg[ 3] : !quantum.reg -> !quantum.bit

    // CHECK: quantum.custom
    %out_qubits:2, %out_ctrl_qubits:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %0, %1 ctrls(%2, %3) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits_1:2, %out_ctrl_qubits_1:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %out_qubits#1, %out_qubits#0 adj ctrls(%out_ctrl_qubits#0, %out_ctrl_qubits#1) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit


    return %out_qubits_1#0, %out_qubits_1#1, %out_ctrl_qubits_1#0, %out_ctrl_qubits_1#1 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}


// -----


// test with unmatched control wires

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %cst_0 = llvm.mlir.constant (9.000000e-01 : f64) : f64
    %cst_1 = llvm.mlir.constant (3.000000e-01 : f64) : f64

    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract {{.+}}[ 0]
    // CHECK: [[IN1:%.+]] = quantum.extract {{.+}}[ 1]
    // CHECK: [[IN2:%.+]] = quantum.extract {{.+}}[ 2]
    // CHECK: [[IN3:%.+]] = quantum.extract {{.+}}[ 3]
    %reg = quantum.alloc( 4) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %reg[ 3] : !quantum.reg -> !quantum.bit

    // CHECK: quantum.custom
    %out_qubits:2, %out_ctrl_qubits:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %0, %1 ctrls(%2, %3) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits_1:2, %out_ctrl_qubits_1:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %out_qubits#0, %out_qubits#1 adj ctrls(%out_ctrl_qubits#1, %out_ctrl_qubits#0) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit


    return %out_qubits_1#0, %out_qubits_1#1, %out_ctrl_qubits_1#0, %out_ctrl_qubits_1#1 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}


// -----


// test with unmatched control values

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %cst_0 = llvm.mlir.constant (9.000000e-01 : f64) : f64
    %cst_1 = llvm.mlir.constant (3.000000e-01 : f64) : f64

    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract {{.+}}[ 0]
    // CHECK: [[IN1:%.+]] = quantum.extract {{.+}}[ 1]
    // CHECK: [[IN2:%.+]] = quantum.extract {{.+}}[ 2]
    // CHECK: [[IN3:%.+]] = quantum.extract {{.+}}[ 3]
    %reg = quantum.alloc( 4) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %reg[ 3] : !quantum.reg -> !quantum.bit

    // CHECK: quantum.custom
    %out_qubits:2, %out_ctrl_qubits:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %0, %1 ctrls(%2, %3) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits_1:2, %out_ctrl_qubits_1:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %out_qubits#0, %out_qubits#1 adj ctrls(%out_ctrl_qubits#0, %out_ctrl_qubits#1) ctrlvals(%false, %true) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit


    return %out_qubits_1#0, %out_qubits_1#1, %out_ctrl_qubits_1#0, %out_ctrl_qubits_1#1 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}


// -----


// test with params in the wrong order

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1
    %cst = llvm.mlir.constant (6.000000e-01 : f64) : f64
    %cst_0 = llvm.mlir.constant (9.000000e-01 : f64) : f64
    %cst_1 = llvm.mlir.constant (3.000000e-01 : f64) : f64

    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract {{.+}}[ 0]
    // CHECK: [[IN1:%.+]] = quantum.extract {{.+}}[ 1]
    // CHECK: [[IN2:%.+]] = quantum.extract {{.+}}[ 2]
    // CHECK: [[IN3:%.+]] = quantum.extract {{.+}}[ 3]
    %reg = quantum.alloc( 4) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %reg[ 3] : !quantum.reg -> !quantum.bit

    // CHECK: quantum.custom
    %out_qubits:2, %out_ctrl_qubits:2 = quantum.custom "Rot"(%cst, %cst_0, %cst_1) %0, %1 ctrls(%2, %3) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits_1:2, %out_ctrl_qubits_1:2 = quantum.custom "Rot"(%cst_0, %cst, %cst_1) %out_qubits#0, %out_qubits#1 adj ctrls(%out_ctrl_qubits#0, %out_ctrl_qubits#1) ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit


    return %out_qubits_1#0, %out_qubits_1#1, %out_ctrl_qubits_1#0, %out_ctrl_qubits_1#1 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}


// -----


// test quantum.multirz labeled with adjoint attribute

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse(%arg0: f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract
    // CHECK: [[IN1:%.+]] = quantum.extract
    // CHECK: [[IN2:%.+]] = quantum.extract
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    %mrz:3 = quantum.multirz(%arg0) %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %mrz_out:3 = quantum.multirz(%arg0) %mrz#0, %mrz#1, %mrz#2 adj : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.multirz
    // CHECK: return [[IN0]], [[IN1]], [[IN2]]
    return %mrz_out#0, %mrz_out#1, %mrz_out#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----


// test quantum.multirz but wrong wire order

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse(%arg0: f64) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract
    // CHECK: [[IN1:%.+]] = quantum.extract
    // CHECK: [[IN2:%.+]] = quantum.extract
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    %mrz:3 = quantum.multirz(%arg0) %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %mrz_out:3 = quantum.multirz(%arg0) %mrz#1, %mrz#2, %mrz#0 adj : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: quantum.multirz
    return %mrz_out#0, %mrz_out#1, %mrz_out#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----


// test with matched control wires on named Hermitian gate

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1

    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract {{.+}}[ 0]
    // CHECK: [[IN1:%.+]] = quantum.extract {{.+}}[ 1]
    // CHECK: [[IN2:%.+]] = quantum.extract {{.+}}[ 2]
    %reg = quantum.alloc( 3) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit

    %out_qubits, %out_ctrl_qubits:2 = quantum.custom "Hadamard"() %0 ctrls(%1, %2) ctrlvals(%true, %false) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits_1, %out_ctrl_qubits_1:2 = quantum.custom "Hadamard"() %out_qubits ctrls(%out_ctrl_qubits#0, %out_ctrl_qubits#1) ctrlvals(%true, %false) : !quantum.bit ctrls !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.custom
    // CHECK: return [[IN0]], [[IN1]], [[IN2]]
    return %out_qubits_1, %out_ctrl_qubits_1#0, %out_ctrl_qubits_1#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----


// test with unmatched control wires on named Hermitian gate

// CHECK-LABEL: test_chained_self_inverse
func.func @test_chained_self_inverse() -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    %true = llvm.mlir.constant (1 : i1) :i1
    %false = llvm.mlir.constant (0 : i1) :i1

    // CHECK: quantum.alloc
    // CHECK: [[IN0:%.+]] = quantum.extract {{.+}}[ 0]
    // CHECK: [[IN1:%.+]] = quantum.extract {{.+}}[ 1]
    // CHECK: [[IN2:%.+]] = quantum.extract {{.+}}[ 2]
    %reg = quantum.alloc( 3) : !quantum.reg
    %0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %reg[ 1] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %reg[ 2] : !quantum.reg -> !quantum.bit

    %out_qubits, %out_ctrl_qubits:2 = quantum.custom "Hadamard"() %0 ctrls(%1, %2) ctrlvals(%true, %false) : !quantum.bit ctrls !quantum.bit, !quantum.bit
    %out_qubits_1, %out_ctrl_qubits_1:2 = quantum.custom "Hadamard"() %out_qubits ctrls(%out_ctrl_qubits#1, %out_ctrl_qubits#0) ctrlvals(%true, %false) : !quantum.bit ctrls !quantum.bit, !quantum.bit

    // CHECK: quantum.custom "Hadamard"
    return %out_qubits_1, %out_ctrl_qubits_1#0, %out_ctrl_qubits_1#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

func.func @test_loop_boundary_rotation(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {

    %stop = arith.constant 10 : index
    %one = arith.constant 1 : index
    %theta = arith.constant 0.2 : f64
    %beta = arith.constant 0.3 : f64

    // Quantum circuit:           // Expected output:
    // RX(theta) Q0               // RX(theta) Q0
    //                            // X Q1
    // for _ in range(n)          // for _ in range(n):
    //    RX(theta) Q0      ->    //    RX(theta) Q0
    //    X Q1                    //
    //    CNOT Q0, Q1             //    CNOT Q0, Q1
    //    RX(theta) Q0            //    RX(theta) Q0
    //    X Q1                    // X Q1

    // CHECK-LABEL: func.func @test_loop_boundary_rotation(
    // CHECK-SAME: [[q0:%.+]]: !quantum.bit,
    // CHECK-SAME: [[q1:%.+]]: !quantum.bit
    // CHECK: [[cst:%.+]] = {{.*}} 2.000000e-01 : f64
    // CHECK: [[cst_0:%.+]] = {{.*}} 3.000000e-01 : f64

    // CHECK: [[qubit_0:%.+]] = {{.*}} "RX"([[cst_0]]) [[q0]]
    %q_00 = quantum.custom "RX"(%beta) %q0 : !quantum.bit
    // CHECK: [[qubit_1:%.+]] = {{.*}} "X"() [[q1]]
    // CHECK: [[scf:%.+]]:2 = scf.for {{.*}} iter_args([[arg0:%.+]] = [[qubit_0]], [[arg1:%.+]] = [[qubit_1]])
    %scf:2 = scf.for %i = %one to %stop step %one iter_args(%q_arg0 = %q_00, %q_arg1 = %q1) -> (!quantum.bit, !quantum.bit) {
        // CHECK-NOT: "X"
        // CHECK: [[qubit_a:%.+]] = {{.*}} "RX"([[cst]]) [[arg0]]
        %q_0 = quantum.custom "RX"(%theta) %q_arg0 : !quantum.bit
        %q_1 = quantum.custom "X"() %q_arg1 : !quantum.bit
        // CHECK: [[qubit_2:%.+]]:2 = {{.*}} "CNOT"() [[qubit_a]], [[arg1]]
        %q_2:2 = quantum.custom "CNOT"() %q_0, %q_1 : !quantum.bit, !quantum.bit
        // CHECK: [[qubit_3:%.+]] = {{.*}} "RX"([[cst]]) [[qubit_2]]#0
        %q_3 = quantum.custom "RX"(%theta) %q_2#0 : !quantum.bit
        // CHECK-NOT: "X"
        %q_4 = quantum.custom "X"() %q_2#1 : !quantum.bit
        scf.yield %q_3, %q_4 : !quantum.bit, !quantum.bit
    }

    // CHECK: [[qubit_6:%.+]] = {{.*}} "X"() [[scf]]#1
    // CHECK: return [[scf]]#0, [[qubit_6]]
    func.return %scf#0, %scf#1 : !quantum.bit, !quantum.bit
}
