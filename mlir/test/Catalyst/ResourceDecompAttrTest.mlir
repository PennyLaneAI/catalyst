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

// RUN: quantum-opt --pass-pipeline="builtin.module(resource-tracker{decomp-attr=true})" --split-input-file %s | FileCheck %s

// Basic decomposition rule

// CHECK: attributes {resources = {measurements = {}, num_alloc_qubits = 2 : i64, operations = {"CNOT(2)" = 1 : i64, "Hadamard(1)" = 1 : i64, "S(1)" = 1 : i64, "T(1)" = 1 : i64}}, target_gate = "basic"}
func.func @basic_gates() attributes {target_gate="basic"}  {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %4 = quantum.custom "T"() %3 : !quantum.bit
    %5 = quantum.custom "S"() %2 : !quantum.bit
    %6:2 = quantum.custom "CNOT"() %4, %5 : !quantum.bit, !quantum.bit
    %7 = quantum.insert %0[ 0], %6#0 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %6#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}

// -----

// Rule with PBC ops

// CHECK: attributes {resources = {measurements = {}, num_alloc_qubits = 2 : i64, operations = {"PPM(1)" = 2 : i64, "PPR-pi/4(1)" = 3 : i64, "PPR-pi/8(1)" = 1 : i64}}, target_gate = "pbc"}
func.func @pbc_operations() attributes {target_gate="pbc"} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = pbc.ppr ["Z"](4) %1 : !quantum.bit
    %4 = pbc.ppr ["X"](4) %3 : !quantum.bit
    %5 = pbc.ppr ["Z"](4) %2 : !quantum.bit
    %6 = pbc.ppr ["Z"](8) %4 : !quantum.bit
    %mres, %out = pbc.ppm ["Z"] %5 : i1, !quantum.bit
    %mres2, %out2 = pbc.ppm ["X"] %6 : i1, !quantum.bit
    %7 = quantum.insert %0[ 0], %out2 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}

// -----

// Rule with measure

// CHECK: attributes {resources = {measurements = {MidCircuitMeasure = 1 : i64}, num_alloc_qubits = 1 : i64, operations = {"Hadamard(1)" = 1 : i64}}, target_gate = "gate"}
func.func @rule_mcm() attributes {target_gate="gate"} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %mres, %out = quantum.measure %2 : i1, !quantum.bit
    %3 = quantum.insert %0[ 0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
}

// -----

// Rule depends on helper functions

// CHECK: @helper_func(%arg0: !quantum.bit) -> !quantum.bit
func.func private @helper_func(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}

// CHECK: rule(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode, resources =
func.func @rule(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode, target_gate="rule"} {
    %r1 = func.call @helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}


// -----

// CNOT Decomposition Rule

// CHECK: resources = {measurements = {}, num_alloc_qubits = 0 : i64, operations = {"CZ(2)" = 1 : i64, "Hadamard(1)" = 2 : i64}}, target_gate = "CNOT"}
func.func public @_cnot_to_cz_h(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {num_wires = 2 : i64, target_gate = "CNOT"} {
    %0 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %2 : !quantum.bit
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %5 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %6 = stablehlo.reshape %5 : (tensor<1xi64>) -> tensor<i64>
    %extracted_0 = tensor.extract %1[] : tensor<i64>
    %7 = quantum.insert %arg0[%extracted_0], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_1 = tensor.extract %4[] : tensor<i64>
    %8 = quantum.extract %7[%extracted_1] : !quantum.reg -> !quantum.bit
    %extracted_2 = tensor.extract %6[] : tensor<i64>
    %9 = quantum.extract %7[%extracted_2] : !quantum.reg -> !quantum.bit
    %out_qubits_3:2 = quantum.custom "CZ"() %8, %9 : !quantum.bit, !quantum.bit
    %10 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %11 = stablehlo.reshape %10 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %4[] : tensor<i64>
    %12 = quantum.insert %7[%extracted_4], %out_qubits_3#0 : !quantum.reg, !quantum.bit
    %extracted_5 = tensor.extract %6[] : tensor<i64>
    %13 = quantum.insert %12[%extracted_5], %out_qubits_3#1 : !quantum.reg, !quantum.bit
    %extracted_6 = tensor.extract %11[] : tensor<i64>
    %14 = quantum.extract %13[%extracted_6] : !quantum.reg -> !quantum.bit
    %out_qubits_7 = quantum.custom "Hadamard"() %14 : !quantum.bit
    %extracted_8 = tensor.extract %11[] : tensor<i64>
    %15 = quantum.insert %13[%extracted_8], %out_qubits_7 : !quantum.reg, !quantum.bit
    return %15 : !quantum.reg
}

// -----

// CZ to PPR Decomposition Rule

// CHECK: resources = {measurements = {}, num_alloc_qubits = 0 : i64, operations = {"GlobalPhase(0)" = 1 : i64, "PauliRot(1)" = 2 : i64, "PauliRot(2)" = 1 : i64}}
func.func public @_cz_to_ppr(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {num_wires = 2 : i64, target_gate = "CZ"} {
    %cst = arith.constant 0.78539816339744828 : f64
    %cst_0 = arith.constant 1.5707963267948966 : f64
    %cst_1 = arith.constant -1.5707963267948966 : f64
    %0 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %1 = stablehlo.reshape %0 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i64>
    %2 = quantum.extract %arg0[%extracted] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.paulirot ["Z"](%cst_1) %2 : !quantum.bit
    %3 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %4 = stablehlo.reshape %3 : (tensor<1xi64>) -> tensor<i64>
    %extracted_2 = tensor.extract %1[] : tensor<i64>
    %5 = quantum.insert %arg0[%extracted_2], %out_qubits : !quantum.reg, !quantum.bit
    %extracted_3 = tensor.extract %4[] : tensor<i64>
    %6 = quantum.extract %5[%extracted_3] : !quantum.reg -> !quantum.bit
    %out_qubits_4 = quantum.paulirot ["Z"](%cst_1) %6 : !quantum.bit
    %7 = stablehlo.slice %arg1 [0:1] : (tensor<2xi64>) -> tensor<1xi64>
    %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
    %9 = stablehlo.slice %arg1 [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
    %extracted_5 = tensor.extract %4[] : tensor<i64>
    %11 = quantum.insert %5[%extracted_5], %out_qubits_4 : !quantum.reg, !quantum.bit
    %extracted_6 = tensor.extract %8[] : tensor<i64>
    %12 = quantum.extract %11[%extracted_6] : !quantum.reg -> !quantum.bit
    %extracted_7 = tensor.extract %10[] : tensor<i64>
    %13 = quantum.extract %11[%extracted_7] : !quantum.reg -> !quantum.bit
    %out_qubits_8:2 = quantum.paulirot ["Z", "Z"](%cst_0) %12, %13 : !quantum.bit, !quantum.bit
    %extracted_9 = tensor.extract %8[] : tensor<i64>
    %14 = quantum.insert %11[%extracted_9], %out_qubits_8#0 : !quantum.reg, !quantum.bit
    %extracted_10 = tensor.extract %10[] : tensor<i64>
    %15 = quantum.insert %14[%extracted_10], %out_qubits_8#1 : !quantum.reg, !quantum.bit
    quantum.gphase(%cst) :
    return %15 : !quantum.reg
}

// -----

// MultiRZ Decomposition Rule
// TODO: Update resource-tracking to return parametric resources with loops

// CHECK: resources = {measurements = {}, num_alloc_qubits = 0 : i64, operations = {"CNOT(2)" = 4 : i64, "RZ(1)" = 1 : i64}}
func.func public @_multi_rz_decomposition(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<3xi64>) -> !quantum.reg attributes {target_gate = "MultiRZ"} {
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c = stablehlo.constant dense<3> : tensor<i64>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg0) -> (!quantum.reg) {
    %6 = arith.subi %c2, %arg3 : index
    %7 = arith.index_cast %6 : index to i64
    %from_elements = tensor.from_elements %7 : tensor<i64>
    %8 = stablehlo.compare  LT, %from_elements, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %9 = stablehlo.convert %from_elements : tensor<i64>
    %10 = stablehlo.add %9, %c : tensor<i64>
    %11 = stablehlo.select %8, %10, %from_elements : tensor<i1>, tensor<i64>
    %12 = stablehlo.dynamic_slice %arg2, %11, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
    %13 = stablehlo.reshape %12 : (tensor<1xi64>) -> tensor<i64>
    %14 = stablehlo.subtract %from_elements, %c_0 : tensor<i64>
    %15 = stablehlo.compare  LT, %14, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %16 = stablehlo.convert %14 : tensor<i64>
    %17 = stablehlo.add %16, %c : tensor<i64>
    %18 = stablehlo.select %15, %17, %14 : tensor<i1>, tensor<i64>
    %19 = stablehlo.dynamic_slice %arg2, %18, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
    %20 = stablehlo.reshape %19 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %13[] : tensor<i64>
    %21 = quantum.extract %arg4[%extracted_4] : !quantum.reg -> !quantum.bit
    %extracted_5 = tensor.extract %20[] : tensor<i64>
    %22 = quantum.extract %arg4[%extracted_5] : !quantum.reg -> !quantum.bit
    %out_qubits_6:2 = quantum.custom "CNOT"() %21, %22 : !quantum.bit, !quantum.bit
    %extracted_7 = tensor.extract %13[] : tensor<i64>
    %23 = quantum.insert %arg4[%extracted_7], %out_qubits_6#0 : !quantum.reg, !quantum.bit
    %extracted_8 = tensor.extract %20[] : tensor<i64>
    %24 = quantum.insert %23[%extracted_8], %out_qubits_6#1 : !quantum.reg, !quantum.bit
    scf.yield %24 : !quantum.reg
    }
    %1 = stablehlo.slice %arg2 [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    %2 = stablehlo.reshape %1 : (tensor<1xi64>) -> tensor<i64>
    %extracted = tensor.extract %2[] : tensor<i64>
    %3 = quantum.extract %0[%extracted] : !quantum.reg -> !quantum.bit
    %extracted_2 = tensor.extract %arg1[%c0] : tensor<1xf64>
    %out_qubits = quantum.custom "RZ"(%extracted_2) %3 : !quantum.bit
    %extracted_3 = tensor.extract %2[] : tensor<i64>
    %4 = quantum.insert %0[%extracted_3], %out_qubits : !quantum.reg, !quantum.bit
    %5 = scf.for %arg3 = %c1 to %c3 step %c1 iter_args(%arg4 = %4) -> (!quantum.reg) {
    %6 = arith.index_cast %arg3 : index to i64
    %from_elements = tensor.from_elements %6 : tensor<i64>
    %7 = stablehlo.compare  LT, %from_elements, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %8 = stablehlo.convert %from_elements : tensor<i64>
    %9 = stablehlo.add %8, %c : tensor<i64>
    %10 = stablehlo.select %7, %9, %from_elements : tensor<i1>, tensor<i64>
    %11 = stablehlo.dynamic_slice %arg2, %10, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
    %12 = stablehlo.reshape %11 : (tensor<1xi64>) -> tensor<i64>
    %13 = stablehlo.subtract %from_elements, %c_0 : tensor<i64>
    %14 = stablehlo.compare  LT, %13, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %15 = stablehlo.convert %13 : tensor<i64>
    %16 = stablehlo.add %15, %c : tensor<i64>
    %17 = stablehlo.select %14, %16, %13 : tensor<i1>, tensor<i64>
    %18 = stablehlo.dynamic_slice %arg2, %17, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
    %19 = stablehlo.reshape %18 : (tensor<1xi64>) -> tensor<i64>
    %extracted_4 = tensor.extract %12[] : tensor<i64>
    %20 = quantum.extract %arg4[%extracted_4] : !quantum.reg -> !quantum.bit
    %extracted_5 = tensor.extract %19[] : tensor<i64>
    %21 = quantum.extract %arg4[%extracted_5] : !quantum.reg -> !quantum.bit
    %out_qubits_6:2 = quantum.custom "CNOT"() %20, %21 : !quantum.bit, !quantum.bit
    %extracted_7 = tensor.extract %12[] : tensor<i64>
    %22 = quantum.insert %arg4[%extracted_7], %out_qubits_6#0 : !quantum.reg, !quantum.bit
    %extracted_8 = tensor.extract %19[] : tensor<i64>
    %23 = quantum.insert %22[%extracted_8], %out_qubits_6#1 : !quantum.reg, !quantum.bit
    scf.yield %23 : !quantum.reg
    }
    return %5 : !quantum.reg
}
