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

// RUN: quantum-opt --lower-gradients --split-input-file %s | FileCheck %s

// CHECK-LABEL: @f(
// CHECK-LABEL: @f.shifted
// CHECK-LABEL: @f.qgrad
// CHECK-NOT: quantum.
// CHECK-LABEL: @f.quantum
// CHECK-LABEL: @f.preprocess
// CHECK-NOT: quantum.
func.func private @f(%arg0: tensor<f64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %0 = "quantum.alloc"() {nqubits_attr = 3 : i64} : () -> !quantum.reg
    %1 = "quantum.extract"(%0, %c2_i64) : (!quantum.reg, i64) -> !quantum.bit
    %2 = "quantum.custom"(%1) {gate_name = "Hadamard", operand_segment_sizes = array<i32: 0, 1, 0, 0>, result_segment_sizes = array<i32: 1, 0> } : (!quantum.bit) -> !quantum.bit
    %3 = "quantum.extract"(%0, %c0_i64) : (!quantum.reg, i64) -> !quantum.bit
    %4 = tensor.extract %arg0[] : tensor<f64>
    %5:2 = "quantum.custom"(%4, %3, %2) {gate_name = "CRX", operand_segment_sizes = array<i32: 1, 2, 0, 0>, result_segment_sizes = array<i32: 2, 0> } : (f64, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    %6 = "quantum.namedobs"(%5#0) {type = #quantum<named_observable PauliZ>} : (!quantum.bit) -> !quantum.obs
    %7 = "quantum.expval"(%6) {shots = 1000 : i64} : (!quantum.obs) -> f64
    %8 = tensor.from_elements %7 : tensor<f64>
    "quantum.dealloc"(%0) : (!quantum.reg) -> ()
    return %8 : tensor<f64>
}

// CHECK-LABEL: @gradCall0
// CHECK: call @f.fullgrad0
func.func @gradCall0(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = gradient.grad "auto" @f(%arg0) : (tensor<f64>) -> tensor<f64>
    func.return %0 : tensor<f64>
}

// -----

// CHECK-LABEL: @f2(
// CHECK-LABEL: @f2.shifted
// CHECK-LABEL: @f2.qgrad
// CHECK-NOT: quantum.
// CHECK-LABEL: @f2.quantum
// CHECK-LABEL: @f2.preprocess
// CHECK-NOT: quantum.
func.func private @f2(%arg0: tensor<f64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = "quantum.alloc"() {nqubits_attr = 3 : i64} : () -> !quantum.reg
    %1 = "quantum.extract"(%0, %c0_i64) : (!quantum.reg, i64) -> !quantum.bit
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %2 = "quantum.custom"(%extracted, %1) {gate_name = "RX", operand_segment_sizes = array<i32: 1, 1, 0, 0>, result_segment_sizes = array<i32: 1, 0>} : (f64, !quantum.bit) -> !quantum.bit
    %3 = "quantum.custom"(%2) {gate_name = "Hadamard", operand_segment_sizes = array<i32: 0, 1, 0, 0>, result_segment_sizes = array<i32: 1, 0>} : (!quantum.bit) -> !quantum.bit
    %4 = "quantum.insert"(%0, %c0_i64, %3) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
    %extracted_0 = tensor.extract %arg1[] : tensor<i64>
    %5 = arith.index_cast %extracted_0 : i64 to index
    %6 = scf.for %arg3 = %c1 to %5 step %c1 iter_args(%arg4 = %4) -> (!quantum.reg) {
        %10 = arith.index_cast %arg3 : index to i64
        %11 = "quantum.extract"(%arg4, %10) : (!quantum.reg, i64) -> !quantum.bit
        %extracted_2 = tensor.extract %arg0[] : tensor<f64>
        %12 = "quantum.custom"(%extracted_2, %11) {gate_name = "RX", operand_segment_sizes = array<i32: 1, 1, 0, 0>, result_segment_sizes = array<i32: 1, 0>} : (f64, !quantum.bit) -> !quantum.bit
        %13 = "quantum.insert"(%arg4, %10, %12) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
        %14 = "quantum.extract"(%13, %c0_i64) : (!quantum.reg, i64) -> !quantum.bit
        %15 = "quantum.extract"(%13, %10) : (!quantum.reg, i64) -> !quantum.bit
        %16:2 = "quantum.custom"(%14, %15) {gate_name = "CNOT", operand_segment_sizes = array<i32: 0, 2, 0, 0>, result_segment_sizes = array<i32: 2, 0>} : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %17 = "quantum.insert"(%13, %10, %16#1) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
        %18 = "quantum.insert"(%17, %c0_i64, %16#0) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
        scf.yield %18 : !quantum.reg
    }
    %extracted_1 = tensor.extract %arg2[] : tensor<i64>
    %7 = "quantum.extract"(%6, %extracted_1) : (!quantum.reg, i64) -> !quantum.bit
    %8 = "quantum.namedobs"(%7) {type = #quantum<named_observable PauliY>} : (!quantum.bit) -> !quantum.obs
    %9 = "quantum.expval"(%8) {shots = 1000 : i64} : (!quantum.obs) -> f64
    %from_elements = tensor.from_elements %9 : tensor<f64>
    "quantum.dealloc"(%0) : (!quantum.reg) -> ()
    return %from_elements : tensor<f64>
}

// CHECK-LABEL: @gradCall1
func.func public @gradCall1(%arg0: tensor<f64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<f64> {
    %0 = gradient.grad "auto" @f2(%arg0, %arg1, %arg2) : (tensor<f64>, tensor<i64>, tensor<i64>) -> tensor<f64>
    return %0 : tensor<f64>
}

// -----

#map = affine_map<() -> ()>

// CHECK-LABEL: @f3(
// CHECK-LABEL: @f3.shifted
// CHECK-LABEL: @f3.qgrad
// CHECK-NOT: quantum.
// CHECK-LABEL: @f3.quantum
// CHECK-LABEL: @f3.preprocess
// CHECK-NOT: quantum.
func.func private @f3(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {qnode, diff_method = "parameter-shift"} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<2.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<1.500000e+00> : tensor<f64>
    %0 = "quantum.alloc"() {nqubits_attr = 1 : i64} : () -> !quantum.reg
    %1 = tensor.empty() : tensor<i1>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg1, %cst_0 : tensor<f64>, tensor<f64>) outs(%1 : tensor<i1>) {
    ^bb0(%in: f64, %in_1: f64, %out: i1):
        %7 = arith.cmpf ogt, %in, %in_1 : f64
        linalg.yield %7 : i1
    } -> tensor<i1>
    %extracted = tensor.extract %2[] : tensor<i1>
    %3 = scf.if %extracted -> (!quantum.reg) {
        %7 = tensor.empty() : tensor<f64>
        %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %cst : tensor<f64>, tensor<f64>) outs(%7 : tensor<f64>) {
        ^bb0(%in: f64, %in_2: f64, %out: f64):
            %12 = arith.mulf %in, %in_2 : f64
            linalg.yield %12 : f64
        } -> tensor<f64>
        %9 = "quantum.extract"(%0, %c0_i64) : (!quantum.reg, i64) -> !quantum.bit
        %extracted_1 = tensor.extract %8[] : tensor<f64>
        %10 = "quantum.custom"(%extracted_1, %9) {gate_name = "RX", operand_segment_sizes = array<i32: 1, 1, 0, 0>, result_segment_sizes = array<i32: 1, 0>} : (f64, !quantum.bit) -> !quantum.bit
        %11 = "quantum.insert"(%0, %c0_i64, %10) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
        scf.yield %11 : !quantum.reg
    } else {
        %7 = "quantum.extract"(%0, %c0_i64) : (!quantum.reg, i64) -> !quantum.bit
        %extracted_1 = tensor.extract %arg0[] : tensor<f64>
        %8 = "quantum.custom"(%extracted_1, %7) {gate_name = "RX", operand_segment_sizes = array<i32: 1, 1, 0, 0>, result_segment_sizes = array<i32: 1, 0>} : (f64, !quantum.bit) -> !quantum.bit
        %9 = "quantum.insert"(%0, %c0_i64, %8) : (!quantum.reg, i64, !quantum.bit) -> !quantum.reg
        scf.yield %9 : !quantum.reg
    }
    %4 = "quantum.extract"(%3, %c0_i64) : (!quantum.reg, i64) -> !quantum.bit
    %5 = "quantum.namedobs"(%4) {type = #quantum<named_observable PauliY>} : (!quantum.bit) -> !quantum.obs
    %6 = "quantum.expval"(%5) {shots = 1000 : i64} : (!quantum.obs) -> f64
    %from_elements = tensor.from_elements %6 : tensor<f64>
    "quantum.dealloc"(%0) : (!quantum.reg) -> ()
    return %from_elements : tensor<f64>
}

// CHECK-LABEL: @gradcall2
func.func public @gradcall2(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
    %0 = gradient.grad "auto" @f3(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
}
