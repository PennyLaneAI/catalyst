// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --detensorize-scf --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @test0
module @test0 {
  // CHECK-LABEL: @test_while_loop
  // CHECK:       scf.while {{.*}} (f64, i64) -> (f64, i64)
  // CHECK:       scf.condition{{.*}} : f64, i64
  // CHECK:       do
  // CHECK-NOT:   tensor<
  // CHECK:       tensor.from_elements
  // CHECK:       scf.if
  // CHECK-NOT:   tensor<
  // CHECK:       scf.yield {{.*}} : f64
  // CHECK-NOT:   tensor<
  // CHECK:       else
  // CHECK-NOT:   tensor<
  // CHECK:       scf.yield {{.*}} : f64
  // CHECK-NOT:   tensor<
  // CHECK:       scf.yield  {{.*}} : f64, i64
  // CHECK-NOT:   tensor<
  // CHECK:       from_elements
  func.func public @test_while_loop(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %c10_i64 = arith.constant 10 : i64
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %cst_1 = arith.constant dense<0> : tensor<i64>
    %cst_2 = arith.constant dense<2> : tensor<i64>
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %extracted_3 = tensor.extract %arg1[] : tensor<f64>
    %0 = arith.addf %extracted, %extracted_3 : f64
    %from_elements = tensor.from_elements %0 : tensor<f64>
    %1:2 = scf.while (%arg2 = %from_elements, %arg3 = %cst_1) : (tensor<f64>, tensor<i64>) -> (tensor<f64>, tensor<i64>) {
      %extracted_6 = tensor.extract %arg3[] : tensor<i64>
      %3 = arith.cmpi slt, %extracted_6, %c10_i64 : i64
      scf.condition(%3) %arg2, %arg3 : tensor<f64>, tensor<i64>
    } do {
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<i64>):
      %extracted_6 = tensor.extract %arg2[] : tensor<f64>
      %extracted_7 = tensor.extract %arg3[] : tensor<i64>
      %3 = func.call @fun(%arg2, %cst_2) : (tensor<f64>, tensor<i64>) -> tensor<f64>
      %extracted_8 = tensor.extract %3[] : tensor<f64>
      %4 = arith.cmpf une, %extracted_8, %cst : f64
      %5 = scf.if %4 -> (tensor<f64>) {
        %7 = arith.subf %extracted_6, %extracted_6 : f64
        %from_elements_10 = tensor.from_elements %7 : tensor<f64>
        scf.yield %from_elements_10 : tensor<f64>
      } else {
        %7 = arith.addf %extracted_6, %cst_0 : f64
        %from_elements_10 = tensor.from_elements %7 : tensor<f64>
        scf.yield %from_elements_10 : tensor<f64>
      }
      %6 = arith.addi %extracted_7, %c1_i64 : i64
      %from_elements_9 = tensor.from_elements %6 : tensor<i64>
      scf.yield %5, %from_elements_9 : tensor<f64>, tensor<i64>
    }
    %extracted_4 = tensor.extract %1#0[] : tensor<f64>
    %2 = arith.mulf %extracted, %extracted_4 : f64
    %from_elements_5 = tensor.from_elements %2 : tensor<f64>
    return %from_elements_5 : tensor<f64>
  }
  module attributes {llvm.linkage = #llvm.linkage<internal>, transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
      transform.yield 
    }
  }
  func.func private @fun(%arg0: tensor<f64> {mhlo.layout_mode = "default"}, %arg1: tensor<i64> {mhlo.layout_mode = "default"}) -> (tensor<f64> {mhlo.layout_mode = "default"}) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f64
    %extracted = tensor.extract %arg1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %0 = arith.sitofp %extracted : i64 to f64
    %1 = arith.remf %extracted_0, %0 : f64
    %from_elements = tensor.from_elements %1 : tensor<f64>
    return %from_elements : tensor<f64>
  }
}
