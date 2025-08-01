// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --detensorize-function-boundary --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func public @test_scalar_tensor(
// CHECK-SAME:                              %[[ARG0:.+]]: i64
// CHECK-SAME:                            ) -> i64 {
// CHECK-NEXT:    return %[[ARG0]] : i64
module{
	func.func public @test_scalar_tensor(%arg0: tensor<i64>) -> tensor<i64> {
    return %arg0 : tensor<i64>
  }
}

// -----

// CHECK-LABEL: func.func public @test_fold_from_elements_extract(
// CHECK-SAME:                              %[[ARG0:.+]]: i64
// CHECK-SAME:                            ) -> i64 {
// CHECK-NEXT:     [[C1:.+]] = arith.constant 1 : i64
// CHECK-NEXT:    [[RES:%.+]] = arith.addi %[[ARG0]], [[C1:.+]] : i64
// CHECK-NEXT:    return [[RES]] : i64
module{
	func.func public @test_fold_from_elements_extract(%arg0: tensor<i64>) -> tensor<i64> {
    %c1 = arith.constant 1 : i64
    %extracted_val = tensor.extract %arg0[] : tensor<i64>
    %added_val = arith.addi %extracted_val, %c1 : i64
    %tensor_result = tensor.from_elements %added_val : tensor<i64>
    return %tensor_result : tensor<i64>
  }
}

// -----

// CHECK-LABEL: func.func public @test_add_tensor_op_to_caller(
// CHECK-SAME:                              %[[ARG0:.+]]: i64
// CHECK-SAME:                            ) -> i64 {
// CHECK-LABEL: func.func @main(
// CHECK-SAME:                   %[[ARG1:.+]]: i64
// CHECK-SAME:                   ) -> i64 {
// CHECK-NEXT:    %[[RET:.+]] = call @test_add_tensor_op_to_caller(%[[ARG1]]) : (i64) -> i64
// CHECK-NEXT:    return %[[RET]] : i64
module {
  func.func public @test_add_tensor_op_to_caller(%arg0: tensor<i64>) -> tensor<i64> {
    %c1 = arith.constant 1 : i64
    %extracted_val = tensor.extract %arg0[] : tensor<i64>
    %added_val = arith.addi %extracted_val, %c1 : i64
    %tensor_result = tensor.from_elements %added_val : tensor<i64>
    return %tensor_result : tensor<i64>
  }

  func.func @main(%arg1: i64) -> i64 {
    %tensor_arg = tensor.from_elements %arg1 : tensor<i64>
    %result_tensor = func.call @test_add_tensor_op_to_caller(%tensor_arg) : (tensor<i64>) -> tensor<i64>
    %result_scalar = tensor.extract %result_tensor[] : tensor<i64>
    return %result_scalar : i64
  }
}

// -----

// CHECK-LABEL: func.func public @test_multi_args(
// CHECK-SAME:                              %[[ARG0:.+]]: i32, %[[ARG1:.+]]: f64
// CHECK-SAME:                            ) -> (i32, f64) {
// CHECK-NEXT:    return %[[ARG0]], %[[ARG1]] : i32, f64
module {
  func.func public @test_multi_args(%arg0: tensor<i32>, %arg1: tensor<f64>) -> (tensor<i32>, tensor<f64>) {
    return %arg0, %arg1 : tensor<i32>, tensor<f64>
  }
}

// -----

// CHECK-LABEL: func.func public @test_mixed_args(
// CHECK-SAME:                              %[[ARG0:.+]]: f32, %[[ARG1:.+]]: tensor<2xf32>
// CHECK-SAME:                            ) -> (f32, tensor<2xf32>) {
// CHECK-NEXT:    return %[[ARG0]], %[[ARG1]] : f32, tensor<2xf32>
module {
  func.func public @test_mixed_args(%arg0: tensor<f32>, %arg1: tensor<2xf32>) -> (tensor<f32>, tensor<2xf32>) {
    %c1 = arith.constant 1.0 : f32
    %extracted_val = tensor.extract %arg0[] : tensor<f32>
    %added_val = arith.addf %extracted_val, %c1 : f32
    %tensor_result = tensor.from_elements %added_val : tensor<f32>
    return %arg0, %arg1 : tensor<f32>, tensor<2xf32>
  }
}

// -----

// CHECK-LABEL: func.func public @test_no_change(
// CHECK-SAME:                              %[[ARG0:.+]]: tensor<1xi64>
// CHECK-SAME:                            ) -> tensor<1xi64> {
// CHECK-NEXT:    return %[[ARG0]] : tensor<1xi64>
module {
  func.func public @test_no_change(%arg0: tensor<1xi64>) -> tensor<1xi64> {
    return %arg0 : tensor<1xi64>
  }
}

// -----

// CHECK-LABEL: func.func public @test_nested_callee(
// CHECK-SAME:                              %[[A:.+]]: i64
// CHECK-SAME:                            ) -> i64 {
// CHECK-NEXT:    return %[[A]] : i64
// CHECK-LABEL: func.func @test_nested_caller(
// CHECK-SAME:                   %[[B:.+]]: i64
// CHECK-SAME:                   ) -> i64 {
// CHECK-NEXT:    %[[RES:.+]] = call @test_nested_callee(%[[B]]) : (i64) -> i64
// CHECK-NEXT:    return %[[RES]] : i64
module {
  func.func public @test_nested_callee(%a: tensor<i64>) -> tensor<i64> {
    return %a : tensor<i64>
  }
  func.func @test_nested_caller(%b: tensor<i64>) -> tensor<i64> {
    %res = func.call @test_nested_callee(%b) : (tensor<i64>) -> tensor<i64>
    return %res : tensor<i64>
  }
}

// -----

// CHECK-LABEL: func.func public @test_preserve_attr(
// CHECK-SAME:                              %[[A:.+]]: i64
// CHECK-SAME:                            ) -> i64 attributes {custom_attr = "test"} {
// CHECK-NEXT:    return %[[A]] : i64
module {
  func.func public @test_preserve_attr(%a: tensor<i64>) -> tensor<i64> attributes {custom_attr = "test"} {
    return %a : tensor<i64>
  }
}
