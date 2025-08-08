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

// -----

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                   %[[ARG1:.+]]: i64
// CHECK-SAME:                   ) -> i64 {
// CHECK-NEXT:    %[[RET:.+]] = call @test_add_tensor_op_to_caller.detensorized(%[[ARG1]]) : (i64) -> i64
// CHECK-NEXT:    return %[[RET]] : i64
// CHECK-LABEL: func.func public @test_add_tensor_op_to_caller.detensorized(
// CHECK-SAME:                              %[[ARG0:.+]]: i64
// CHECK-SAME:                            ) -> i64 {
module {
  func.func @main(%arg1: i64) -> i64 {
    %tensor_arg = tensor.from_elements %arg1 : tensor<i64>
    %result_tensor = func.call @test_add_tensor_op_to_caller(%tensor_arg) : (tensor<i64>) -> tensor<i64>
    %result_scalar = tensor.extract %result_tensor[] : tensor<i64>
    return %result_scalar : i64
  }

  func.func public @test_add_tensor_op_to_caller(%arg0: tensor<i64>) -> tensor<i64> {
    %c1 = arith.constant 1 : i64
    %extracted_val = tensor.extract %arg0[] : tensor<i64>
    %added_val = arith.addi %extracted_val, %c1 : i64
    %tensor_result = tensor.from_elements %added_val : tensor<i64>
    return %tensor_result : tensor<i64>
  }
}

// -----

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                              %[[ARG0:.+]]: i32, %[[ARG1:.+]]: f64
// CHECK-SAME:                            ) -> (i32, f64) {
// CHECK-NEXT:    %[[RESULT:[a-zA-Z0-9_.-]+]]:2 = call @test_multi_args.detensorized(%[[ARG0]], %[[ARG1]]) : (i32, f64) -> (i32, f64)
// CHECK-NEXT:    return %[[RESULT]]#0, %[[RESULT]]#1 : i32, f64
// CHECK-LABEL: func.func public @test_multi_args.detensorized(
// CHECK-SAME:                              %[[ARG0:.+]]: i32, %[[ARG1:.+]]: f64
// CHECK-SAME:                            ) -> (i32, f64) {
// CHECK-NEXT:    return %[[ARG0]], %[[ARG1]] : i32, f64
module {
  func.func @main(%arg1: i32, %arg2: f64) -> (i32, f64) {
    %tensor_arg1 = tensor.from_elements %arg1 : tensor<i32>
    %tensor_arg2 = tensor.from_elements %arg2 : tensor<f64>
    %result_1, %result_2 = func.call @test_multi_args(%tensor_arg1, %tensor_arg2) : (tensor<i32>, tensor<f64>) -> (tensor<i32>, tensor<f64>)
    %result_1_scalar = tensor.extract %result_1[] : tensor<i32>
    %result_2_scalar = tensor.extract %result_2[] : tensor<f64>
    return %result_1_scalar, %result_2_scalar : i32, f64
  }
  func.func public @test_multi_args(%arg0: tensor<i32>, %arg1: tensor<f64>) -> (tensor<i32>, tensor<f64>) {
    return %arg0, %arg1 : tensor<i32>, tensor<f64>
  }
}


// -----

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                              %[[ARG0:.+]]: f32, %[[ARG1:.+]]: f64
// CHECK-SAME:                            ) -> (f32, tensor<2xf64>) {
// CHECK:    %[[RESULT:[a-zA-Z0-9_.-]+]]:3 = call @test_mixed_args.detensorized(%[[ARG0]], %[[TENSOR_ARG1:.+]], %[[ARG2:.+]]) : (f32, tensor<2xf64>, i1) -> (f32, tensor<2xf64>, i1)
// CHECK-NEXT:    return %[[RESULT]]#0, %[[RESULT]]#1 : f32, tensor<2xf64>
// CHECK-LABEL: func.func public @test_mixed_args.detensorized(
// CHECK-SAME:                              %[[ARG0:.+]]: f32, %[[ARG1:.+]]: tensor<2xf64>, %[[ARG2:.+]]: i1
// CHECK-SAME:                            ) -> (f32, tensor<2xf64>, i1) {
// CHECK:    %[[RET0:.+]] = arith.addf %[[ARG0]], %[[CST:.+]] : f32
// CHECK-NEXT:    return %[[RET0]], %[[ARG1]], %[[ARG2]] : f32, tensor<2xf64>, i1
module {
  func.func @main(%arg0: f32, %arg1: f64) -> (f32, tensor<2xf64>) {
    %tensor_arg0 = tensor.from_elements %arg0 : tensor<f32>
    %tensor_arg1 = tensor.from_elements %arg1, %arg1 : tensor<2xf64>
    %bool_arg2 = arith.constant 1 : i1
    %result_1, %result_2, %result_3 = func.call @test_mixed_args(%tensor_arg0, %tensor_arg1, %bool_arg2) : (tensor<f32>, tensor<2xf64>, i1) -> (tensor<f32>, tensor<2xf64>, i1)
    %result_1_scalar = tensor.extract %result_1[] : tensor<f32>
    return %result_1_scalar, %result_2 : f32, tensor<2xf64>
  }
  func.func public @test_mixed_args(%arg0: tensor<f32>, %arg1: tensor<2xf64>, %arg2: i1) -> (tensor<f32>, tensor<2xf64>, i1) {
    %c1 = arith.constant 1.0 : f32
    %extracted_val = tensor.extract %arg0[] : tensor<f32>
    %added_val = arith.addf %extracted_val, %c1 : f32
    %tensor_result = tensor.from_elements %added_val : tensor<f32>
    return %tensor_result, %arg1, %arg2 : tensor<f32>, tensor<2xf64>, i1
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

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                   %[[ARG0:.+]]: i64
// CHECK-SAME:                   ) -> i64 {
// CHECK-NEXT:    %[[RES0:.+]] = call @test_nested_caller.detensorized(%[[ARG0]]) : (i64) -> i64
// CHECK-NEXT:    return %[[RES0]] : i64
// CHECK-LABEL: func.func @test_nested_callee.detensorized(
// CHECK-SAME:                              %[[ARG2:.+]]: i64
// CHECK-SAME:                            ) -> i64 {
// CHECK-NEXT:    return %[[ARG2]] : i64
// CHECK-LABEL: func.func @test_nested_caller.detensorized(
// CHECK-SAME:                   %[[ARG1:.+]]: i64
// CHECK-SAME:                   ) -> i64 {
// CHECK-NEXT:    %[[RES1:.+]] = call @test_nested_callee.detensorized(%[[ARG1]]) : (i64) -> i64
// CHECK-NEXT:    return %[[RES1]] : i64
module {
  func.func @main(%arg0: i64) -> i64 {
    %tensor_arg = tensor.from_elements %arg0 : tensor<i64>
    %result_tensor = func.call @test_nested_caller(%tensor_arg) : (tensor<i64>) -> tensor<i64>
    %result_scalar = tensor.extract %result_tensor[] : tensor<i64>
    return %result_scalar : i64
  }
  func.func @test_nested_caller(%b: tensor<i64>) -> tensor<i64> {
    %res = func.call @test_nested_callee(%b) : (tensor<i64>) -> tensor<i64>
    return %res : tensor<i64>
  }
  func.func @test_nested_callee(%a: tensor<i64>) -> tensor<i64> {
    return %a : tensor<i64>
  }
}
