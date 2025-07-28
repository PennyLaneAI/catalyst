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

//---

// CHECK-LABEL: func.func public @shortname(
// CHECK-SAME:                              %[[ARG0:.+]]: i64
// CHECK-SAME:                            ) -> i64 {
// CHECK-NEXT:     [[C1:.+]] = arith.constant 1 : i64
// CHECK-NEXT:    [[RES:%.+]] = arith.addi %[[ARG0]], [[C1:.+]] : i64
// CHECK-NEXT:    return [[RES]] : i64
// CHECK-LABEL: func.func @main(
// CHECK-SAME:                %[[ARG0:.+]]: i64
// CHECK-SAME:              ) -> i64
// CHECK:         %{{.*}} = call @shortname(%[[ARG0]]) : (i64) -> i64
// CHECK-NEXT:    return %{{.*}} : i64
module {
  func.func public @shortname(%arg0: tensor<i64>) -> tensor<i64> {
    %c1 = arith.constant 1 : i64
    %extracted_val = tensor.extract %arg0[] : tensor<i64>
    %added_val = arith.addi %extracted_val, %c1 : i64
    %tensor_result = tensor.from_elements %added_val : tensor<i64>
    return %tensor_result : tensor<i64>
  }

  func.func @main(%arg0: i64) -> i64 {
    %tensor_arg = tensor.from_elements %arg0 : tensor<i64>
    %result_tensor = func.call @shortname(%tensor_arg) : (tensor<i64>) -> tensor<i64>
    %result_scalar = tensor.extract %result_tensor[] : tensor<i64>
    return %result_scalar : i64
  }
}

// -----

// CHECK-LABEL: func.func public @callee_mixed(
// CHECK-SAME:                                 %[[ARG0:.+]]: f32,
// CHECK-SAME:                                 %[[ARG1:.+]]: tensor<2xf32>
// CHECK-SAME:                               ) -> (f32, tensor<2xf32>)
// CHECK-NEXT:    return %[[ARG0]], %[[ARG1]] : f32, tensor<2xf32>
// CHECK-LABEL: func.func @caller_mixed(
// CHECK-SAME:                         %[[ARG0:.+]]: f32,
// CHECK-SAME:                         %[[ARG1:.+]]: tensor<2xf32>
// CHECK-SAME:                       ) -> (f32, tensor<2xf32>)
// CHECK:         [[CALL:%.+]]:2 = call @callee_mixed(%[[ARG0]], %[[ARG1]]) : (f32, tensor<2xf32>) -> (f32, tensor<2xf32>)
// CHECK-NEXT:    return [[CALL]]#0, [[CALL]]#1 : f32, tensor<2xf32>
module {
  func.func public @callee_mixed(%arg0: tensor<f32>, %arg1: tensor<2xf32>) -> (tensor<f32>, tensor<2xf32>) {
    return %arg0, %arg1 : tensor<f32>, tensor<2xf32>
  }

  func.func @caller_mixed(%arg0: f32, %arg1: tensor<2xf32>) -> (f32, tensor<2xf32>) {
    %tensor_arg = tensor.from_elements %arg0 : tensor<f32>
    %results:2 = func.call @callee_mixed(%tensor_arg, %arg1) : (tensor<f32>, tensor<2xf32>) -> (tensor<f32>, tensor<2xf32>)
    %scalar_res = tensor.extract %results#0[] : tensor<f32>
    return %scalar_res, %results#1 : f32, tensor<2xf32>
  }
}

// -----

// CHECK-LABEL: func.func public @no_change(
// CHECK-SAME:                             %[[ARG0:.+]]: tensor<1xi64>
// CHECK-SAME:                           ) -> tensor<1xi64>
// CHECK-NEXT:    return %[[ARG0]] : tensor<1xi64>
module {
  // This function should not be modified by the pass since it does not use 0-D tensors.
  func.func public @no_change(%arg0: tensor<1xi64>) -> tensor<1xi64> {
    return %arg0 : tensor<1xi64>
  }
}
