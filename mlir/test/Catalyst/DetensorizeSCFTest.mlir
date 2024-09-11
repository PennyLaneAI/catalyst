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
  // CHECK:       [[WHILE_RESULTS:.+]]:2 = scf.while {{.*}} (f64, i64) -> (f64, i64)
  // CHECK:         scf.condition{{.*}} : f64, i64
  // CHECK:       do
  // CHECK-NOT:     tensor<
  // CHECK:         tensor.from_elements
  // CHECK:         scf.if {{.*}} -> (f64)
  // CHECK-NOT:       tensor<
  // CHECK:           scf.yield {{.*}} : f64
  // CHECK-NOT:       tensor<
  // CHECK:         else
  // CHECK-NOT:       tensor<
  // CHECK:           scf.yield {{.*}} : f64
  // CHECK-NOT:       tensor<
  // CHECK:         scf.yield  {{.*}} : f64, i64
  // CHECK-NOT:     tensor<
  // CHECK:       tensor.from_elements [[WHILE_RESULTS:.+]]#0 : tensor<f64>
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
    %from_elements_5 = tensor.from_elements %extracted_4 : tensor<f64>
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

// -----

// CHECK-LABEL: @test1
module @test1 {
  // CHECK-LABEL: func.func public @test_nested_ifs
  // CHECK:     %[[IF0_RESULTS:.+]]:2 = scf.if {{.*}}-> (f64, f64)
  // CHECK-NOT:   tensor<
  // CHECK:       %[[IF1_RESULTS:.+]] = scf.if {{.*}}-> (f64)
  // CHECK-NOT:     tensor<
  // CHECK:         scf.yield {{.*}} : f64
  // CHECK-NOT:     tensor<
  // CHECK:       else
  // CHECK:         scf.yield {{.*}} : f64
  // CHECK-NOT:     tensor<
  // CHECK:       }
  // CHECK:       scf.yield {{.*}}, %[[IF1_RESULTS]] : f64, f64
  // CHECK-NOT:   tensor<
  // CHECK:     else
  // CHECK:       %[[IF1_RESULTS:.+]] = scf.if {{.*}}-> (f64)
  // CHECK-NOT:     tensor<
  // CHECK:         scf.yield {{.*}} : f64
  // CHECK-NOT:     tensor<
  // CHECK:       else
  // CHECK:         scf.yield {{.*}} : f64
  // CHECK-NOT:     tensor<
  // CHECK:       }
  // CHECK:       scf.yield %[[IF1_RESULTS]], {{.*}} : f64, f64
  // CHECK-NOT:   tensor<
  // CHECK:     }
  // CHECK:     %[[MULF_RES:.+]] = arith.mulf %[[IF0_RESULTS]]#0, %[[IF0_RESULTS]]#1 : f64
  // CHECK:     tensor.from_elements %[[MULF_RES]] : tensor<f64>
  func.func public @test_nested_ifs(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 2.000000e+00 : f64
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %extracted_0 = tensor.extract %arg1[] : tensor<f64>
    %0 = arith.cmpf ogt, %extracted, %extracted_0 : f64
    %1:2 = scf.if %0 -> (tensor<f64>, tensor<f64>) {
      %8 = arith.mulf %extracted_0, %cst : f64
      %9 = arith.cmpf ogt, %extracted, %8 : f64
      %10 = scf.if %9 -> (tensor<f64>) {
        %11 = arith.mulf %extracted, %cst : f64
        %from_elements_3 = tensor.from_elements %11 : tensor<f64>
        scf.yield %from_elements_3 : tensor<f64>
      } else {
        %11 = arith.divf %extracted, %cst : f64
        %from_elements_3 = tensor.from_elements %11 : tensor<f64>
        scf.yield %from_elements_3 : tensor<f64>
      }
      scf.yield %arg0, %10 : tensor<f64>, tensor<f64>
    } else {
      %8 = arith.mulf %extracted, %cst : f64
      %9 = arith.cmpf ogt, %extracted_0, %8 : f64
      %10 = scf.if %9 -> (tensor<f64>) {
        %11 = arith.mulf %extracted_0, %cst : f64
        %from_elements_3 = tensor.from_elements %11 : tensor<f64>
        scf.yield %from_elements_3 : tensor<f64>
      } else {
        %11 = arith.divf %extracted_0, %cst : f64
        %from_elements_3 = tensor.from_elements %11 : tensor<f64>
        scf.yield %from_elements_3 : tensor<f64>
      }
      scf.yield %10, %arg1 : tensor<f64>, tensor<f64>
    }
    %extracted_1 = tensor.extract %1#0[] : tensor<f64>
    %extracted_2 = tensor.extract %1#1[] : tensor<f64>
    %2 = arith.mulf %extracted_1, %extracted_2 : f64
    %from_elements = tensor.from_elements %2 : tensor<f64>
    return %from_elements : tensor<f64>
  }
}

// -----

// CHECK-LABEL: @test2
module @test2 {
  // CHECK-LABEL: @test_for_loop
  func.func public @test_for_loop(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> attributes {llvm.emit_c_interface} {
  // CHECK:     [[FOR_RESULT:.+]] = scf.for {{.*}} iter_args(%[[ARG:.+]] = {{.*}}) -> (f64) {
  // CHECK-NOT:   tensor<
  // CHECK:       tensor.from_elements %[[ARG]] : tensor<f64>
  // CHECK:       [[IF_RESULT:.+]] = scf.if {{.*}} -> (f64)
  // CHECK:         scf.yield {{.*}} : f64
  // CHECK-NOT:     tensor<
  // CHECK:       else
  // CHECK:         scf.yield {{.*}} : f64
  // CHECK-NOT:     tensor<
  // CHECK:       }
  // CHECK:       scf.yield {{.*}} : f64
  // CHECK-NOT:     tensor<
  // CHECK:     }
  // CHECK:     arith.mulf {{.*}} [[FOR_RESULT:.+]] : f64
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant dense<2> : tensor<i64>
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %extracted_2 = tensor.extract %arg1[] : tensor<f64>
    %0 = arith.addf %extracted, %extracted_2 : f64
    %from_elements = tensor.from_elements %0 : tensor<f64>
    %1 = scf.for %arg2 = %c0 to %c10 step %c1 iter_args(%arg3 = %from_elements) -> (tensor<f64>) {
      %extracted_5 = tensor.extract %arg3[] : tensor<f64>
      %3 = func.call @remainder(%arg3, %cst_1) : (tensor<f64>, tensor<i64>) -> tensor<f64>
      %extracted_6 = tensor.extract %3[] : tensor<f64>
      %4 = arith.cmpf une, %extracted_6, %cst : f64
      %5 = scf.if %4 -> (f64) {
        %6 = arith.subf %extracted_5, %extracted_5 : f64
        scf.yield %6 : f64
      } else {
        %6 = arith.addf %extracted_5, %cst_0 : f64
        scf.yield %6 : f64
      }
      %from_elements_7 = tensor.from_elements %5 : tensor<f64>
      scf.yield %from_elements_7 : tensor<f64>
    }
    %extracted_3 = tensor.extract %1[] : tensor<f64>
    %2 = arith.mulf %extracted, %extracted_3 : f64
    %from_elements_4 = tensor.from_elements %2 : tensor<f64>
    return %from_elements_4 : tensor<f64>
  }
  func.func private @remainder(%arg0: tensor<f64> {mhlo.layout_mode = "default"}, %arg1: tensor<i64> {mhlo.layout_mode = "default"}) -> (tensor<f64> {mhlo.layout_mode = "default"}) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f64
    %extracted = tensor.extract %arg1[] : tensor<i64>
    %extracted_0 = tensor.extract %arg0[] : tensor<f64>
    %0 = arith.sitofp %extracted : i64 to f64
    %1 = arith.remf %extracted_0, %0 : f64
    %from_elements = tensor.from_elements %1 : tensor<f64>
    return %from_elements : tensor<f64>
  }
}
