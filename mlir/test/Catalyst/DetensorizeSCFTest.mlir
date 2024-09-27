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

// CHECK-LABEL: @test_for_loop
// CHECK-NOT:     scf.for {{.*}} -> (tensor<f64>)
// CHECK:         [[FOR_RES:%.+]] = scf.for {{.*}} -> (f64)
// CHECK:           arith.addf
// CHECK:           scf.yield {{.*}} : f64
// CHECK-NOT:     scf.for {{.*}} -> (tensor<f64>)
// CHECK:         [[FUN_RES:%.+]] = tensor.from_elements [[FOR_RES]]
// CHECK:         return [[FUN_RES]] : tensor<f64>
func.func public @test_for_loop(%arg0: tensor<f64>) -> tensor<f64> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index

  %0 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %arg0) -> (tensor<f64>) {
    %extracted = tensor.extract %arg2[] : tensor<f64>
    %1 = arith.addf %extracted, %extracted : f64
    %from_elements = tensor.from_elements %1 : tensor<f64>
    scf.yield %from_elements : tensor<f64>
  }

  return %0 : tensor<f64>
}

// -----

// CHECK-LABEL: @test_if_mixed_return_types
// CHECK-NOT:     scf.if {{.*}} -> (f64, tensor<f64>)
// CHECK:         [[IF_RES:%.*]]:2 = scf.if {{.*}} -> (f64, f64)
// CHECK:           arith.addf
// CHECK:           arith.subf
// CHECK:           scf.yield {{.*}} : f64, f64
// CHECK:         else
// CHECK:           arith.subf
// CHECK:           scf.yield {{.*}} : f64, f64
// CHECK-NOT:     scf.if {{.*}} -> (f64, tensor<f64>)
// CHECK:         [[FUN_RES:%.+]] = tensor.from_elements [[IF_RES]]#1
// CHECK:         return [[IF_RES]]#0, [[FUN_RES]] : f64, tensor<f64>
func.func public @test_if_mixed_return_types(%arg0: f64, %arg1: f64) -> (f64, tensor<f64>) {
  %0 = arith.cmpf ogt, %arg0, %arg1 : f64

  %1:2 = scf.if %0 -> (f64, tensor<f64>) {
    %2 = arith.addf %arg0, %arg1 : f64
    %3 = arith.subf %arg0, %arg1 : f64
    %from_elements_0 = tensor.from_elements %3 : tensor<f64>
    scf.yield %2, %from_elements_0 : f64, tensor<f64>
  } else {
    %3 = arith.subf %arg1, %arg0 : f64
    %from_elements_0 = tensor.from_elements %3 : tensor<f64>
    scf.yield %arg0, %from_elements_0 : f64, tensor<f64>
  }

  return %1#0, %1#1 : f64, tensor<f64>
}

// -----

// CHECK-LABEL: @test_while_loop
// CHECK-NOT:     scf.while {{.*}} -> (tensor<i64>)
// CHECK:         [[WHILE_RES:%.*]] = scf.while {{.*}} (i64) -> i64
// CHECK:           arith.cmpi
// CHECK:           scf.condition{{.*}} : i64
// CHECK:         do
// CHECK:         ^bb0({{.*}}: i64)
// CHECK:           arith.addi
// CHECK:           scf.yield {{.*}} : i64
// CHECK-NOT:     scf.while {{.*}} -> (tensor<i64>)
// CHECK:         [[FUN_RES:%.+]] = tensor.from_elements [[WHILE_RES]]
// CHECK:         return [[FUN_RES]] : tensor<i64>
func.func public @test_while_loop(%arg0: tensor<i64>) -> tensor<i64> {
  %cst = arith.constant dense<10> : tensor<i64>

  %0 = scf.while (%arg1 = %arg0) : (tensor<i64>) -> (tensor<i64>) {
    %1 = arith.cmpi slt, %arg1, %cst : tensor<i64>
    %extracted = tensor.extract %1[] : tensor<i1>
    scf.condition(%extracted) %arg1 : tensor<i64>
  } do {
  ^bb0(%arg2: tensor<i64>):
    %2 = arith.addi %arg2, %cst : tensor<i64>
    scf.yield %2 : tensor<i64>
  }

  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func.func public @test_nested_ifs
// CHECK-NOT:     scf.if {{.*}} -> (tensor<f64>)
// CHECK:         [[IF_RES:%.+]] = scf.if {{.*}} -> (f64)
// CHECK:           arith.cmpf
// CHECK-NOT:       scf.if {{.*}} -> (tensor<f64>)
// CHECK:           [[IF1_RES:%.+]] = scf.if {{.*}} -> (f64)
// CHECK:             arith.mulf
// CHECK:             scf.yield {{.*}} : f64
// CHECK:           else
// CHECK:             scf.yield {{.*}} : f64
// CHECK-NOT:       scf.if {{.*}} -> (tensor<f64>)
// CHECK:           scf.yield [[IF1_RES]] : f64
// CHECK:         else
// CHECK:           scf.yield {{.*}} : f64
// CHECK-NOT:     scf.if {{.*}} -> (tensor<f64>)
// CHECK:         [[FUN_RES:%.+]] = tensor.from_elements [[IF_RES]]
// CHECK:         return [[FUN_RES]] : tensor<f64>
func.func public @test_nested_ifs(%arg0: tensor<f64>) -> tensor<f64> {
  %cst = arith.constant 2.000000e+00 : f64
  %extracted = tensor.extract %arg0[] : tensor<f64>
  %0 = arith.cmpf ogt, %extracted, %cst : f64

  %1 = scf.if %0 -> (tensor<f64>) {
    %2 = arith.cmpf ogt, %cst, %extracted : f64

    %3 = scf.if %2 -> (tensor<f64>) {
      %4 = arith.mulf %arg0, %arg0 : tensor<f64>
      scf.yield %4 : tensor<f64>
    } else {
      %from_elements = tensor.from_elements %cst : tensor<f64>
      scf.yield %from_elements : tensor<f64>
    }

    scf.yield %3 : tensor<f64>
  } else {
    scf.yield %arg0 : tensor<f64>
  }

  return %1 : tensor<f64>
}
