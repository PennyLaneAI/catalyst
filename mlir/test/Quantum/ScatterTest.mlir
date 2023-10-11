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

// RUN: quantum-opt %s --scatter-lowering --split-input-file --verify-diagnostics | FileCheck %s

func.func public @scatter_multiply(%arg0: tensor<3xf64>, %arg1: tensor<i64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %c0_i64 = arith.constant 0 : i64
    %c3_i64 = arith.constant 3 : i64
    %cst = arith.constant dense<2.000000e+00> : tensor<f64>
    %extracted = tensor.extract %arg1[] : tensor<i64>
    %0 = arith.cmpi slt, %extracted, %c0_i64 : i64
    %extracted_0 = tensor.extract %arg1[] : tensor<i64>
    %1 = arith.addi %extracted_0, %c3_i64 : i64
    %extracted_1 = tensor.extract %arg1[] : tensor<i64>
    %2 = arith.select %0, %1, %extracted_1 : i64
    %3 = arith.trunci %2 : i64 to i32
    %from_elements = tensor.from_elements %3 : tensor<1xi32>
    %4 = "mhlo.scatter"(%arg0, %from_elements, %cst) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %extracted_2 = tensor.extract %arg2[] : tensor<f64>
      %extracted_3 = tensor.extract %arg3[] : tensor<f64>
      %5 = arith.mulf %extracted_2, %extracted_3 : f64
      %from_elements_4 = tensor.from_elements %5 : tensor<f64>
      mhlo.return %from_elements_4 : tensor<f64>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xf64>, tensor<1xi32>, tensor<f64>) -> tensor<3xf64>
    return %4 : tensor<3xf64>
  }

// CHECK: func.func private @__catalyst_update_scatter[[NUMBER:.*]](%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64>
    // CHECK-NEXT:   [[EXTRACTED0:%.+]] = tensor.extract %arg0[] : tensor<f64>
    // CHECK-NEXT:   [[EXTRACTED1:%.+]] = tensor.extract %arg1[] : tensor<f64>
    // CHECK-NEXT:   [[RES:%.+]] = arith.mulf [[EXTRACTED0]], [[EXTRACTED1]] : f64
    // CHECK-NEXT:   [[RESTENSOR:%.+]] = tensor.from_elements [[RES]] : tensor<f64>
    // CHECK-NEXT:   return [[RESTENSOR]] : tensor<f64>

// CHECK: func.func public @scatter_multiply(%arg0: tensor<3xf64>, %arg1: tensor<i64>) -> tensor<3xf64>
    // CHECK:    [[IDX0:%.+]] = index.constant 0
    // CHECK:    [[IDX1:%.+]] = index.constant 1
    // CHECK:    [[CST:%.+]] = arith.constant dense<2.000000e+00> : tensor<f64>
    // CHECK:    [[SCF:%.+]] = scf.for %arg2 = [[IDX0]] to [[IDX1]] step [[IDX1]] iter_args(%arg3 = %arg0) -> (tensor<3xf64>)
    // CHECK:    [[INDEX:%.+]] = arith.index_cast %6 : i32 to index
    // CHECK:    [[EXTRACTED:%.+]] = tensor.extract %arg3[[[INDEX]]] : tensor<3xf64>
    // CHECK:    [[FROM:%.+]] = tensor.from_elements [[EXTRACTED]] : tensor<f64>
    // CHECK:    [[CALL:%.+]] = func.call @__catalyst_update_scatter[[NUMBER]]([[FROM]], [[CST]]) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    // CHECK:    [[RES:%.+]] = tensor.extract [[CALL]][] : tensor<f64>
    // CHECK:    [[INSERTED:%.+]] = tensor.insert [[RES]] into %arg3[[[INDEX]]] : tensor<3xf64>
    // CHECK:    scf.yield [[INSERTED]] : tensor<3xf64>
    // CHECK:    return [[SCF]] : tensor<3xf64>

// -----

#map = affine_map<(d0) -> (d0)>
func.func public @two_scatter(%arg0: tensor<3xf64>, %arg1: tensor<i64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
  %c0_i64 = arith.constant 0 : i64
  %c3_i64 = arith.constant 3 : i64
  %cst = arith.constant dense<2.000000e+00> : tensor<f64>
  %cst_0 = arith.constant dense<3.000000e+00> : tensor<f64>
  %extracted = tensor.extract %arg1[] : tensor<i64>
  %0 = arith.cmpi slt, %extracted, %c0_i64 : i64
  %extracted_1 = tensor.extract %arg1[] : tensor<i64>
  %1 = arith.addi %extracted_1, %c3_i64 : i64
  %extracted_2 = tensor.extract %arg1[] : tensor<i64>
  %2 = arith.select %0, %1, %extracted_2 : i64
  %3 = arith.trunci %2 : i64 to i32
  %from_elements = tensor.from_elements %3 : tensor<1xi32>
  %4 = "mhlo.scatter"(%arg0, %from_elements, %cst_0) ({
  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
    %extracted_7 = tensor.extract %arg2[] : tensor<f64>
    %extracted_8 = tensor.extract %arg3[] : tensor<f64>
    %12 = arith.mulf %extracted_7, %extracted_8 : f64
    %from_elements_9 = tensor.from_elements %12 : tensor<f64>
    mhlo.return %from_elements_9 : tensor<f64>
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xf64>, tensor<1xi32>, tensor<f64>) -> tensor<3xf64>
  %extracted_3 = tensor.extract %arg1[] : tensor<i64>
  %5 = arith.cmpi slt, %extracted_3, %c0_i64 : i64
  %extracted_4 = tensor.extract %arg1[] : tensor<i64>
  %6 = arith.addi %extracted_4, %c3_i64 : i64
  %extracted_5 = tensor.extract %arg1[] : tensor<i64>
  %7 = arith.select %5, %6, %extracted_5 : i64
  %8 = arith.trunci %7 : i64 to i32
  %from_elements_6 = tensor.from_elements %8 : tensor<1xi32>
  %9 = "mhlo.scatter"(%arg0, %from_elements_6, %cst) ({
  ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
    %extracted_7 = tensor.extract %arg2[] : tensor<f64>
    %extracted_8 = tensor.extract %arg3[] : tensor<f64>
    %12 = arith.addf %extracted_7, %extracted_8 : f64
    %from_elements_9 = tensor.from_elements %12 : tensor<f64>
    mhlo.return %from_elements_9 : tensor<f64>
  }) {indices_are_sorted = true, scatter_dimension_numbers = #mhlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xf64>, tensor<1xi32>, tensor<f64>) -> tensor<3xf64>
  %10 = tensor.empty() : tensor<3xf64>
  %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %9 : tensor<3xf64>, tensor<3xf64>) outs(%10 : tensor<3xf64>) {
  ^bb0(%in: f64, %in_7: f64, %out: f64):
    %12 = arith.addf %in, %in_7 : f64
    linalg.yield %12 : f64
  } -> tensor<3xf64>
  return %11 : tensor<3xf64>
}

// CHECK: func.func private @__catalyst_update_scatter[[NUMBER0:.*]](%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64>
    // CHECK-NEXT:   [[EXTRACTED0:%.+]] = tensor.extract %arg0[] : tensor<f64>
    // CHECK-NEXT:   [[EXTRACTED1:%.+]] = tensor.extract %arg1[] : tensor<f64>
    // CHECK-NEXT:   [[RES:%.+]] = arith.mulf [[EXTRACTED0]], [[EXTRACTED1]] : f64
    // CHECK-NEXT:   [[RESTENSOR:%.+]] = tensor.from_elements [[RES]] : tensor<f64>
    // CHECK-NEXT:   return [[RESTENSOR]] : tensor<f64>

// CHECK: func.func private @__catalyst_update_scatter[[NUMBER1:.*]](%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64>
    // CHECK-NEXT:   [[EXTRACTED2:%.+]] = tensor.extract %arg0[] : tensor<f64>
    // CHECK-NEXT:   [[EXTRACTED3:%.+]] = tensor.extract %arg1[] : tensor<f64>
    // CHECK-NEXT:   [[RES1:%.+]] = arith.addf [[EXTRACTED2]], [[EXTRACTED3]] : f64
    // CHECK-NEXT:   [[RESTENSOR1:%.+]] = tensor.from_elements [[RES1]] : tensor<f64>
    // CHECK-NEXT:   return [[RESTENSOR1]] : tensor<f64>

// CHECK: func.func public @two_scatter(%arg0: tensor<3xf64>, %arg1: tensor<i64>) -> tensor<3xf64>
    // CHECK:    [[IDX0:%.+]] = index.constant 0
    // CHECK:    [[IDX1:%.+]] = index.constant 1
    // CHECK:    [[CST0:%.+]] = arith.constant dense<2.000000e+00> : tensor<f64>
    // CHECK:    [[CST1:%.+]] = arith.constant dense<3.000000e+00> : tensor<f64>

    // CHECK:    [[SCF0:%.+]] = scf.for %arg2 = [[IDX0]] to [[IDX1]] step [[IDX1]] iter_args(%arg3 = %arg0) -> (tensor<3xf64>)
    // CHECK:    [[INDEX:%.+]] = arith.index_cast %13 : i32 to index
    // CHECK:    [[EXTRACTED:%.+]] = tensor.extract %arg3[[[INDEX]]] : tensor<3xf64>
    // CHECK:    [[FROM:%.+]] = tensor.from_elements [[EXTRACTED]] : tensor<f64>
    // CHECK:    [[CALL:%.+]] = func.call @__catalyst_update_scatter[[NUMBER0]]([[FROM]], [[CST1]]) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    // CHECK:    [[RES:%.+]] = tensor.extract [[CALL]][] : tensor<f64>
    // CHECK:    [[INSERTED:%.+]] = tensor.insert [[RES]] into %arg3[[[INDEX]]] : tensor<3xf64>
    // CHECK:    scf.yield [[INSERTED]] : tensor<3xf64>

    // CHECK:    [[SCF1:%.+]] = scf.for %arg2 = [[IDX0]] to [[IDX1]] step [[IDX1]] iter_args(%arg3 = %arg0) -> (tensor<3xf64>)
    // CHECK:    [[INDEX:%.+]] = arith.index_cast %13 : i32 to index
    // CHECK:    [[EXTRACTED:%.+]] = tensor.extract %arg3[[[INDEX]]] : tensor<3xf64>
    // CHECK:    [[FROM:%.+]] = tensor.from_elements [[EXTRACTED]] : tensor<f64>
    // CHECK:    [[CALL:%.+]] = func.call @__catalyst_update_scatter[[NUMBER1]]([[FROM]], [[CST0]]) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    // CHECK:    [[RES:%.+]] = tensor.extract [[CALL]][] : tensor<f64>
    // CHECK:    [[INSERTED:%.+]] = tensor.insert [[RES]] into %arg3[[[INDEX]]] : tensor<3xf64>
    // CHECK:    scf.yield [[INSERTED]] : tensor<3xf64>

// -----

func.func public @full_example_scatter(%input: tensor<3x4x2xi64>, %update: tensor<2x3x2x2xi64>) -> tensor<3x4x2xi64> attributes {llvm.emit_c_interface} {
  %scatter_indices = arith.constant dense<2> : tensor<2x3x2xi32>
  %result = "mhlo.scatter"(%input, %scatter_indices, %update) ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %extracted_1 = tensor.extract %arg2[] : tensor<i64>
      %extracted_2 = tensor.extract %arg3[] : tensor<i64>
      %1 = arith.addi %extracted_1, %extracted_2 : i64
      %from_elements = tensor.from_elements %1 : tensor<i64>
      mhlo.return %from_elements : tensor<i64>
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 2>,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<3x4x2xi64>, tensor<2x3x2xi32>, tensor<2x3x2x2xi64>) -> tensor<3x4x2xi64>
  return %result : tensor<3x4x2xi64>
}

// CHECK: func.func private @__catalyst_update_scatter[[NUMBER0:.*]](%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64>
    // CHECK-NEXT:   [[EXTRACTED0:%.+]] = tensor.extract %arg0[] : tensor<i64>
    // CHECK-NEXT:   [[EXTRACTED1:%.+]] = tensor.extract %arg1[] : tensor<i64>
    // CHECK-NEXT:   [[RES:%.+]] = arith.addi [[EXTRACTED0]], [[EXTRACTED1]] : i64
    // CHECK-NEXT:   [[RESTENSOR:%.+]] = tensor.from_elements [[RES]] : tensor<i64>
    // CHECK-NEXT:   return [[RESTENSOR]] : tensor<i64>

    // CHECK:    [[CST:%.+]] = arith.constant dense<[[DENSE:.*]]> : tensor<24x4xindex>
    // CHECK:    [[IDX0:%.+]] = index.constant 0
    // CHECK:    [[IDX24:%.+]] = index.constant 24
    // CHECK:    [[IDX1:%.+]] = index.constant 1
    // CHECK:    [[SCF:%.+]] = scf.for %arg2 = [[IDX0]] to [[IDX24]] step [[IDX1]] iter_args(%arg3 = %arg0) -> (tensor<3x4x2xi64>)
        // CHECK:    [[EXTRACT:%.+]] = tensor.extract_slice [[CST]][%arg2, 0] [1, 4] [1, 1] : tensor<24x4xindex> to tensor<4xindex>
        // CHECK:    [[EXTRACT0:%.+]] = tensor.extract %arg1[%extracted_1, %extracted_2, %extracted_3, %extracted_4] : tensor<2x3x2x2xi64>
        // CHECK:    [[EXTRACT1:%.+]] = tensor.extract %arg3[%3, %6, %8] : tensor<3x4x2xi64>
        // CHECK:    [[FROM0:%.+]] = tensor.from_elements [[EXTRACT0]] : tensor<i64>
        // CHECK:    [[FROM1:%.+]] = tensor.from_elements [[EXTRACT1]] : tensor<i64>
        // CHECK:    [[CALL:%.+]] = func.call @__catalyst_update_scatter[[NUMBER0]]([[FROM1]], [[FROM0]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
        // CHECK:    [[RES:%.+]] = tensor.extract [[CALL]][] : tensor<i64>
        // CHECK:    [[INSERTED:%.+]] = tensor.insert [[RES]] into %arg3[%3, %6, %8] : tensor<3x4x2xi64>
        // CHECK:    scf.yield [[INSERTED]] : tensor<3x4x2xi64>
    // CHECK:    return [[SCF]] : tensor<3x4x2xi64>
