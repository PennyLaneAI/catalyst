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

// RUN: quantum-opt %s --allow-unregistered-dialect --scatter-lowering --split-input-file --verify-diagnostics | FileCheck %s

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