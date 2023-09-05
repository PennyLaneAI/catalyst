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

func.func public @jit_test(%arg0: tensor<3xf64>, %arg1: tensor<i64>) -> tensor<3xf64> attributes {llvm.emit_c_interface} {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %2 = stablehlo.constant dense<3> : tensor<i64>
    %3 = stablehlo.add %arg1, %2 : tensor<i64>
    %4 = stablehlo.select %1, %3, %arg1 : tensor<i1>, tensor<i64>
    %5 = stablehlo.convert %4 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %8 = "stablehlo.scatter"(%arg0, %6, %7) ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
        %9 = stablehlo.multiply %arg2, %arg3 : tensor<f64>
        stablehlo.return %9 : tensor<f64>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true} : (tensor<3xf64>, tensor<1xi32>, tensor<f64>) -> tensor<3xf64>
    return %8 : tensor<3xf64>
    }
