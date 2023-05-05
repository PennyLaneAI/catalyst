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

// RUN: quantum-opt %s --lower-jvp-vjp | FileCheck %s

func.func private @func1(tensor<4xf64>) -> tensor<3x4xf64>


// CHECK: func.func @jvptest1(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> (tensor<3x4xf64>, tensor<3x4xf64>) attributes {llvm.emit_c_interface} {
// CHECK:    %cst = arith.constant 0.000000e+00 : f64
// CHECK:    %0 = call @func1(%arg0) : (tensor<4xf64>) -> tensor<3x4xf64>
// CHECK:    %1 = gradient.grad "fd" @func1(%arg0) {diffArgIndices = dense<0> : tensor<1xi64>, finiteDiffParam = 9.9999999999999995E-8 : f64} : (tensor<4xf64>) -> tensor<4x3x4xf64>
// CHECK:    %2 = tensor.empty() : tensor<3x4xf64>
// CHECK:    %3 = linalg.fill ins(%cst : f64) outs(%2 : tensor<3x4xf64>) -> tensor<3x4xf64>
// CHECK:    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel"]} ins(%1, %arg1 : tensor<4x3x4xf64>, tensor<4xf64>) outs(%3 : tensor<3x4xf64>) {
// CHECK:    ^bb0(%in: f64, %in_0: f64, %out: f64):
// CHECK:      %5 = arith.mulf %in, %in_0 : f64
// CHECK:      %6 = arith.addf %out, %5 : f64
// CHECK:      linalg.yield %6 : f64
// CHECK:    } -> tensor<3x4xf64>
// CHECK:    return %0, %4 : tensor<3x4xf64>, tensor<3x4xf64>
// CHECK:  }

func.func @jvptest1(
  %arg0: tensor<4xf64>,
  %arg1: tensor<4xf64>
  ) -> (tensor<3x4xf64>, tensor<3x4xf64>)
        attributes {llvm.emit_c_interface}
{
  %0:2 = "gradient.jvp"(%arg0, %arg1) {
    callee = @func1,
    diffArgIndices = dense<0> : tensor<1xi64>,
    finiteDiffParam = 9.9999999999999995E-8 : f64,
    method = "fd"
  } : (tensor<4xf64>, tensor<4xf64>) -> (tensor<3x4xf64>, tensor<3x4xf64>)

  return %0#0, %0#1 : tensor<3x4xf64>, tensor<3x4xf64>
}

