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

// RUN: quantum-opt %s --lower-gradients | FileCheck %s

func.func private @func1(tensor<4xf64>) -> tensor<3x4xf64>

func.func @jvptest1(
    %arg0: tensor<4xf64>
  , %arg1: tensor<4xf64>
  ) -> (tensor<3x4xf64>, tensor<3x4xf64>)
        attributes {llvm.emit_c_interface}
{
  // CHECK: call @func1({{[%a-z0-9, ]+}}) : (tensor<4xf64>) -> tensor<3x4xf64>
  // CHECK: linalg.generic {{{[^}]*}}}  ins({{[^:]*}} : tensor<3x4x4xf64>, tensor<4xf64>) outs({{[^:]*}} : tensor<3x4xf64>)
  // CHECK: return {{[^:]+}} : tensor<3x4xf64>, tensor<3x4xf64>
  %0:2 = "gradient.jvp"(%arg0, %arg1) {
      callee = @func1
    , diffArgIndices = dense<0> : tensor<1xi64>
    , finiteDiffParam = 9.9999999999999995E-8 : f64
    , operand_segment_sizes = array<i32: 1, 1>
    , method = "auto"
  } : (tensor<4xf64>, tensor<4xf64>) -> (tensor<3x4xf64>, tensor<3x4xf64>)
  return %0#0, %0#1 : tensor<3x4xf64>, tensor<3x4xf64>
}

func.func private @func2(%arg0: tensor<3x2xf64>, %arg1: tensor<2x3xf64>) -> (tensor<6xf64>, tensor<2x6xf64>)

func.func public @jvptest2(
    %arg0: tensor<3x2xf64>
  , %arg1: tensor<2x3xf64>
  , %arg2: tensor<3x2xf64>
  , %arg3: tensor<2x3xf64>
  ) -> (tensor<6xf64>, tensor<2x6xf64>, tensor<6xf64>, tensor<2x6xf64>)
  attributes {llvm.emit_c_interface}
{
  // CHECK:      call @func2
  // CHECK-SAME:     : (tensor<3x2xf64>, tensor<2x3xf64>) -> (tensor<6xf64>, tensor<2x6xf64>)

  // CHECK:      linalg.generic
  // CHECK-SAME:     ins({{[^:]*}} : tensor<6x3x2xf64>, tensor<3x2xf64>)
  // CHECK-SAME:     outs({{[^:]*}} : tensor<6xf64>)

  // CHECK:      linalg.generic
  // CHECK-SAME:     ins({{[^:]*}} : tensor<6x2x3xf64>, tensor<2x3xf64>)
  // CHECK-SAME:     outs({{[^:]*}} : tensor<6xf64>)

  // CHECK:      linalg.generic
  // CHECK-SAME:     ins({{[^:]*}} : tensor<2x6x3x2xf64>, tensor<3x2xf64>)
  // CHECK-SAME:     outs({{[^:]*}} : tensor<2x6xf64>)

  // CHECK:      linalg.generic
  // CHECK-SAME:     ins({{[^:]*}} : tensor<2x6x2x3xf64>, tensor<2x3xf64>)
  // CHECK-SAME:     outs({{[^:]*}} : tensor<2x6xf64>)

  // CHECK:      return
  // CHECK-SAME:     : tensor<6xf64>, tensor<2x6xf64>, tensor<6xf64>, tensor<2x6xf64>
  %0:4 = "gradient.jvp"(%arg0, %arg1, %arg2, %arg3) {
      callee = @func2
    , diffArgIndices = dense<[0, 1]> : tensor<2xi64>
    , finiteDiffParam = 9.9999999999999995E-8 : f64
    , operand_segment_sizes = array<i32: 2, 2>
    , method = "auto"
    } : (tensor<3x2xf64>, tensor<2x3xf64>, tensor<3x2xf64>, tensor<2x3xf64>) -> (tensor<6xf64>, tensor<2x6xf64>, tensor<6xf64>, tensor<2x6xf64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<6xf64>, tensor<2x6xf64>, tensor<6xf64>, tensor<2x6xf64>
}





