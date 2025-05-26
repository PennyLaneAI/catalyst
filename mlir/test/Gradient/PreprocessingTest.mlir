// Copyright 2024-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --split-input-file --gradient-preprocess %s | FileCheck %s


func.func private @callback_fn_fwd(tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
gradient.forward @callback_fn_fwd.fwd(tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>) attributes {argc = 1 : i64, implementation = @callback_fn_fwd, resc = 1 : i64, tape = 1 : i64}

// CHECK:   func.func private @callback_fn_fwd(tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
//
// CHECK:   gradient.forward @callback_fn_fwd.fwd(%arg0: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
// CHECK-SAME:  attributes {argc = 1 : i64, implementation = @callback_fn_fwd, resc = 1 : i64, tape = 1 : i64} {
// CHECK:     [[res:%.+]]:2 = func.call @callback_fn_fwd(%arg0) : (tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
// CHECK:     gradient.return {empty = false} [[res]]#0, [[res]]#1 : tensor<f64>, tensor<2xf64>
// CHECK:   }

// -----

func.func private @callback_fn_vjp(tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
gradient.reverse @callback_fn_vjp.rev(tensor<f64>, tensor<2xf64>) -> tensor<2xf64> attributes {argc = 1 : i64, implementation = @callback_fn_vjp, resc = 1 : i64, tape = 1 : i64}

// CHECK:   func.func private @callback_fn_vjp(tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
//
// CHECK:   gradient.reverse @callback_fn_vjp.rev(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<2xf64>
// CHECK-SAME:  attributes {argc = 1 : i64, implementation = @callback_fn_vjp, resc = 1 : i64, tape = 1 : i64} {
// CHECK:     [[res:%.+]] = func.call @callback_fn_vjp(%arg1, %arg0) : (tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
// CHECK:     gradient.return {empty = true} [[res]] : tensor<2xf64>
// CHECK:   }
