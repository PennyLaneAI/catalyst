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

// RUN: quantum-opt --split-input-file --gradient-postprocess %s | FileCheck %s


// CHECK: func.func private @callback_fn_fwd(tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
func.func private @callback_fn_fwd(tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)

// CHECK: gradient.forward @callback_fn_fwd.fwd(%arg0: memref<2xf64>, %arg1: memref<2xf64>, %arg2: memref<f64>, %arg3: memref<f64>) -> memref<2xf64>
// CHECK-SAME: attributes {argc = 1 : i64, implementation = @callback_fn_fwd, resc = 1 : i64, tape = 1 : i64}
// CHECK-SAME: {
gradient.forward @callback_fn_fwd.fwd(%arg0: memref<2xf64>) -> (memref<f64>, memref<2xf64>) attributes {argc = 1 : i64, implementation = @callback_fn_fwd, resc = 1 : i64, tape = 1 : i64} {

    // CHECK: [[in:%.+]] = bufferization.to_tensor %arg0 : memref<2xf64> to tensor<2xf64>
    // CHECK: [[callOut:%.+]]:2 = func.call @callback_fn_fwd([[in]]) : (tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
    // CHECK: [[res0:%.+]] = bufferization.to_memref [[callOut]]#0 : tensor<f64> to memref<f64>
    // CHECK: [[res1:%.+]] = bufferization.to_memref [[callOut]]#1 : tensor<2xf64> to memref<2xf64>
    // CHECK: memref.copy [[res0]], %arg2 : memref<f64> to memref<f64>
    // CHECK: gradient.return {empty = false} [[res1]] : memref<2xf64>

	%0 = bufferization.to_tensor %arg0 : memref<2xf64> to tensor<2xf64>
	%1:2 = func.call @callback_fn_fwd(%0) : (tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
	%2 = bufferization.to_memref %1#0 : tensor<f64> to memref<f64>
	%3 = bufferization.to_memref %1#1 : tensor<2xf64> to memref<2xf64>
	gradient.return {empty = false} %2, %3 : memref<f64>, memref<2xf64>
}

// -----

// CHECK: func.func private @callback_fn_vjp(tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
func.func private @callback_fn_vjp(tensor<2xf64>, tensor<f64>) -> tensor<2xf64>

// CHECK: gradient.reverse @callback_fn_vjp.rev(%arg0: memref<2xf64>, %arg1: memref<2xf64>, %arg2: memref<f64>, %arg3: memref<f64>, %arg4: memref<2xf64>)
// CHECK-SAME: attributes {argc = 1 : i64, implementation = @callback_fn_vjp, resc = 1 : i64, tape = 1 : i64}
// CHECK-SAME: {
gradient.reverse @callback_fn_vjp.rev(%arg0: memref<f64>, %arg1: memref<2xf64>) -> memref<2xf64> attributes {argc = 1 : i64, implementation = @callback_fn_vjp, resc = 1 : i64, tape = 1 : i64} {

    // CHECK: [[tape:%.+]] = bufferization.to_tensor %arg4 : memref<2xf64> to tensor<2xf64>
    // CHECK: [[cotan:%.+]] = bufferization.to_tensor %arg3 : memref<f64> to tensor<f64>
    // CHECK: [[callOut:%.+]] = func.call @callback_fn_vjp([[tape]], [[cotan]]) : (tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
    // CHECK: [[res:%.+]] = bufferization.to_memref [[callOut]] : tensor<2xf64> to memref<2xf64>
    // CHECK: memref.copy [[res]], %arg1 : memref<2xf64> to memref<2xf64>
    // CHECK: gradient.return {empty = true}

	%0 = bufferization.to_tensor %arg1 : memref<2xf64> to tensor<2xf64>
	%1 = bufferization.to_tensor %arg0 : memref<f64> to tensor<f64>
	%2 = func.call @callback_fn_vjp(%0, %1) : (tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
	%3 = bufferization.to_memref %2 : tensor<2xf64> to memref<2xf64>
	gradient.return {empty = true} %3 : memref<2xf64>
}
