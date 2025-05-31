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

// RUN: quantum-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     one-shot-bufferize{ \
// RUN:        unknown-type-conversion=identity-layout-map \
// RUN:        function-boundary-type-conversion=identity-layout-map \
// RUN:   })" %s | FileCheck %s

//////////////////////
// Native Gradients //
//////////////////////

func.func private @circuit(%arg0: f64)

// CHECK-LABEL: @adjoint
func.func @adjoint(%arg0: f64, %arg1: index) {

    // CHECK:   [[alloc:%.+]] = memref.alloc(%arg1) : memref<?xf64>
    // CHECK:   gradient.adjoint @circuit(%arg0) size(%arg1) in([[alloc]] : memref<?xf64>) : (f64) -> ()
    %grad = gradient.adjoint @circuit(%arg0) size(%arg1) : (f64) -> tensor<?xf64>
    return
}

// -----

func.func private @circuit(%arg0: f64)

// CHECK-LABEL: @adjoint_with_scalar_return
func.func @adjoint_with_scalar_return(%arg0: f64, %arg1: index) {

    // CHECK:   [[alloc:%.+]] = memref.alloc(%arg1) : memref<?xf64>
    // CHECK:   [[bufferedRes:%.+]]:2 = gradient.adjoint @circuit(%arg0) size(%arg1) in([[alloc]] : memref<?xf64>)
    // CHECK-SAME: (f64) -> (f64, f64)
    %grad:3 = gradient.adjoint @circuit(%arg0) size(%arg1) : (f64) -> (f64, tensor<?xf64>, f64)

    // CHECK: [[toTen:%.+]] = bufferization.to_tensor [[alloc]] : memref<?xf64>
    // CHECK: math.sin [[toTen]]
    %user0 = math.sin %grad#1 : tensor<?xf64>

    // CHECK: arith.addf [[bufferedRes]]#0, [[bufferedRes]]#1
    %user1 = arith.addf %grad#0, %grad#2 : f64

    return
}

// -----

func.func private @circuit(%arg0: tensor<2xf64>)

// CHECK-LABEL: @adjoint_with_tensor_arg
func.func @adjoint_with_tensor_arg(%arg0: tensor<2xf64>, %arg1: index) {

    // CHECK:   [[argBuffer:%.+]] = bufferization.to_memref %arg0 : tensor<2xf64> to memref<2xf64>
    // CHECK:   [[alloc:%.+]] = memref.alloc(%arg1) : memref<?xf64>
    // CHECK:   gradient.adjoint @circuit([[argBuffer]]) size(%arg1) in([[alloc]] : memref<?xf64>) : (memref<2xf64>) -> ()
    %grad = gradient.adjoint @circuit(%arg0) size(%arg1) : (tensor<2xf64>) -> tensor<?xf64>
    return
}

// -----

func.func private @circuit(%arg0: tensor<2xf64>)

// CHECK-LABEL: @adjoint_with_multiple_results
func.func @adjoint_with_multiple_results(%arg0: tensor<2xf64>, %arg1: index) {

    // CHECK:   [[argBuffer:%.+]] = bufferization.to_memref %arg0 : tensor<2xf64> to memref<2xf64>
    // CHECK:   [[alloc0:%.+]] = memref.alloc(%arg1) : memref<?xf64>
    // CHECK:   [[alloc1:%.+]] = memref.alloc(%arg1) : memref<?xf32>
    // CHECK:   gradient.adjoint @circuit([[argBuffer]]) size(%arg1) in([[alloc0]], [[alloc1]]
    // CHECK-SAME: memref<?xf64>, memref<?xf32>) : (memref<2xf64>) -> ()
    %grad:2 = gradient.adjoint @circuit(%arg0) size(%arg1) : (tensor<2xf64>) -> (tensor<?xf64>, tensor<?xf32>)
    return
}

// -----

func.func private @circuit(%arg0: f64)

// CHECK-LABEL: @backprop_scalar_in
func.func @backprop_scalar_in(%arg0: f64, %arg1: tensor<?xf64>) {

    // CHECK:   [[cotangentSource:%.+]] = bufferization.to_memref %arg1 : tensor<?xf64> to memref<?xf64>
    // CHECK:   [[dim1:%.+]] = memref.dim [[cotangentSource]]
    // CHECK:   [[cotangentRes:%.+]] = memref.alloc([[dim1]]) {alignment = 64 : i64} : memref<?xf64>
    // CHECK:   memref.copy [[cotangentSource]], [[cotangentRes]]
    // CHECK:   [[dim:%.+]] = memref.dim [[cotangentRes]]
    // CHECK:   [[calleeRes:%.+]] = memref.alloc([[dim]]) : memref<?xf64>
    // CHECK:   {{%.+}} = gradient.backprop @circuit(%arg0)
    // CHECK-SAME:  callee_out([[calleeRes]] : memref<?xf64>)
    // CHECK-SAME:  cotangents([[cotangentRes]] : memref<?xf64>)
    // CHECK-SAME:  {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 1>}
    // CHECK-SAME:  : (f64) -> f64
    %grad = gradient.backprop @circuit(%arg0) cotangents(%arg1: tensor<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 1>}: (f64) -> f64
    return
}

// -----

func.func private @circuit(%arg0: tensor<?x2xf64>)

// CHECK-LABEL: @backprop_tensor_in
func.func @backprop_tensor_in(%arg0: tensor<?x2xf64>, %arg1: tensor<?xf64>) {

    // CHECK-DAG:   [[argSource:%.+]] = bufferization.to_memref %arg0 : tensor<?x2xf64> to memref<?x2xf64>
    // CHECK-DAG:   [[cotangentSource:%.+]] = bufferization.to_memref %arg1 : tensor<?xf64> to memref<?xf64>
    // CHECK:   [[dim2:%.+]] = memref.dim [[cotangentSource]]
    // CHECK:   [[cotangentRes:%.+]] = memref.alloc([[dim2]]) {alignment = 64 : i64} : memref<?xf64>
    // CHECK:   memref.copy [[cotangentSource]], [[cotangentRes]]
    // CHECK:   [[dim:%.+]] = memref.dim [[argSource]]
    // CHECK:   [[argShadow:%.+]] = memref.alloc([[dim]]) : memref<?x2xf64>
    // CHECK:   [[dim1:%.+]] = memref.dim [[cotangentRes]]
    // CHECK:   [[calleeRes:%.+]] = memref.alloc([[dim1]]) : memref<?xf64>
    // CHECK:   gradient.backprop @circuit([[argSource]])
    // CHECK-SAME:  grad_out([[argShadow]] : memref<?x2xf64>)
    // CHECK-SAME:  callee_out([[calleeRes]] : memref<?xf64>)
    // CHECK-SAME:  cotangents([[cotangentRes]] : memref<?xf64>)
    // CHECK-SAME:  {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 0>}
    // CHECK-SAME:  : (memref<?x2xf64>) -> ()
    %grad = gradient.backprop @circuit(%arg0) cotangents(%arg1: tensor<?xf64>) {diffArgIndices = dense<0> : tensor<1xindex>, resultSegmentSizes = array<i32: 0, 1>}: (tensor<?x2xf64>) -> tensor<?x2xf64>
    return
}

// -----

func.func private @circuit(%arg0: tensor<10xf64>, %arg1: tensor<2xf64>)

// CHECK-LABEL: @backprop_multiple_tensors_in
func.func @backprop_multiple_tensors_in(%arg0: tensor<10xf64>, %arg1: tensor<2xf64>, %arg2: tensor<?xf64>) {

    // CHECK-DAG:   [[argSource0:%.+]] = bufferization.to_memref %arg0 : tensor<10xf64> to memref<10xf64>
    // CHECK-DAG:   [[argSource1:%.+]] = bufferization.to_memref %arg1 : tensor<2xf64> to memref<2xf64>
    // CHECK:   memref.alloc
    // CHECK:   memref.copy
    // CHECK:   [[argShadow1:%.+]] = memref.alloc() : memref<10xf64>
    // CHECK:   [[argShadow2:%.+]] = memref.alloc() : memref<2xf64>
    // CHECK:   [[dim:%.+]] = memref.dim
    // CHECK:   [[calleeRes:%.+]] = memref.alloc([[dim]]) : memref<?xf64>
    // CHECK:   gradient.backprop @circuit([[argSource0]], [[argSource1]])
    // CHECK-SAME: grad_out([[argShadow1]], [[argShadow2]] : memref<10xf64>, memref<2xf64>)
    // CHECK-SAME: callee_out([[calleeRes]] : memref<?xf64>)
    // CHECK-SAME: cotangents({{%.+}} : memref<?xf64>)
    // CHECK-SAME: {diffArgIndices = dense<[0, 1]> : tensor<2xindex>, resultSegmentSizes = array<i32: 0, 0>}
    // CHECK-SAME: : (memref<10xf64>, memref<2xf64>) -> ()
    %grad:2 = gradient.backprop @circuit(%arg0, %arg1) cotangents(%arg2: tensor<?xf64>) {diffArgIndices = dense<[0, 1]> : tensor<2xindex>, resultSegmentSizes = array<i32: 0, 2>}: (tensor<10xf64>, tensor<2xf64>) -> (tensor<10xf64>, tensor<2xf64>)
    return
}

// -----

// CHECK: func.func private @callback_fn_fwd(tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
func.func private @callback_fn_fwd(tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)

// CHECK: gradient.forward @callback_fn_fwd.fwd(%arg0: memref<2xf64>) -> (memref<f64>, memref<2xf64>)
// CHECK-SAME: attributes {argc = 1 : i64, implementation = @callback_fn_fwd, resc = 1 : i64, tape = 1 : i64}
// CHECK-SAME: {
gradient.forward @callback_fn_fwd.fwd(%arg0: tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>) attributes {argc = 1 : i64, implementation = @callback_fn_fwd, resc = 1 : i64, tape = 1 : i64} {

    // CHECK: [[in:%.+]] = bufferization.to_tensor %arg0 : memref<2xf64>
    // CHECK: [[callOut:%.+]]:2 = func.call @callback_fn_fwd([[in]]) : (tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
    // CHECK: [[res0:%.+]] = bufferization.to_memref [[callOut]]#0 : tensor<f64> to memref<f64>
    // CHECK: [[res1:%.+]] = bufferization.to_memref [[callOut]]#1 : tensor<2xf64> to memref<2xf64>
    // CHECK: gradient.return {empty = false} [[res0]], [[res1]] : memref<f64>, memref<2xf64>

    %0:2 = func.call @callback_fn_fwd(%arg0) : (tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
    gradient.return {empty = false} %0#0, %0#1 : tensor<f64>, tensor<2xf64>
}

// -----

// CHECK: func.func private @callback_fn_vjp(tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
func.func private @callback_fn_vjp(tensor<2xf64>, tensor<f64>) -> tensor<2xf64>

// CHECK: gradient.reverse @callback_fn_vjp.rev(%arg0: memref<f64>, %arg1: memref<2xf64>) -> memref<2xf64>
// CHECK-SAME: attributes {argc = 1 : i64, implementation = @callback_fn_vjp, resc = 1 : i64, tape = 1 : i64}
// CHECK-SAME: {
gradient.reverse @callback_fn_vjp.rev(%arg0: tensor<f64>, %arg1: tensor<2xf64>) -> tensor<2xf64> attributes {argc = 1 : i64, implementation = @callback_fn_vjp, resc = 1 : i64, tape = 1 : i64} {

    // CHECK: [[in1:%.+]] = bufferization.to_tensor %arg1 : memref<2xf64>
    // CHECK: [[in0:%.+]] = bufferization.to_tensor %arg0 : memref<f64>
    // CHECK: [[callOut:%.+]] = func.call @callback_fn_vjp([[in1]], [[in0]]) : (tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
    // CHECK: [[res:%.+]] = bufferization.to_memref [[callOut]] : tensor<2xf64> to memref<2xf64>
    // CHECK: gradient.return {empty = true} [[res]] : memref<2xf64>

    %0 = func.call @callback_fn_vjp(%arg1, %arg0) : (tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
    gradient.return {empty = true} %0 : tensor<2xf64>
}
