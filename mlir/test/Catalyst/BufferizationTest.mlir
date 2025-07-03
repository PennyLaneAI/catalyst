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

// RUN: quantum-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     one-shot-bufferize{unknown-type-conversion=identity-layout-map} \
// RUN:   )" %s | FileCheck %s

//////////////////////
// Catalyst PrintOp //
//////////////////////

func.func @dbprint_val(%arg0: tensor<?xf64>) {

    // CHECK: %0 = bufferization.to_memref %arg0
    // CHECK: "catalyst.print"(%0) : (memref<?xf64>) -> ()
    "catalyst.print"(%arg0) : (tensor<?xf64>) -> ()

    return
}

// -----

func.func @dbprint_memref(%arg0: tensor<?xf64>) {

    // CHECK: %0 = bufferization.to_memref %arg0
    // CHECK: "catalyst.print"(%0) <{print_descriptor}> : (memref<?xf64>) -> ()
    "catalyst.print"(%arg0) {print_descriptor} : (tensor<?xf64>) -> ()

    return
}

// -----

func.func @dbprint_str() {

    // CHECK: "catalyst.print"() <{const_val = "Hello, Catalyst"}> : () -> ()
    "catalyst.print"() {const_val = "Hello, Catalyst"} : () -> ()

    return
}

// -----

func.func @custom_call(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    // CHECK: [[sourceAlloc:%.+]] = bufferization.to_memref %arg0
    // CHECK: [[destAlloc:%.+]] = memref.alloc() {{.*}}: memref<3x3xf64>
    // CHECK: catalyst.custom_call fn("lapack_dgesdd") ([[sourceAlloc]], [[destAlloc]]) {number_original_arg = array<i32: 1>} :
    // CHECK-SAME: (memref<3x3xf64>, memref<3x3xf64>) -> ()
    // CHECK: [[res:%.+]] = bufferization.to_tensor [[destAlloc]] : memref<3x3xf64>
    // CHECK: return [[res]] : tensor<3x3xf64>
    %0 = catalyst.custom_call fn("lapack_dgesdd") (%arg0) : (tensor<3x3xf64>) -> (tensor<3x3xf64>)

    return %0 : tensor<3x3xf64>
}

// -----

func.func @custom_call_copy(%arg0: tensor<2x3xf64>) -> tensor<2x2xf64> {
    // COM: when buffer has non-identity layout, e.g. with strides
    // COM: e.g. coming from tensor subviews
    // COM: a copy needs to be performed because the kernels only allow for contiguous arrays as inputs
    //
    // CHECK: [[sourceAlloc:%.+]] = bufferization.to_memref %arg0
    // CHECK: [[subview:%.+]] = memref.subview [[sourceAlloc]]
    // CHECK-SAME: memref<2x3xf64> to memref<2x2xf64, strided<[3, 1]>>
    // CHECK: [[copyAlloc:%.+]] = memref.alloc() : memref<2x2xf64>
    // CHECK: memref.copy [[subview]], [[copyAlloc]]
    // CHECK-SAME: memref<2x2xf64, strided<[3, 1]>> to memref<2x2xf64>
    // CHECK: [[destAlloc:%.+]] = memref.alloc() {{.*}}: memref<2x2xf64>
    // CHECK: catalyst.custom_call fn("lapack_dgesdd") ([[copyAlloc]], [[destAlloc]]) {number_original_arg = array<i32: 1>} :
    // CHECK-SAME: (memref<2x2xf64>, memref<2x2xf64>) -> ()
    // CHECK: [[res:%.+]] = bufferization.to_tensor [[destAlloc]] : memref<2x2xf64>
    // CHECK: return [[res]] : tensor<2x2xf64>
    %extract = tensor.extract_slice %arg0[0, 0] [2, 2] [1, 1] : tensor<2x3xf64> to tensor<2x2xf64>
    %0 = catalyst.custom_call fn("lapack_dgesdd") (%extract) : (tensor<2x2xf64>) -> (tensor<2x2xf64>)

    return %0 : tensor<2x2xf64>
}

// -----

// CHECK-LABEL: @test0
module @test0 {
  // CHECK: catalyst.callback @callback_1(memref<f64>, memref<f64>)
  catalyst.callback @callback_1(tensor<f64>) -> tensor<f64> attributes { argc = 1:i64, resc = 1 : i64, id = 1:i64}
}

// -----

// CHECK-LABEL: @test1
module @test1 {
  catalyst.callback @callback_1(tensor<f64>) -> tensor<f64> attributes { argc = 1:i64, resc = 1 : i64, id = 1:i64}

  // CHECK-LABEL: @foo(
  // CHECK-SAME: [[arg0:%.+]]: tensor<f64>)
  func.func private @foo(%arg0: tensor<f64>) -> tensor<f64> {
    // CHECK-DAG: [[memref0:%.+]] = bufferization.to_memref [[arg0]] : tensor<f64> to memref<f64>
    // CHECK-DAG: [[resAlloc:%.+]] = memref.alloc() {{.*}}: memref<f64>
    // CHECK:     catalyst.callback_call @callback_1([[memref0]], [[resAlloc]]) : (memref<f64>, memref<f64>) -> ()
    %1 = catalyst.callback_call @callback_1(%arg0) : (tensor<f64>) -> (tensor<f64>)
    // CHECK:     [[retval:%.+]] = bufferization.to_tensor [[resAlloc]]
    // CHECK:     return [[retval]]
    return %1 : tensor<f64>
  }
}
