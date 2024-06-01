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

// RUN: quantum-opt --catalyst-bufferize --split-input-file %s | FileCheck %s

//////////////////////
// Catalyst PrintOp //
//////////////////////

func.func @dbprint_val(%arg0: tensor<?xf64>) {

    // CHECK: "catalyst.print"(%0) : (memref<?xf64>) -> ()
    "catalyst.print"(%arg0) : (tensor<?xf64>) -> ()

    return
}

// -----

func.func @dbprint_memref(%arg0: tensor<?xf64>) {

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
    // CHECK: [[memrefArg:%.+]] = bufferization.to_memref %arg0 : memref<3x3xf64>
    // CHECK: [[alloc:%.+]] = bufferization.alloc_tensor() {{.*}}: tensor<3x3xf64>
    // CHECK: [[allocmemref:%.+]]  = bufferization.to_memref [[alloc]] : memref<3x3xf64>
    // CHECK: catalyst.custom_call fn("lapack_dgesdd") ([[memrefArg]], [[allocmemref]]) {number_original_arg = array<i32: 1>} : (memref<3x3xf64>, memref<3x3xf64>) -> ()
    // CHECK: [[res:%.+]] = bufferization.to_tensor [[allocmemref]] : memref<3x3xf64>
    // CHECK: return [[res]] : tensor<3x3xf64>
    %0 = catalyst.custom_call fn("lapack_dgesdd") (%arg0) : (tensor<3x3xf64>) -> (tensor<3x3xf64>)

    return %0 : tensor<3x3xf64>
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
    // CHECK-DAG: [[memref0:%.+]] = bufferization.to_memref [[arg0]]
    // CHECK-DAG: [[tensor1:%.+]] = bufferization.alloc_tensor
    // CHECK:     [[memref1:%.+]] = bufferization.to_memref [[tensor1]]
    // CHECK:     catalyst.callback_call @callback_1([[memref0]], [[memref1]])
    %1 = catalyst.callback_call @callback_1(%arg0) : (tensor<f64>) -> (tensor<f64>)
    // CHECK:     [[retval:%.+]] = bufferization.to_tensor [[memref1]]
    // CHECK:     return [[retval]]
    return %1 : tensor<f64>
  }
}
