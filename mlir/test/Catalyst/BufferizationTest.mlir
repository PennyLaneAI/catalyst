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

// RUN: quantum-opt --one-shot-bufferize --split-input-file %s | FileCheck %s

//////////////////////
// Catalyst PrintOp //
//////////////////////

func.func @dbprint_val(%arg0: tensor<?xf64>) {

    // CHECK: %0 = bufferization.to_memref %arg0
    // CHECK: "catalyst.print"(%0) : (memref<?xf64, strided<[?], offset: ?>>) -> ()
    "catalyst.print"(%arg0) : (tensor<?xf64>) -> ()

    return
}

// -----

func.func @dbprint_memref(%arg0: tensor<?xf64>) {

    // CHECK: %0 = bufferization.to_memref %arg0
    // CHECK: "catalyst.print"(%0) <{print_descriptor}> : (memref<?xf64, strided<[?], offset: ?>>) -> ()
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

func.func @custom_call_no_write(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    // CHECK: [[sourceAlloc:%.+]] = bufferization.to_memref %arg0 : memref<3x3xf64, strided<[?, ?], offset: ?>>
    // CHECK: [[destAlloc:%.+]] = memref.alloc() {{.*}}: memref<3x3xf64>
    // CHECK: catalyst.custom_call fn("lapack_dgesdd") ([[sourceAlloc]], [[destAlloc]]) {number_original_arg = array<i32: 1>} :
    // CHECK-SAME: (memref<3x3xf64, strided<[?, ?], offset: ?>>, memref<3x3xf64>) -> ()
    // CHECK: [[res:%.+]] = bufferization.to_tensor [[destAlloc]] : memref<3x3xf64>
    // CHECK: return [[res]] : tensor<3x3xf64>
    %0 = catalyst.custom_call fn("lapack_dgesdd") (%arg0) : (tensor<3x3xf64>) -> (tensor<3x3xf64>)

    return %0 : tensor<3x3xf64>
}

// -----

func.func @custom_call_with_write(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    // CHECK: [[memrefArg:%.+]] = bufferization.to_memref %arg0 : memref<3x3xf64, strided<[?, ?], offset: ?>>
    // CHECK: [[sourceAlloc:%.+]] = memref.alloc() {{.*}}: memref<3x3xf64>
    // CHECK: memref.copy [[memrefArg]], [[sourceAlloc]]
    // CHECK: [[destAlloc:%.+]] = memref.alloc() {{.*}}: memref<3x3xf64>
    // CHECK: catalyst.custom_call fn("write_to_arg") ([[sourceAlloc]], [[destAlloc]]) {number_original_arg = array<i32: 1>} :
    // CHECK-SAME: (memref<3x3xf64>, memref<3x3xf64>) -> ()
    // CHECK: [[res:%.+]] = bufferization.to_tensor [[destAlloc]] : memref<3x3xf64>
    // CHECK: return [[res]] : tensor<3x3xf64>
    %0 = catalyst.custom_call fn("write_to_arg") (%arg0) : (tensor<3x3xf64>) -> (tensor<3x3xf64>)

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
    // CHECK-DAG: [[memref0:%.+]] = bufferization.to_memref [[arg0]] : memref<f64, strided<[], offset: ?>>
    // CHECK-DAG: [[resAlloc:%.+]] = memref.alloc() {{.*}}: memref<f64>
    // CHECK:     catalyst.callback_call @callback_1([[memref0]], [[resAlloc]]) : (memref<f64, strided<[], offset: ?>>, memref<f64>) -> ()
    %1 = catalyst.callback_call @callback_1(%arg0) : (tensor<f64>) -> (tensor<f64>)
    // CHECK:     [[retval:%.+]] = bufferization.to_tensor [[resAlloc]]
    // CHECK:     return [[retval]]
    return %1 : tensor<f64>
  }
}
