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

// RUN: quantum-opt --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     one-shot-bufferize{unknown-type-conversion=identity-layout-map} \
// RUN:   )" %s | FileCheck %s

//////////////////////
// Catalyst PrintOp //
//////////////////////

func.func @dbprint_val(%arg0: tensor<?xf64>) {

    // CHECK: %0 = bufferization.to_buffer %arg0
    // CHECK: "catalyst.print"(%0) : (memref<?xf64>) -> ()
    "catalyst.print"(%arg0) : (tensor<?xf64>) -> ()

    return
}

// -----

func.func @dbprint_memref(%arg0: tensor<?xf64>) {

    // CHECK: %0 = bufferization.to_buffer %arg0
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

func.func public @symbolic_array_error() attributes {llvm.emit_c_interface} {
    // expected-error @below {{catalyst::symbolic_array is a placeholder op}}
    // expected-error @below {{failed to bufferize op}}
    %0 = catalyst.symbolic_array : tensor<i64>
    return
}

// -----

func.func @custom_call(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    // CHECK: [[sourceAlloc:%.+]] = bufferization.to_buffer %arg0
    // CHECK: [[destAlloc:%.+]] = memref.alloc() {{.*}}: memref<3x3xf64>
    // CHECK: catalyst.custom_call fn("lapack_dgesdd") ([[sourceAlloc]], [[destAlloc]]) {number_original_arg = 1 : i32} :
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
    // CHECK: [[sourceAlloc:%.+]] = bufferization.to_buffer %arg0
    // CHECK: [[subview:%.+]] = memref.subview [[sourceAlloc]]
    // CHECK-SAME: memref<2x3xf64> to memref<2x2xf64, strided<[3, 1]>>
    // CHECK: [[copyAlloc:%.+]] = memref.alloc() : memref<2x2xf64>
    // CHECK: memref.copy [[subview]], [[copyAlloc]]
    // CHECK-SAME: memref<2x2xf64, strided<[3, 1]>> to memref<2x2xf64>
    // CHECK: [[destAlloc:%.+]] = memref.alloc() {{.*}}: memref<2x2xf64>
    // CHECK: catalyst.custom_call fn("lapack_dgesdd") ([[copyAlloc]], [[destAlloc]]) {number_original_arg = 1 : i32} :
    // CHECK-SAME: (memref<2x2xf64>, memref<2x2xf64>) -> ()
    // CHECK: [[res:%.+]] = bufferization.to_tensor [[destAlloc]] : memref<2x2xf64>
    // CHECK: return [[res]] : tensor<2x2xf64>
    %extract = tensor.extract_slice %arg0[0, 0] [2, 2] [1, 1] : tensor<2x3xf64> to tensor<2x2xf64>
    %0 = catalyst.custom_call fn("lapack_dgesdd") (%extract) : (tensor<2x2xf64>) -> (tensor<2x2xf64>)

    return %0 : tensor<2x2xf64>
}

// -----

// `backend_config` attribute survives bufferization.
// CHECK-LABEL: func.func @custom_call_backend_config
// CHECK: catalyst.custom_call fn("lapack_dgesdd") (%{{.*}}, %{{.*}}) {backend_config = {foo = "bar"}, number_original_arg = 1 : i32} :
// CHECK-SAME: (memref<3x3xf64>, memref<3x3xf64>) -> ()
func.func @custom_call_backend_config(%arg0: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %0 = catalyst.custom_call fn("lapack_dgesdd") (%arg0) {backend_config = {foo = "bar"}} : (tensor<3x3xf64>) -> (tensor<3x3xf64>)
    return %0 : tensor<3x3xf64>
}

// -----

// CHECK-LABEL: func.func @runtime_call_buf
// CHECK: [[source:%.+]] = bufferization.to_buffer %arg0
// CHECK: [[status:%.+]] = catalyst.runtime_call fn("native_sum") ([[source]], %arg1)
// CHECK-SAME: {c_params = ["buf", "u64"], c_result = "i32"}
// CHECK-SAME: (memref<4xi8>, i64) -> (i32)
// CHECK: return [[status]]
func.func @runtime_call_buf(%arg0: tensor<4xi8>, %arg1: i64) -> i32 {
    %0 = catalyst.runtime_call fn("native_sum") (%arg0, %arg1) {
        c_params = ["buf", "u64"],
        c_result = "i32"
    } : (tensor<4xi8>, i64) -> (i32)
    return %0 : i32
}

// -----

// CHECK-LABEL: func.func @runtime_call_out
// CHECK: [[dest:%.+]] = memref.alloc() {{.*}}: memref<4xi8>
// CHECK: [[status:%.+]] = catalyst.runtime_call fn("native_fill") (%arg0, %arg1) in([[dest]] : memref<4xi8>)
// CHECK-SAME: {c_params = ["out", "u64", "u8"], c_result = "i32"}
// CHECK-SAME: (i64, i8) -> (i32)
// CHECK: [[tensor:%.+]] = bufferization.to_tensor [[dest]]
// CHECK: return [[status]], [[tensor]]
func.func @runtime_call_out(%arg0: i64, %arg1: i8) -> (i32, tensor<4xi8>) {
    %status, %out = catalyst.runtime_call fn("native_fill") (%arg0, %arg1) {
        c_params = ["out", "u64", "u8"],
        c_result = "i32"
    } : (i64, i8) -> (i32) outs(tensor<4xi8>)
    return %status, %out : i32, tensor<4xi8>
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
    // CHECK-DAG: [[memref0:%.+]] = bufferization.to_buffer [[arg0]] : tensor<f64> to memref<f64>
    // CHECK-DAG: [[resAlloc:%.+]] = memref.alloc() {{.*}}: memref<f64>
    // CHECK:     catalyst.callback_call @callback_1([[memref0]], [[resAlloc]]) : (memref<f64>, memref<f64>) -> ()
    %1 = catalyst.callback_call @callback_1(%arg0) : (tensor<f64>) -> (tensor<f64>)
    // CHECK:     [[retval:%.+]] = bufferization.to_tensor [[resAlloc]]
    // CHECK:     return [[retval]]
    return %1 : tensor<f64>
  }
}
