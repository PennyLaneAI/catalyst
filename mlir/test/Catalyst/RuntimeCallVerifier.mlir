// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --split-input-file --verify-diagnostics

func.func @bad_input_count(%arg0: i32) {
  // expected-error@+1 {{expected 2 input operand(s) for the declared C parameters, but got 1}}
  %0 = catalyst.runtime_call fn("native_add") (%arg0) {
      c_params = ["i32", "i32"],
      c_result = "i32"
  } : (i32) -> (i32)
  return
}

// -----

func.func @tensor_scalar(%arg0: tensor<i32>, %arg1: i32) {
  // expected-error@+1 {{scalar/ptr parameter must remain an SSA scalar}}
  %0 = catalyst.runtime_call fn("native_add") (%arg0, %arg1) {
      c_params = ["i32", "i32"],
      c_result = "i32"
  } : (tensor<i32>, i32) -> (i32)
  return
}

// -----

func.func @missing_string(%arg0: i64) {
  // expected-error@+1 {{expected 1 compile-time string(s), but got 0}}
  %0 = catalyst.runtime_call fn("native_strlen") (%arg0) {
      c_params = ["ptr", "str"],
      c_result = "u64"
  } : (i64) -> (i64)
  return
}

// -----

func.func @dispatched_buf(%arg0: tensor<4xi8>, %arg1: i64) {
  // expected-error@+1 {{a buf names memory in this process and cannot be dispatched to 'board:9000'}}
  %0 = catalyst.runtime_call fn("native_sum") (%arg0, %arg1) {
      c_params = ["buf", "u64"],
      c_result = "i32",
      dispatch = "board:9000"
  } : (tensor<4xi8>, i64) -> (i32)
  return
}

// -----

func.func @dispatched_after_bufferization(%arg0: i64, %out: memref<4xi8>) {
  // expected-error@+1 {{a dispatched call must be lowered before bufferization}}
  %0 = catalyst.runtime_call fn("native_fill") (%arg0) in(%out : memref<4xi8>) {
      c_params = ["out", "u64"],
      c_result = "i32",
      dispatch = "board:9000"
  } : (i64) -> (i32)
  return
}

// -----

func.func @dest_buffer_without_out_param(%arg0: i64, %out: memref<4xi8>) {
  // expected-error@+1 {{expected 0 out tensor result(s) or destination buffer(s), but got 1}}
  %0 = catalyst.runtime_call fn("native_noop") (%arg0) in(%out : memref<4xi8>) {
      c_params = ["u64"],
      c_result = "i32"
  } : (i64) -> (i32)
  return
}

// -----

func.func @void_with_scalar_result() {
  // expected-error@+1 {{expected 0 scalar result(s) for C result type 'void', but got 1}}
  %0 = catalyst.runtime_call fn("native_noop") () {
      c_params = [],
      c_result = "void"
  } -> (i32)
  return
}

// -----

func.func @missing_scalar_result() {
  // expected-error@+1 {{expected 1 scalar result(s) for C result type 'i32', but got 0}}
  catalyst.runtime_call fn("native_noop") () {
      c_params = [],
      c_result = "i32"
  }
  return
}

// -----

func.func @two_scalar_results() {
  // A C call has at most one return value.
  // expected-error@+1 {{expected 1 scalar result(s) for C result type 'i32', but got 2}}
  %0:2 = catalyst.runtime_call fn("native_noop") () {c_params = [], c_result = "i32"} -> (i32, i32)
  return
}
