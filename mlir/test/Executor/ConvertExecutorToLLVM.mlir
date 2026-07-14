// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --convert-executor-to-llvm --split-input-file | FileCheck %s

// CHECK: llvm.func @__catalyst__remote__open(!llvm.ptr) -> i64
// CHECK-LABEL: func.func @open
func.func @open() {
  // CHECK: llvm.call @__catalyst__remote__open
  executor.open("127.0.0.1:9000")
  return
}

// -----

// CHECK: llvm.func @__catalyst__remote__send_binary(!llvm.ptr, !llvm.ptr, i32) -> i64
// CHECK-LABEL: func.func @send_binary
func.func @send_binary() {
  // CHECK: llvm.call @__catalyst__remote__send_binary
  executor.send_binary("127.0.0.1:9000", "/tmp/qnode_0.o")
  return
}

// -----

// CHECK: llvm.func @__catalyst__remote__launch
// CHECK-LABEL: func.func @launch
func.func @launch(%arg0: memref<f64>) -> memref<f64> {
  // CHECK: llvm.call @__catalyst__remote__launch
  %0 = executor.launch("qnode_0", "127.0.0.1:9000") (%arg0) : (memref<f64>) -> memref<f64>
  return %0 : memref<f64>
}

// -----

// CHECK-DAG: llvm.func @__catalyst__remote__call_wrapper
// CHECK-DAG: llvm.func @__catalyst__remote__free_result
// CHECK-LABEL: func.func @call
func.func @call(%arg0: memref<4xf64>, %arg1: memref<4xf64>) {
  // CHECK: llvm.call @__catalyst__remote__call_wrapper
  // CHECK: llvm.call @__catalyst__remote__free_result
  executor.call("foo", "127.0.0.1:9000") (%arg0, %arg1)
      {num_input_args = 1 : i32} : (memref<4xf64>, memref<4xf64>) -> ()
  return
}
