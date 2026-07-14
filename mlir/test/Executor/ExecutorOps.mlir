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

// RUN: quantum-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @open
// CHECK:         executor.open("127.0.0.1:9000")
func.func @open() {
  executor.open("127.0.0.1:9000")
  return
}

// -----

// CHECK-LABEL: func.func @send_binary
// CHECK:         executor.send_binary("127.0.0.1:9000", "/tmp/qnode_0.o")
func.func @send_binary() {
  executor.send_binary("127.0.0.1:9000", "/tmp/qnode_0.o")
  return
}

// -----

// CHECK-LABEL: func.func @launch
// CHECK:         executor.launch("qnode_0", "127.0.0.1:9000") (%{{.*}}) :
// CHECK-SAME:    (memref<f64>) -> memref<f64>
func.func @launch(%arg0: memref<f64>) -> memref<f64> {
  %0 = executor.launch("qnode_0", "127.0.0.1:9000") (%arg0) : (memref<f64>) -> memref<f64>
  return %0 : memref<f64>
}

// -----

// CHECK-LABEL: func.func @call
// CHECK:         executor.call("foo", "127.0.0.1:9000")
// CHECK-SAME:    num_input_args = 1 : i32
// CHECK-SAME:    (memref<4xf64>, memref<4xf64>) -> ()
func.func @call(%arg0: memref<4xf64>, %arg1: memref<4xf64>) {
  executor.call("foo", "127.0.0.1:9000") (%arg0, %arg1)
      {num_input_args = 1 : i32} : (memref<4xf64>, memref<4xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @close
// CHECK:         executor.close("127.0.0.1:9000")
func.func @close() {
  executor.close("127.0.0.1:9000")
  return
}
