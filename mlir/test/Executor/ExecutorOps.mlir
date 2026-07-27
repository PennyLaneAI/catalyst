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
func.func @open() {
  // CHECK: %{{.*}} = executor.open("127.0.0.1:9000") : !executor.session
  %s = executor.open("127.0.0.1:9000") : !executor.session
  return
}

// -----

// CHECK-LABEL: func.func @send_binary
func.func @send_binary() {
  // CHECK: %[[S:.*]] = executor.open("127.0.0.1:9000") : !executor.session
  %s = executor.open("127.0.0.1:9000") : !executor.session
  // CHECK: executor.send_binary %[[S]]("/tmp/qnode_0.o") : !executor.session
  executor.send_binary %s("/tmp/qnode_0.o") : !executor.session
  return
}

// -----

// CHECK-LABEL: func.func @launch
func.func @launch(%arg0: memref<f64>) -> memref<f64> {
  // CHECK: %[[S:.*]] = executor.open("127.0.0.1:9000") : !executor.session
  %s = executor.open("127.0.0.1:9000") : !executor.session
  // CHECK: executor.launch %[[S]]("qnode_0") (%{{.*}}) : !executor.session, (memref<f64>) -> memref<f64>
  %0 = executor.launch %s("qnode_0") (%arg0) : !executor.session, (memref<f64>) -> memref<f64>
  return %0 : memref<f64>
}

// -----

// CHECK-LABEL: func.func @call
func.func @call(%arg0: memref<4xf64>, %arg1: memref<4xf64>) {
  // CHECK: %[[S:.*]] = executor.open("127.0.0.1:9000") : !executor.session
  %s = executor.open("127.0.0.1:9000") : !executor.session
  // CHECK: executor.call %[[S]]("foo") (%{{.*}}, %{{.*}}) {num_input_args = 1 : i32} : !executor.session, (memref<4xf64>, memref<4xf64>) -> ()
  executor.call %s("foo") (%arg0, %arg1)
      {num_input_args = 1 : i32} : !executor.session, (memref<4xf64>, memref<4xf64>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @close
func.func @close() {
  // CHECK: %[[S:.*]] = executor.open("127.0.0.1:9000") : !executor.session
  %s = executor.open("127.0.0.1:9000") : !executor.session
  // CHECK: executor.close %[[S]] : !executor.session
  executor.close %s : !executor.session
  return
}

// -----

// A single session handle chains open -> send -> launch -> close, and may be launched on more than
// once.
// CHECK-LABEL: func.func @session_lifecycle
func.func @session_lifecycle(%arg0: memref<f64>) -> memref<f64> {
  // CHECK: %[[S:.*]] = executor.open("127.0.0.1:9000") : !executor.session
  %s = executor.open("127.0.0.1:9000") : !executor.session
  executor.send_binary %s("/tmp/qnode_0.o") : !executor.session
  %0 = executor.launch %s("qnode_0") (%arg0) : !executor.session, (memref<f64>) -> memref<f64>
  %1 = executor.launch %s("qnode_0") (%0) : !executor.session, (memref<f64>) -> memref<f64>
  executor.close %s : !executor.session
  return %1 : memref<f64>
}
