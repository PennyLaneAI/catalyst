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
  // CHECK: executor.launch %[[S]]("qnode_0", "/tmp/qnode.o") (%{{.*}}) : !executor.session, (memref<f64>) -> memref<f64>
  %0 = executor.launch %s("qnode_0", "/tmp/qnode.o") (%arg0) : !executor.session, (memref<f64>) -> memref<f64>
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

// A `()->()` entry dispatched fire-and-forget yields a token, joined by executor.await.
// CHECK-LABEL: func.func @launch_async
func.func @launch_async() {
  // CHECK: %[[S:.*]] = executor.open("127.0.0.1:9000") : !executor.session
  %s = executor.open("127.0.0.1:9000") : !executor.session
  // CHECK: %[[T:.*]] = executor.launch_async %[[S]]("serve", "/tmp/coproc.o") : !executor.session -> !executor.token
  %t = executor.launch_async %s("serve", "/tmp/coproc.o") : !executor.session -> !executor.token
  // CHECK: executor.await %[[T]] : !executor.token
  executor.await %t : !executor.token
  executor.close %s : !executor.session
  return
}

// -----

// Two async launches on one session yield two distinct tokens, each awaited independently.
// CHECK-LABEL: func.func @launch_async_two
func.func @launch_async_two() {
  %s = executor.open("127.0.0.1:9000") : !executor.session
  // CHECK: %[[T0:.*]] = executor.launch_async %{{.*}}("serve_0"
  %t0 = executor.launch_async %s("serve_0", "/tmp/coproc.o") : !executor.session -> !executor.token
  // CHECK: %[[T1:.*]] = executor.launch_async %{{.*}}("serve_1"
  %t1 = executor.launch_async %s("serve_1", "/tmp/coproc.o") : !executor.session -> !executor.token
  // CHECK: executor.await %[[T0]] : !executor.token
  executor.await %t0 : !executor.token
  // CHECK: executor.await %[[T1]] : !executor.token
  executor.await %t1 : !executor.token
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
  %0 = executor.launch %s("qnode_0", "/tmp/qnode_0.o") (%arg0) : !executor.session, (memref<f64>) -> memref<f64>
  %1 = executor.launch %s("qnode_0", "/tmp/qnode_0.o") (%0) : !executor.session, (memref<f64>) -> memref<f64>
  executor.close %s : !executor.session
  return %1 : memref<f64>
}
