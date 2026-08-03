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

// RUN: quantum-opt %s --convert-transport-to-llvm --split-input-file | FileCheck %s

// CHECK-DAG: llvm.func @__catalyst__transport__create(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @__catalyst__transport__connect(!llvm.ptr, !llvm.ptr, i16) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__exchange_keys(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__establish_channel(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__commit_work_item(!llvm.ptr, i32, i64, i64) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__write_data_slot(!llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__kick(!llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__collect(!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__start(!llvm.ptr)
// CHECK-DAG: llvm.func @__catalyst__transport__stop(!llvm.ptr)
// CHECK-DAG: llvm.func @__catalyst__transport__destroy(!llvm.ptr)

// Controller: create (role in the result type) -> bring-up -> kick/collect over buffers.
// CHECK-LABEL: func.func @controller
func.func @controller(%syndrome: memref<?xi8>, %correction: memref<?xi8>) {
  // CHECK: %[[S:.*]] = llvm.call @__catalyst__transport__create({{.*}}) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %s = transport.create {backend_lib = "libbackend.so", config = "cfg"} -> !transport.session<controller>
  // CHECK: llvm.call @__catalyst__transport__connect(%[[S]]
  transport.connect %s {peer = "127.0.0.1", oob_port = 18560 : i16} : !transport.session<controller>
  // CHECK: llvm.call @__catalyst__transport__exchange_keys(%[[S]])
  transport.exchange_keys %s : !transport.session<controller>
  // CHECK: llvm.call @__catalyst__transport__establish_channel(%[[S]]
  transport.establish_channel %s "cpu_verbs" : !transport.session<controller>
  // CHECK: llvm.call @__catalyst__transport__commit_work_item(%[[S]]
  transport.commit_work_item %s {work_item_idx = 0 : i32, in_bytes = 8 : i64, out_bytes = 8 : i64} : !transport.session<controller>
  // CHECK: llvm.call @__catalyst__transport__start(%[[S]])
  transport.start %s : !transport.session<controller>
  // CHECK: llvm.call @__catalyst__transport__write_data_slot(%[[S]]
  // CHECK: llvm.call @__catalyst__transport__kick(%[[S]]
  transport.kick %s, %syndrome {work_item_idx = 0 : i32} : !transport.session<controller>, memref<?xi8>
  // CHECK: llvm.call @__catalyst__transport__collect(%[[S]]
  transport.collect %s, %correction : !transport.session<controller>, memref<?xi8>
  // CHECK: llvm.call @__catalyst__transport__stop(%[[S]])
  transport.stop %s : !transport.session<controller>
  // CHECK: llvm.call @__catalyst__transport__destroy(%[[S]])
  transport.destroy %s : !transport.session<controller>
  return
}

// -----

// Coprocessor: create + bind the coprocessor function symbol + async bring-up.
// CHECK-LABEL: func.func @coprocessor
func.func @coprocessor() {
  // CHECK: %[[C:.*]] = llvm.call @__catalyst__transport__create({{.*}}) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
  %c = transport.create {backend_lib = "libbackend.so", config = "cfg"} -> !transport.session<coprocessor>
  // CHECK: llvm.call @__catalyst__transport__connect_async(%[[C]]
  %t = transport.connect_async %c {peer = "127.0.0.1", oob_port = 18560 : i16} : !transport.session<coprocessor> -> !transport.token
  // CHECK: llvm.call @__catalyst__transport__barrier
  transport.barrier %t : !transport.token
  // An omitted convention lowers to 0 (per-message).
  // CHECK: llvm.call @__catalyst__transport__set_coprocessor_fn(%[[C]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
  transport.set_coprocessor_fn %c {symbol = "foo"} : !transport.session<coprocessor>
  // An explicit launch-once convention lowers to 1.
  // CHECK: llvm.call @__catalyst__transport__set_coprocessor_fn(%[[C]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
  transport.set_coprocessor_fn %c {symbol = "bar", convention = 1 : i32} : !transport.session<coprocessor>
  // CHECK: llvm.call @__catalyst__transport__destroy(%[[C]])
  transport.destroy %c : !transport.session<coprocessor>
  return
}

// -----

// get_session resolves a session by (role from result type, key) via the runtime registry.
// CHECK-DAG: llvm.func @__catalyst__transport__get_session(i32, !llvm.ptr) -> !llvm.ptr
// CHECK-LABEL: func.func @resolve
func.func @resolve(%syndrome: memref<?xi8>, %correction: memref<?xi8>) {
  // CHECK: %[[R:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[S:.*]] = llvm.call @__catalyst__transport__get_session(%[[R]], {{.*}}) : (i32, !llvm.ptr) -> !llvm.ptr
  %s = transport.get_session {key = "cop0"} : !transport.session<controller>
  // CHECK: llvm.call @__catalyst__transport__kick(%[[S]]
  transport.kick %s, %syndrome {work_item_idx = 0 : i32} : !transport.session<controller>, memref<?xi8>
  // CHECK: llvm.call @__catalyst__transport__collect(%[[S]]
  transport.collect %s, %correction : !transport.session<controller>, memref<?xi8>
  return
}
