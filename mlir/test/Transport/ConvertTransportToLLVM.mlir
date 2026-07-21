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

// CHECK-DAG: llvm.func @__catalyst__transport__controller_create(!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @__catalyst__transport__connect(!llvm.ptr, !llvm.ptr, i16) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__exchange_keys(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__establish_channel(!llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__commit_work_item(!llvm.ptr, i32, i64, i64) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__data_slot(!llvm.ptr) -> !llvm.ptr
// CHECK-DAG: llvm.func @__catalyst__transport__kick(!llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__collect(!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK-DAG: llvm.func @__catalyst__transport__start(!llvm.ptr)
// CHECK-DAG: llvm.func @__catalyst__transport__stop(!llvm.ptr)
// CHECK-DAG: llvm.func @__catalyst__transport__destroy(!llvm.ptr)

// CHECK-LABEL: func.func @controller_roundtrip
func.func @controller_roundtrip() -> i64 {
  // CHECK: %[[S:.*]] = llvm.call @__catalyst__transport__controller_create
  %s = transport.controller_create {backend_lib = "libtransport_backend.so", config = "key=value"} -> !transport.session
  // CHECK: llvm.call @__catalyst__transport__connect(%[[S]]
  %c = transport.connect %s {peer = "127.0.0.1", oob_port = 18560 : i16} : (!transport.session) -> i32
  // CHECK: %[[PEER:.*]] = llvm.alloca
  // CHECK: llvm.call @__catalyst__transport__exchange_keys(%[[S]], %[[PEER]])
  %cs, %peer = transport.exchange_keys %s : !transport.session -> !transport.peer
  // CHECK: llvm.call @__catalyst__transport__establish_channel(%[[S]], {{.*}}, %[[PEER]])
  %e = transport.establish_channel %s, %peer {data_path = 0 : i32} : !transport.session, !transport.peer
  // CHECK: llvm.call @__catalyst__transport__commit_work_item(%[[S]]
  %w = transport.commit_work_item %s {work_item_idx = 0 : i32, in_bytes = 8 : i64, out_bytes = 8 : i64} : !transport.session
  // CHECK: llvm.call @__catalyst__transport__start(%[[S]])
  transport.start %s : !transport.session
  %payload = arith.constant 81985529216486895 : i64
  // CHECK: %[[SLOT:.*]] = llvm.call @__catalyst__transport__data_slot(%[[S]])
  // CHECK: llvm.store %{{.*}}, %[[SLOT]]
  // CHECK: llvm.call @__catalyst__transport__kick(%[[S]]
  %k = transport.kick %s, %payload {work_item_idx = 0 : i32} : !transport.session, i64
  // CHECK: llvm.call @__catalyst__transport__collect(%[[S]]
  // CHECK: %[[RESULT:.*]] = llvm.load
  %result = transport.collect %s {bytes = 8 : i64} : !transport.session -> i64
  // CHECK: llvm.call @__catalyst__transport__stop(%[[S]])
  transport.stop %s : !transport.session
  // CHECK: llvm.call @__catalyst__transport__destroy(%[[S]])
  transport.destroy %s : !transport.session
  return %result : i64
}
