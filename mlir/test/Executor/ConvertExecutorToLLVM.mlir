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

// CHECK-DAG: llvm.mlir.global internal constant @executor_addr_127_0_0_1_9000("127.0.0.1:9000\00")
// CHECK-DAG: llvm.func @__catalyst__executor__open(!llvm.ptr) -> i64
// CHECK-LABEL: func.func @open
func.func @open() {
  // CHECK-NOT: llvm.alloca
  // CHECK: llvm.mlir.addressof @executor_addr_127_0_0_1_9000
  // CHECK: llvm.getelementptr
  // CHECK: %{{.*}} = llvm.call @__catalyst__executor__open(%{{.*}}) : (!llvm.ptr) -> i64
  %s = executor.open("127.0.0.1:9000") : !executor.session
  return
}

// -----

// CHECK-DAG: llvm.mlir.global internal constant @executor_path_qnode_0("/tmp/qnode_0.o\00")
// CHECK-DAG: llvm.func @__catalyst__executor__send_binary(i64, !llvm.ptr, i32) -> i64
// CHECK-LABEL: func.func @send_binary
func.func @send_binary() {
  // CHECK: %[[S:.*]] = llvm.call @__catalyst__executor__open(%{{.*}}) : (!llvm.ptr) -> i64
  // CHECK-NOT: llvm.alloca
  // CHECK: llvm.call @__catalyst__executor__send_binary(%[[S]], %{{.*}}, %{{.*}}) : (i64, !llvm.ptr, i32) -> i64
  %s = executor.open("127.0.0.1:9000") : !executor.session
  executor.send_binary %s("/tmp/qnode_0.o") : !executor.session
  return
}

// -----

// CHECK-DAG: llvm.mlir.global internal constant @executor_sym_qnode_0("_catalyst_pyface_qnode_0\00")
// CHECK-DAG: llvm.mlir.global internal constant @executor_in_ranks_qnode_0(dense<0> : tensor<1xi64>)
// CHECK-DAG: llvm.mlir.global internal constant @executor_in_sizes_qnode_0(dense<8> : tensor<1xi64>)
// CHECK-DAG: llvm.mlir.global internal constant @executor_out_ranks_qnode_0(dense<0> : tensor<1xi64>)
// CHECK-DAG: llvm.mlir.global internal constant @executor_out_sizes_qnode_0(dense<8> : tensor<1xi64>)
// CHECK-DAG: llvm.func @__catalyst__executor__launch(i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
// CHECK-LABEL: func.func @launch
func.func @launch(%arg0: memref<f64>) -> memref<f64> {
  // Four stack slots: the input/output descriptors, and the two pointer arrays handed to the
  // runtime. They are hoisted to the function entry, ahead of the open call.
  // CHECK-COUNT-4: llvm.alloca
  // CHECK-NOT: llvm.alloca

  // open returns the i64 session handle threaded into the launch call
  // CHECK: %[[S:.*]] = llvm.call @__catalyst__executor__open(%{{.*}}) : (!llvm.ptr) -> i64

  // Input descriptor is stored, then packed into the input pointer-array
  // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.array<1 x ptr>
  // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.array<1 x ptr>, !llvm.ptr

  // Output pointer-array is packed the same way
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.array<1 x ptr>
  // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.array<1 x ptr>, !llvm.ptr

  // A single dispatch call threading the session handle plus the nine marshalled operands
  // CHECK: llvm.call @__catalyst__executor__launch(%[[S]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
  // Result read back from the output descriptor slot.
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64)>

  // Launch marshals via stores only
  // CHECK-NOT: llvm.intr.memcpy
  %s = executor.open("127.0.0.1:9000") : !executor.session
  %0 = executor.launch %s("qnode_0") (%arg0) : !executor.session, (memref<f64>) -> memref<f64>
  return %0 : memref<f64>
}

// -----

// CHECK-DAG: llvm.mlir.global internal constant @executor_lib_sym_foo("foo\00")
// CHECK-DAG: llvm.func @__catalyst__executor__call_wrapper(i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__catalyst__executor__free_result(!llvm.ptr)
// CHECK-LABEL: func.func @call
func.func @call(%arg0: memref<4xf64>, %arg1: memref<4xf64>) {
  // Three stack slots: the packed args buffer, plus the out-buffer pointer and out-size.
  // They are hoisted to the function entry, ahead of the open call.
  // CHECK-COUNT-3: llvm.alloca
  // CHECK-NOT: llvm.alloca

  // open returns the i64 session handle threaded into the call_wrapper call
  // CHECK: %[[S:.*]] = llvm.call @__catalyst__executor__open(%{{.*}}) : (!llvm.ptr) -> i64

  // Input operand bytes copied into the args buffer
  // CHECK: "llvm.intr.memcpy"

  // Dispatch, then read the out-buffer pointer the callee filled in
  // CHECK: llvm.call @__catalyst__executor__call_wrapper(%[[S]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i64, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> i32
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr

  // Results copied back into the output memref buffer
  // CHECK: "llvm.intr.memcpy"

  // The runtime-owned result buffer is freed
  // CHECK: llvm.call @__catalyst__executor__free_result(%{{.*}}) : (!llvm.ptr) -> ()

  // CHECK-NOT: "llvm.intr.memcpy"
  %s = executor.open("127.0.0.1:9000") : !executor.session
  executor.call %s("foo") (%arg0, %arg1)
      {num_input_args = 1 : i32} : !executor.session, (memref<4xf64>, memref<4xf64>) -> ()
  return
}

// -----

// close consumes the i64 session handle.
// CHECK-DAG: llvm.func @__catalyst__executor__close(i64) -> i64
// CHECK-LABEL: func.func @close
func.func @close() {
  // CHECK: %[[S:.*]] = llvm.call @__catalyst__executor__open(%{{.*}}) : (!llvm.ptr) -> i64
  // CHECK-NOT: llvm.alloca
  // CHECK: llvm.call @__catalyst__executor__close(%[[S]]) : (i64) -> i64
  %s = executor.open("127.0.0.1:9000") : !executor.session
  executor.close %s : !executor.session
  return
}
