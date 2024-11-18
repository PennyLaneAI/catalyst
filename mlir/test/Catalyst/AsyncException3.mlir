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

// RUN: quantum-opt --add-exception-handling=stop-after-step=3 --verify-diagnostics --split-input-file %s | FileCheck %s

module {

  // Check that nothing happens if attribute is not present
  llvm.func @async_execute_fn() -> !llvm.ptr
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr) -> i1
  llvm.func @abort() -> ()

  llvm.func @caller() {
    %0 = llvm.call @async_execute_fn() : () -> (!llvm.ptr)
    %1 = llvm.call @mlirAsyncRuntimeIsTokenError(%0) : (!llvm.ptr) -> i1
    %2 = llvm.mlir.constant(1 : i64) : i1
    %3 = llvm.xor %1, %2: i1
    llvm.cond_br %3, ^fail, ^success
    ^fail:
    // CHECK: llvm.call @abort
    llvm.call @abort() : () -> ()
    llvm.unreachable
    ^success:
    llvm.return
  }

}

// -----

module {

  // Check that we call __catalyst__host__rt__unrecoverable_error
  llvm.func internal @async_execute_fn() -> !llvm.ptr attributes { catalyst.preHandleError } {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }

  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr) -> i1
  llvm.func @abort() -> ()

  // CHECK: llvm.func @caller
  // CHECK-NOT:  llvm.call @abort
  // CHECK-NOT:  llvm.unreachable
  llvm.func @caller() {
    %0 = llvm.call @async_execute_fn() : () -> (!llvm.ptr)
    %1 = llvm.call @mlirAsyncRuntimeIsTokenError(%0) : (!llvm.ptr) -> i1
    %2 = llvm.mlir.constant(1 : i64) : i1
    %3 = llvm.xor %1, %2: i1
    llvm.cond_br %3, ^fail, ^success
    ^fail:
    // CHECK:      llvm.cond_br %{{.*}}, [[fail:\^.+]], [[success:\^.+]]
    // CHECK:      [[fail]]:
    // CHECK-NEXT: llvm.call @__catalyst__host__rt__unrecoverable_error
    // CHECK-NEXT: llvm.br [[success]]
    llvm.call @abort() : () -> ()
    llvm.unreachable
    ^success:
    llvm.return
  }
  // CHECK-NOT:  llvm.call @abort
  // CHECK-NOT:  llvm.unreachable
}

// -----

module {

  // Check that we add a source and a sink
  llvm.func internal @async_execute_fn() -> !llvm.ptr attributes { catalyst.preHandleError } {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }

  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr) -> i1
  llvm.func @abort() -> ()

  // CHECK: llvm.func @caller
  llvm.func @caller() {
    // CHECK: llvm.call @async_execute_fn
    // CHECK-SAME: catalyst.sourceOfRefCounts
    // CHECK: llvm.call @__catalyst__host__rt__unrecoverable_error
    // CHECK-SAME: catalyst.sink
    %0 = llvm.call @async_execute_fn() : () -> (!llvm.ptr)
    %1 = llvm.call @mlirAsyncRuntimeIsTokenError(%0) : (!llvm.ptr) -> i1
    %2 = llvm.mlir.constant(1 : i64) : i1
    %3 = llvm.xor %1, %2: i1
    llvm.cond_br %3, ^fail, ^success
    ^fail:
    llvm.call @abort() : () -> ()
    llvm.unreachable
    ^success:
    llvm.return
  }
}

// -----

// CHECK-LABEL: @remove_puts
module @remove_puts {

  // Check that puts is deleted
  llvm.func internal @async_execute_fn() -> !llvm.ptr attributes { catalyst.preHandleError } {
    %0 = llvm.mlir.zero : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }

  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr) -> i1
  llvm.func @abort() -> ()
  llvm.func @puts(!llvm.ptr) -> ()

  llvm.func @caller() {
    %0 = llvm.call @async_execute_fn() : () -> (!llvm.ptr)
    %1 = llvm.call @mlirAsyncRuntimeIsTokenError(%0) : (!llvm.ptr) -> i1
    %2 = llvm.mlir.constant(1 : i64) : i1
    %3 = llvm.xor %1, %2: i1
    llvm.cond_br %3, ^fail, ^success
    ^fail:
    // CHECK-NOT: llvm.call @puts
    %message = "test.op"() : () -> (!llvm.ptr)
    llvm.call @puts(%message) : (!llvm.ptr) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
    ^success:
    llvm.return
  }
}
