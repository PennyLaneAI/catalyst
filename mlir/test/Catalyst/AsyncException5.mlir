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

// RUN: quantum-opt --add-exception-handling=stop-after-step=5 --verify-diagnostics --split-input-file %s | FileCheck %s

module {

  // Check that we remove sources and transform back branches marked as unreachable to unreachables.
  llvm.func @async_execute_fn() -> !llvm.struct<(ptr, ptr)>
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr) -> i1
  llvm.func @mlirAsyncRuntimeAwaitToken(!llvm.ptr) -> ()
  llvm.func @abort() -> ()
  llvm.func @raiseException() -> ()

  llvm.func @caller() {
    // CHECK-NOT: catalyst.sourceOfRefCounts
    // CHECK: llvm.call @async_execute_fn
    // CHECK-NOT: catalyst.sourceOfRefCounts
    %0 = llvm.call @async_execute_fn() { catalyst.sourceOfRefCounts }  : () -> (!llvm.struct<(ptr, ptr)>)
    %token = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr)>
    %1 = llvm.call @mlirAsyncRuntimeIsTokenError(%token) : (!llvm.ptr) -> i1
    %2 = llvm.mlir.constant(1 : i64) : i1
    %3 = llvm.xor %1, %2: i1
    llvm.cond_br %3, ^fail, ^success
    ^fail:
    llvm.call @raiseException() { catalyst.sink } : () -> ()
    // CHECK-NOT: catalyst.unreachable
    // CHECK: llvm.unreachable
    // CHECK-NOT: catalyst.unreachable
    llvm.br ^success { catalyst.unreachable }
    ^success:
    llvm.call @mlirAsyncRuntimeAwaitToken(%token) : (!llvm.ptr) -> ()
    llvm.return
  }

}
