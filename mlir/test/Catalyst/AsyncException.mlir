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

// RUN: quantum-opt --detect-qnode --verify-diagnostics --split-input-file %s | FileCheck %s

// Check to make sure that calls which are not annotated are left alone.

module {
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  // CHECK-LABEL: caller
  llvm.func @caller() {
    // CHECK-NOT: llvm.invoke
    llvm.call @callee() : () -> ()
    llvm.return
  }
}

// -----

// Check to make sure that personality and abort were added

module {
  // CHECK: llvm.func @__gxx_personality_v0
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  // CHECK-LABEL: @caller
  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return
  }
}

// -----

// Check to make sure that the caller was annotated with personality

module {
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  // CHECK-LABEL: @caller
  // CHECK-SAME: personality = @__gxx_personality_v0
  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return
  }
}

// -----

// Check to make sure that the the `llvm.call` op has been transformed to llvm.invoke
module {
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  // CHECK-LABEL: caller
  llvm.func @caller() {
    // CHECK: llvm.invoke
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return
  }
}

// -----

// Check to make sure that the token gets set to error
module {
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  // CHECK-LABEL: caller
  llvm.func @caller() {
    %0 = llvm.call @mlirAsyncRuntimeCreateToken() : () -> !llvm.ptr
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    // CHECK: llvm.call @mlirAsyncRuntimeSetTokenError
    llvm.return
  }
}

// -----

// Check to make sure that the value gets set to error
module {
  llvm.func @mlirAsyncRuntimeCreateValue() -> !llvm.ptr
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  // CHECK-LABEL: caller
  llvm.func @caller() {
    %0 = llvm.call @mlirAsyncRuntimeCreateValue() : () -> !llvm.ptr
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    // CHECK: llvm.call @mlirAsyncRuntimeSetValueError
    llvm.return
  }
}

// -----

// Check to make sure that the we unconditionally jump from failure to success
module {
  // CHECK-LABEL: caller
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    // CHECK: ^bb1
    llvm.return
  }
}


// -----

// Check to make sure that the caller of async region gets annotated
module {
  // CHECK-LABEL: caller
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr
  llvm.func @mlirAsyncRuntimeCreateValue() -> !llvm.ptr
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr) -> i1
  llvm.func @abort()

  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  llvm.func @async_region() -> !llvm.struct<(ptr, ptr)> {
    %0 = llvm.call @mlirAsyncRuntimeCreateToken() : () -> !llvm.ptr
    %1 = llvm.call @mlirAsyncRuntimeCreateValue() : () -> !llvm.ptr
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(ptr, ptr)>
    %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(ptr, ptr)>
    llvm.return %4 : !llvm.struct<(ptr, ptr)>
  }

  llvm.func @caller() {
     %c1 = llvm.mlir.constant(1 : i64) : i1
     %0 = llvm.call @async_region() : () -> !llvm.struct<(ptr, ptr)>
     %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr)> 
     %2 = llvm.call @mlirAsyncRuntimeIsTokenError(%1) : (!llvm.ptr) -> i1
     %3 = llvm.xor %2, %c1 : i1
     llvm.cond_br %3, ^bbgood, ^bbbad
  ^bbgood:
     llvm.return
  ^bbbad:
     // CHECK: llvm.call @__catalyst__host__rt__unrecoverable_error
     llvm.call @abort() : () -> ()
     llvm.unreachable
  }
}

// -----

// Check to make sure that the caller of async region gets annotated
module {
  // CHECK-LABEL: caller
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr
  llvm.func @mlirAsyncRuntimeCreateValue() -> !llvm.ptr
  llvm.func @mlirAsyncRuntimeIsTokenError(!llvm.ptr) -> i1
  llvm.func @abort()
  llvm.func @source() -> !llvm.struct<(ptr, ptr)>
  llvm.func @sink(!llvm.ptr)

  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  llvm.func @async_region() -> !llvm.struct<(ptr, ptr)> {
    %0 = llvm.call @mlirAsyncRuntimeCreateToken() : () -> !llvm.ptr
    %1 = llvm.call @mlirAsyncRuntimeCreateValue() : () -> !llvm.ptr
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr)>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(ptr, ptr)>
    %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(ptr, ptr)>
    llvm.return %4 : !llvm.struct<(ptr, ptr)>
  }

  llvm.func @caller() {
     %c1 = llvm.mlir.constant(1 : i64) : i1
     %refcounted = llvm.call @source() { catalyst.sourceOfRefCounts } : () -> !llvm.struct<(ptr, ptr)>
     %token = llvm.extractvalue %refcounted[0] : !llvm.struct<(ptr, ptr)> 
     %0 = llvm.call @async_region() : () -> !llvm.struct<(ptr, ptr)>
     %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr)> 
     %2 = llvm.call @mlirAsyncRuntimeIsTokenError(%1) : (!llvm.ptr) -> i1
     %3 = llvm.xor %2, %c1 : i1
     llvm.cond_br %3, ^bbgood, ^bbbad
  ^finally:
     llvm.call @sink(%1) : (!llvm.ptr) -> ()
     llvm.call @sink(%token) : (!llvm.ptr) -> ()
     llvm.return
  ^bbgood:
     llvm.br ^finally
  ^bbbad:
     // CHECK: llvm.call @__catalyst__host__rt__unrecoverable_error
     llvm.call @abort() : () -> ()
     llvm.br ^finally
  }
}
