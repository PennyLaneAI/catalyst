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
  // CHECK: llvm.func @abort
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

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

  // CHECK: llvm.func @caller
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
  llvm.func @mlirAsyncRuntimeCreateValue() -> !llvm.ptr
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  llvm.func @async_region() -> !llvm.ptr {
    %0 = llvm.call @mlirAsyncRuntimeCreateValue() : () -> !llvm.ptr
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return %0 : !llvm.ptr
  }

  // CHECK: catalyst.preHandleError
  llvm.func @caller() -> !llvm.ptr {
     %0 = llvm.call @async_region() : () -> !llvm.ptr
     llvm.return %0 : !llvm.ptr
  }

}
