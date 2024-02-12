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

// RUN: quantum-opt --add-exception-handling=stop-after-step=2 --verify-diagnostics --split-input-file %s | FileCheck %s


// Check that personality does not get added.


module {

  // CHECK-NOT: @__gxx_personality_v0

  llvm.func @callee()

  llvm.func @caller() {
    llvm.call @callee() : () -> ()
    llvm.return
  }
}

// -----

module {

  // Check that personality does get added
  // CHECK: @__gxx_personality_v0

  llvm.func @callee()

  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return
  }
}

// -----

module {


  llvm.func @callee()

  // Check that caller has been annotated with personality
  // CHECK: llvm.func @caller
  // CHECK-SAME: personality = @__gxx_personality_v0
  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return
  }
}

// -----

// Check for three basic blocks in caller.

module {


  llvm.func @callee()

  // CHECK: llvm.func @caller
  llvm.func @caller() {
    // The first one is the entry block.
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    // CHECK: ^bb1:
    // CHECK: ^bb2:
    llvm.return
  }
}

// -----

// Check that the first instruction to ^bb1 is a landingpad

module {

  llvm.func @callee()

  // CHECK: llvm.func @caller
  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    // CHECK: ^bb1:
    // CHECK-NEXT: llvm.landingpad
    llvm.return
  }
}

// -----


// Check that the success is the return.

module {

  llvm.func @callee()

  // CHECK: llvm.func @caller
  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    // CHECK: ^bb2:
    // CHECK-NEXT: llvm.return
    llvm.return
  }
}

// -----

// Check that the llvm.call has been transformed to llvm.invoke
// with correct basic blocks.

module {

  llvm.func @callee()

  // CHECK: llvm.func @caller
  llvm.func @caller() {
    // CHECK: llvm.invoke @callee() to ^bb2 unwind ^bb1
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return
  }
}

// -----

// Check that tokens are set to errors

module {

  llvm.func @callee()
  llvm.func @mlirAsyncRuntimeCreateToken() -> !llvm.ptr
  llvm.func @mlirAsyncRuntimeSetTokenError(!llvm.ptr) -> ()

  // CHECK: llvm.func @caller
  llvm.func @caller() {
    // CHECK: [[token:%.+]] = llvm.call @mlirAsyncRuntimeCreateToken
    %0 = llvm.call @mlirAsyncRuntimeCreateToken() : () -> !llvm.ptr
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    // CHECK: ^bb1:
    // CHECK: llvm.call @mlirAsyncRuntimeSetTokenError([[token]])
    // CHECK: ^bb2:
    llvm.return
  }
}

// -----

// Check that after setting tokens to errors, we unconditionally jump to
// success basic block.

module {

  llvm.func @callee()

  // CHECK: llvm.func @caller
  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    // CHECK:      [[fail:\^bb[0-9]+]]:
    // CHECK-NEXT: llvm.landingpad
    // CHECK:      llvm.br [[final:\^bb[0-9]+]]
    llvm.br ^bb0

    // CHECK:      [[final]]:
    // CHECK-SAME: pred
    // CHECK-SAME: [[fail]]
    ^bb0:
    llvm.return
  }
}

// -----

// Check that annotation has been deleted.

module {

  llvm.func @callee()

  // CHECK: llvm.func @caller
  // CHECK-NOT: catalyst.preInvoke
  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return
  }
}

// -----

// Check that next step is scheduled

module {

  llvm.func @callee()

  // CHECK: llvm.func @caller
  // CHECK-SAME: catalyst.preHandleError
  llvm.func @caller() {
    llvm.call @callee() { catalyst.preInvoke } : () -> ()
    llvm.return
  }
}
