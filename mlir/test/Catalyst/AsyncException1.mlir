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

// RUN: quantum-opt --add-exception-handling=stop-after-step=1 --verify-diagnostics --split-input-file %s | FileCheck %s

// Check to make sure that callers to qnodes
// are left alone if they are not annotated with "presplitcoroutine".

module {
  llvm.func @callee() attributes { qnode } {}

  llvm.func @caller() {
    // CHECK: llvm.call @callee
    // CHECK-NOT: catalyst.preInvoke
    llvm.call @callee() : () -> ()
    llvm.return
  }
}


// -----

// Check to make sure that functions which are annotated with presplitcoroutine
// leave their callees alone if the callees are not qnodes.


module {
  llvm.func @callee() {}

  llvm.func @caller() attributes { passthrough = ["presplitcoroutine"] } {
    // CHECK: llvm.call @callee
    // CHECK-NOT: catalyst.preInvoke
    llvm.call @callee() : () -> ()
    llvm.return
  }
}

// -----

// Check to make sure that functions which are annotated with presplitcoroutine
// and their callees are qnodes get annotated with catalyst.preInvoke


module {
  llvm.func @callee() attributes { qnode } {}

  llvm.func @caller() attributes { passthrough = ["presplitcoroutine"] } {
    // CHECK: llvm.call @callee
    // CHECK-SAME: catalyst.preInvoke
    llvm.call @callee() : () -> ()
    llvm.return
  }
}
