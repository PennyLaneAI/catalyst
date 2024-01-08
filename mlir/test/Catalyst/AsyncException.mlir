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

// Check to make sure that personality was added

module {
  // CHECK-LABEL: @__gxx_personality_v4
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  llvm.func @caller() {
    llvm.call @callee() : () -> ()
    llvm.return
  }
}

// -----

// Check to make sure that the the `llvm.call` op has been annotated
module {
  llvm.func @callee() attributes { qnode } {
    llvm.return
  }

  llvm.func @caller() {
    // CHECK: llvm.call
    // CHECK-SAME: catalyst.transformed
    llvm.call @callee() : () -> ()
    llvm.return
  }
}

