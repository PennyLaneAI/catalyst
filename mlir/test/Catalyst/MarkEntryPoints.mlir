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

// RUN: quantum-opt %s --mark-entry-points --split-input-file | FileCheck %s

// Externally-called function in a nested target module gets marked as entry-points in
// the contataining module

// CHECK-LABEL: module @case_simple
module @case_simple {
  func.func public @host() {
    func.call @entry() : () -> ()
    return
  }
  // CHECK: func.func private @entry()
  // CHECK-NOT: catalyst.entry_point
  func.func private @entry()

  module @target {
    // CHECK: func.func public @entry()
    // CHECK-SAME: catalyst.entry_point
    func.func public @entry() {
      return
    }
    // CHECK: func.func private @helper()
    // CHECK-NOT: catalyst.entry_point
    func.func private @helper() {
      return
    }
  }
}

// -----

// Target module with no externally-referenced functions

// CHECK-LABEL: module @case_orphan
module @case_orphan {
  func.func public @host() { return }

  module @target {
    // CHECK-NOT: catalyst.entry_point
    func.func public @unused() { return }
  }
}

// -----

// Intra-module references don't count as "external".

// CHECK-LABEL: module @case_intra_module
module @case_intra_module {
  func.func public @host() {
    func.call @entry() : () -> ()
    return
  }
  // CHECK: func.func private @entry()
  // CHECK-NOT: catalyst.entry_point
  func.func private @entry()

  module @qnode {
    // CHECK: func.func public @entry()
    // CHECK-SAME: catalyst.entry_point
    func.func public @entry() {
      func.call @callee() : () -> ()
      return
    }
    // CHECK: func.func private @callee()
    // CHECK-NOT: catalyst.entry_point
    func.func private @callee() { return }
  }
}

