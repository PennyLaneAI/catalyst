// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Step 5 is cleaning up
// RUN: quantum-opt --inline-nested-module=stop-after-step=5 --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @outer

// This tests that all temporary annotations are removed.
// CHECK-NOT: catalyst
module @outer {
  module @inner {
    func.func @f() {
      return
    }
  }

  catalyst.launch_kernel @inner::@f() : () -> ()
}

// -----

// Test external API func decl names are preserved
module @global {

  // CHECK-DAG: func.func private @f()
  // CHECK-DAG: func.func @main_0() {
  // CHECK-DAG:   call @f() : () -> ()
  // CHECK-DAG:   return
  // CHECK-DAG: }
  // CHECK-DAG: func.func @main_1() {
  // CHECK-DAG:   call @f() : () -> ()
  // CHECK-DAG:   return
  // CHECK-DAG: }

  module @local0 {
    func.func private @f()
    func.func @main() {
      func.call @f() : () -> ()
      return
    }
  }

  module @local1 {
    func.func private @f()
    func.func @main() {
      func.call @f() : () -> ()
      return
    }
  }

}
