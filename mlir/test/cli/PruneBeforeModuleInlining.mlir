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

// Unused rules must disappear before module inlining visits and renames them.
// A private function called by the kernel must survive this early cleanup.
// RUN: catalyst --tool=opt %s --mlir-print-ir-before=inline-nested-module --mlir-print-ir-module-scope 2>&1 | FileCheck %s

// CHECK: IR Dump Before InlineNestedModulePass
// CHECK-NOT: unused_rule
// CHECK: module @kernel_module
// CHECK-NOT: unused_rule
// CHECK: func.func @kernel
// CHECK: call @used_rule
// CHECK-NOT: unused_rule
// CHECK: func.func private @used_rule
// CHECK-NOT: unused_rule

module @host {
  func.func @main() {
    catalyst.launch_kernel @kernel_module::@kernel() : () -> ()
    return
  }

  module @kernel_module {
    func.func private @unused_rule() attributes {target_gate = "Unused"} {
      return
    }

    func.func @kernel() {
      func.call @used_rule() : () -> ()
      return
    }

    func.func private @used_rule() attributes {target_gate = "Used"} {
      return
    }
  }
}
