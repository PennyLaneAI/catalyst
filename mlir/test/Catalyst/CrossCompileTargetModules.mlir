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

// RUN: mkdir -p %t
// RUN: quantum-opt %s --cross-compile-targets="workspace=%t" | FileCheck %s

// The catalyst.target module is compiled to an object file recorded as
// catalyst.object_file, and reduced to declarations of its entry functions.
// CHECK: module @target_compute
// CHECK-SAME: catalyst.object_file = "{{.*}}.o"
// CHECK: func.func private @noop() attributes {catalyst.entry_point}
// CHECK-NOT: noop_helper

// dump-intermediate writes the extracted MLIR and translated LLVM IR to the workspace.
// RUN: rm -rf %t/target_compute
// RUN: quantum-opt %s --cross-compile-targets="workspace=%t dump-intermediate=true"
// RUN: cat %t/target_compute/extracted.mlir | FileCheck %s --check-prefix=EXTRACTED
// RUN: cat %t/target_compute/target_compute.ll | FileCheck %s --check-prefix=LL
// EXTRACTED: func.func {{.*}}@noop
// LL: define {{.*}}@noop

module @jit_test_cross_compile_target {

  func.func private @noop() attributes {catalyst.target = {backend = "my-backend"}}

  func.func public @jit_main() attributes {llvm.emit_c_interface} {
    func.call @noop() : () -> ()
    return
  }

  module @target_compute attributes {catalyst.target = {backend = "my-backend"}} {
    func.func public @noop() attributes {catalyst.entry_point} {
      return
    }
    func.func private @noop_helper() {
      return
    }
  }
}
