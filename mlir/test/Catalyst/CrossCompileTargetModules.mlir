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

// A local (non-dispatch) catalyst.target module is compiled to an object and statically linked: the
// object path is recorded on the root as catalyst.object_files, the host launch_kernel is flattened
// to a func.call against an external declaration of the entry, and the module is erased.
// CHECK: module @jit_test_cross_compile_target
// CHECK-SAME: catalyst.object_files = ["{{.*}}.o"]
// CHECK: func.func private @noop()
// CHECK: func.func public @jit_main
// CHECK: call @noop()
// CHECK-NOT: catalyst.launch_kernel
// CHECK-NOT: module @target_compute
// CHECK-NOT: noop_helper

// dump-intermediate writes the extracted MLIR and translated LLVM IR to the workspace. The entry
// points are derived from the launch_kernel call edges (no separate visibility pass): @noop is the
// callee, so it is exposed through the C ABI (public + llvm.emit_c_interface); @noop_helper is not
// referenced from the host, so it is privatized.
// RUN: rm -rf %t/target_compute
// RUN: quantum-opt %s --cross-compile-targets="workspace=%t dump-intermediate=true"
// RUN: cat %t/target_compute/extracted.mlir | FileCheck %s --check-prefix=EXTRACTED
// RUN: cat %t/target_compute/target_compute.ll | FileCheck %s --check-prefix=LL
// EXTRACTED: func.func @noop() attributes {llvm.emit_c_interface}
// EXTRACTED: func.func private @noop_helper()
// LL: define {{.*}}@noop

module @jit_test_cross_compile_target {

  func.func public @jit_main() attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @target_compute::@noop() : () -> ()
    return
  }

  // Both functions start public; cross-compile-targets derives the entry set from the host's
  // launch_kernel and privatizes the unreferenced helper.
  module @target_compute attributes {catalyst.target = {}} {
    func.func public @noop() {
      return
    }
    func.func public @noop_helper() {
      return
    }
  }
}
