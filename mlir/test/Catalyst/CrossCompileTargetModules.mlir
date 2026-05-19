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
// RUN: quantum-opt %s \
// RUN:   --cross-compile-remote-kernels="workspace=%t target=x86_64-linux-gnu" \
// RUN:   | FileCheck %s

// CHECK-NOT: catalyst.target
// CHECK-NOT: module @target_compute

// CHECK:      func.func public @jit_main
// CHECK-NEXT: catalyst.custom_call fn("remote_call")
// CHECK-SAME: catalyst.remote_address = "ADDRESS:PORT"

// CHECK:      func.func @setup
// CHECK:      catalyst.custom_call fn("remote_open")
// CHECK-SAME: catalyst.remote_address = "ADDRESS:PORT"
// CHECK:      catalyst.custom_call fn("remote_send_binary")
// CHECK-SAME: catalyst.remote_address = "ADDRESS:PORT"

// CHECK:      func.func @teardown
// CHECK:      catalyst.custom_call fn("remote_close")

module @jit_test_cross_compile_target {

  func.func private @noop() attributes {catalyst.target = {address = "ADDRESS:PORT", backend = "cpu_target"}}

  func.func public @jit_main() attributes {llvm.emit_c_interface} {
    func.call @noop() : () -> ()
    return
  }

  module @target_compute attributes {
    catalyst.target = {address = "ADDRESS:PORT", backend = "cpu_target"}
  } {
    func.func public @noop() {
      return
    }
  }

  func.func @setup() {
    quantum.init
    return
  }

  func.func @teardown() {
    quantum.finalize
    return
  }
}
