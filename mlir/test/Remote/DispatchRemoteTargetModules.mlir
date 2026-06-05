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

// RUN: quantum-opt %s --dispatch-remote-targets | FileCheck %s

// This pass consumes the input IR left by cross-compile-targets which carries catalyst.object_file,
// plus a catalyst.dispatch attributes.

// setup() opens the session once and ships the module's object.
// CHECK-LABEL: func.func @setup()
// CHECK: remote.open("ADDR:PORT")
// CHECK: remote.send_binary("ADDR:PORT", "/tmp/target_compute.o")
// CHECK-NOT: remote.send_binary
// CHECK: return

// teardown() is left untouched: the runtime closes sessions at process exit.
// CHECK-LABEL: func.func @teardown()
// CHECK-NOT: remote.
// CHECK: return

// Each host-side call is rewritten to its own remote.launch, preserving the call's
// operand/result types (a typed call exercises the memref-result path).
// CHECK-LABEL: func.func public @jit_main
// CHECK: remote.launch("noop", "ADDR:PORT") () : () -> ()
// CHECK: remote.launch("compute", "ADDR:PORT") (%{{.*}}) : (memref<4xf64>) -> memref<4xf64>

// The bodyless declarations and the nested module are erased.
// CHECK-NOT: func.call @noop
// CHECK-NOT: func.call @compute
// CHECK-NOT: module @target_compute

module @jit_test_dispatch {

  func.func @setup() {
    return
  }

  func.func @teardown() {
    return
  }

  func.func private @noop()
  func.func private @compute(memref<4xf64>) -> memref<4xf64>

  func.func public @jit_main(%arg0: memref<4xf64>) -> memref<4xf64> attributes {llvm.emit_c_interface} {
    func.call @noop() : () -> ()
    %0 = func.call @compute(%arg0) : (memref<4xf64>) -> memref<4xf64>
    return %0 : memref<4xf64>
  }

  module @target_compute attributes {catalyst.object_file = "/tmp/target_compute.o", catalyst.dispatch = {address = "ADDR:PORT"}} {
    func.func private @noop() attributes {catalyst.entry_point}
    func.func private @compute(memref<4xf64>) -> memref<4xf64> attributes {catalyst.entry_point}
  }
}
