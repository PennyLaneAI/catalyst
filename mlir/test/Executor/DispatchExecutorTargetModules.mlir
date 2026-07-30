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

// RUN: quantum-opt %s --dispatch-executor-targets | FileCheck %s

// This pass consumes the input IR left by cross-compile-targets which carries catalyst.object_file,
// plus a catalyst.dispatch attributes.


// The host entry opens one session, ships the object once, and rewrites each launch_kernel into an
// executor.launch reusing that session.
// CHECK-LABEL: func.func public @jit_main
// CHECK: %[[S:.*]] = executor.open("ADDR:PORT") : !executor.session
// CHECK: executor.send_binary %[[S]]("/tmp/target_compute.o") : !executor.session
// CHECK: executor.launch %[[S]]("noop", "/tmp/target_compute.o") () : !executor.session, () -> ()
// CHECK: executor.launch %[[S]]("compute", "/tmp/target_compute.o") (%{{.*}}) : !executor.session, (memref<4xf64>) -> memref<4xf64>
// One open and one send_binary for the whole function even though it has two launches.
// CHECK-NOT: executor.open
// CHECK-NOT: executor.send_binary

// The launch_kernels and the nested module are gone.
// CHECK-NOT: catalyst.launch_kernel
// CHECK-NOT: module @target_compute

module @jit_test_dispatch {

  func.func @setup() {
    return
  }

  func.func @teardown() {
    return
  }

  // inline-nested-module keeps dispatch-module calls as launch_kernel (no flattening, no host-side
  // declarations), so dispatch matches the launch_kernel callee module directly.
  func.func public @jit_main(%arg0: memref<4xf64>) -> memref<4xf64> attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @target_compute::@noop() : () -> ()
    %0 = catalyst.launch_kernel @target_compute::@compute(%arg0) : (memref<4xf64>) -> memref<4xf64>
    return %0 : memref<4xf64>
  }

  // cross-compile-targets leaves executor-dispatch modules intact (public, with bodies); dispatch
  // erases the whole module after rewriting the host launch_kernels.
  module @target_compute attributes {catalyst.object_file = "/tmp/target_compute.o", catalyst.dispatch = {address = "ADDR:PORT"}} {
    func.func public @noop() {
      return
    }
    func.func public @compute(%arg0: memref<4xf64>) -> memref<4xf64> {
      return %arg0 : memref<4xf64>
    }
  }
}
