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

// Verify that modules annotated with catalyst.target are not inlined, renamed, or flattened: their
// launch_kernel survives intact for a later consumer (e.g. dispatch-remote-targets).

// RUN: quantum-opt --inline-nested-module --split-input-file %s | FileCheck %s


// CHECK-LABEL: module @host
module @host {
  // No root declaration is emitted; the launch_kernel is kept (not flattened to a func.call).
  // CHECK-NOT: func.func private @ghz
  // CHECK: func.func public @jit_main
  func.func public @jit_main() attributes {llvm.emit_c_interface} {
    // CHECK: catalyst.launch_kernel @module_accel::@ghz() : () -> ()
    catalyst.launch_kernel @module_accel::@ghz() : () -> ()
    func.return
  }

  // CHECK: module @module_accel attributes {catalyst.target
  // CHECK-NOT: catalyst.unique_names
  // CHECK: func.func @ghz()
  module @module_accel attributes {catalyst.target = {backend = "accel"}} {
    func.func @ghz() {
      func.return
    }
  }

  func.func @setup()    { func.return }
  func.func @teardown() { func.return }
}

// -----

// A mix: the non-target module is inlined+flattened (call @work_0); the catalyst.target module
// keeps its nested launch_kernel.
// CHECK-LABEL: module @mixed
// CHECK: func.func public @jit_run
// CHECK: call @work_0()
// CHECK: catalyst.launch_kernel @module_accel2::@kernel()
// CHECK: func.func @work_0()
// CHECK-NOT: module @module_cpu
// CHECK: module @module_accel2 attributes {catalyst.target
// CHECK-NOT: catalyst.unique_names
// CHECK: func.func @kernel()
module @mixed {

  func.func public @jit_run() attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @module_cpu::@work() : () -> ()
    catalyst.launch_kernel @module_accel2::@kernel() : () -> ()
    func.return
  }

  module @module_cpu {
    func.func @work() {
      func.return
    }
  }

  module @module_accel2 attributes {catalyst.target = {backend = "accel"}} {
    func.func @kernel() {
      func.return
    }
  }

  func.func @setup()    { func.return }
  func.func @teardown() { func.return }
}

// -----

// Two target modules with the same entry name: since neither is inlined or renamed, both keep the
// name @kernel in their own module (no collision — separate symbol tables), and both launch_kernels
// survive.
// CHECK-LABEL: module @duplicate_target_names
// CHECK: func.func public @jit_run
// CHECK: catalyst.launch_kernel @module_accel_a::@kernel()
// CHECK: catalyst.launch_kernel @module_accel_b::@kernel()
// CHECK-DAG: module @module_accel_a attributes {catalyst.target
// CHECK-DAG: module @module_accel_b attributes {catalyst.target
// CHECK-NOT: catalyst.unique_names
module @duplicate_target_names {

  func.func public @jit_run() attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @module_accel_a::@kernel() : () -> ()
    catalyst.launch_kernel @module_accel_b::@kernel() : () -> ()
    func.return
  }

  module @module_accel_a attributes {catalyst.target = {backend = "accel"}} {
    func.func @kernel() attributes {attr = "value"} {
      func.return
    }
  }

  module @module_accel_b attributes {catalyst.target = {backend = "accel"}} {
    func.func @kernel() {
      func.return
    }
  }
}

// -----

// Dispatch modules (catalyst.dispatch) are not inlined, not flattened, AND not renamed:
// dispatch-remote-targets consumes the launch_kernel directly and resolves the entry in the
// object's own JITDylib on the executor, so the kernel keeps its original name and no root
// declaration is emitted.
// CHECK-LABEL: module @host_dispatch
// CHECK-NOT: func.func private @ghz
// CHECK: func.func public @jit_main
// CHECK: catalyst.launch_kernel @module_remote::@ghz() : () -> ()
// CHECK: module @module_remote attributes {catalyst.dispatch
// CHECK: func.func @ghz()
module @host_dispatch {

  func.func public @jit_main() attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @module_remote::@ghz() : () -> ()
    func.return
  }

  module @module_remote attributes {catalyst.target = {backend = "accel"}, catalyst.dispatch = {address = "h:1"}} {
    func.func @ghz() {
      func.return
    }
  }

  func.func @setup()    { func.return }
  func.func @teardown() { func.return }
}
