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

// Verify that modules annotated with catalyst.target are not inlined.

// RUN: quantum-opt --inline-nested-module --split-input-file %s | FileCheck %s


// CHECK-LABEL: module @host
module @host {
  // CHECK: func.func private @ghz_0() attributes {catalyst.target = {backend = "accel"}}
  // CHECK: func.func public @jit_main
  func.func public @jit_main() attributes {llvm.emit_c_interface} {
    // CHECK-NOT: catalyst.launch_kernel
    // CHECK: call @ghz_0()
    catalyst.launch_kernel @module_accel::@ghz() : () -> ()
    func.return
  }

  // CHECK: module @module_accel attributes {catalyst.target
  // CHECK-NOT: catalyst.unique_names
  // CHECK: func.func @ghz_0()
  module @module_accel attributes {catalyst.target = {backend = "accel"}} {
    func.func @ghz() {
      func.return
    }
  }

  func.func @setup()    { func.return }
  func.func @teardown() { func.return }
}

// -----

// CHECK-LABEL: module @mixed
// CHECK: func.func private @kernel_0() attributes {catalyst.target = {backend = "accel"}}
// CHECK: func.func public @jit_run
// CHECK: call @work_0()
// CHECK: call @kernel_0()
// CHECK-NOT: module @module_cpu
// CHECK: module @module_accel2 attributes {catalyst.target
// CHECK-NOT: catalyst.unique_names
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

// CHECK-LABEL: module @duplicate_target_names
// CHECK-DAG: func.func private @kernel_1() attributes {attr = "value", catalyst.target = {backend = "accel"}}
// CHECK-DAG: func.func private @kernel_0() attributes {catalyst.target = {backend = "accel"}}
// CHECK: func.func public @jit_run
// CHECK: call @kernel_1()
// CHECK: call @kernel_0()
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
