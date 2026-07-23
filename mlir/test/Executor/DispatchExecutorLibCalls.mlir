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

// RUN: quantum-opt %s --split-input-file --dispatch-executor-targets --verify-diagnostics | FileCheck %s

// dispatch-executor-targets also rewrites standalone executor library calls which are 
// catalyst.custom_call ops whose backend_config carries a `dispatch` entry into
// executor.call ops invoking a symbol already loaded on the executor.

// -----

// When backend_config.dispatch names the executor explicitly, it is rewritten to a executor.call on 
// that address, and a session is opened once in setup().
// CHECK-LABEL: func.func @setup()
// CHECK: executor.open("ADDR:PORT")
// CHECK: return
// CHECK-LABEL: func.func @main
// CHECK: executor.call("fpga_trampoline_a_setup", "ADDR:PORT")
// CHECK-SAME: (memref<256xi8>) -> ()
// CHECK-NOT: catalyst.custom_call
module @jit_bound {
  func.func @setup() {
    return
  }
  func.func @main(%arg0: memref<256xi8>) {
    catalyst.custom_call fn("fpga_trampoline_a_setup") (%arg0) {backend_config = {dispatch = "ADDR:PORT"}, number_original_arg = 1 : i32} : (memref<256xi8>) -> ()
    return
  }
}

// -----

// backend_config.dispatch = "" binds to the program's single executor, which
// is supplied by the QNode target module's catalyst.dispatch address.
// CHECK-LABEL: func.func @setup()
// CHECK: executor.open("ADDR:PORT")
// CHECK-LABEL: func.func @main
// CHECK: executor.call("fpga_trampoline_a_teardown", "ADDR:PORT")
// CHECK-NOT: catalyst.custom_call
module @jit_inherit {
  func.func @setup() {
    return
  }
  func.func @teardown() {
    return
  }
  func.func @main() {
    catalyst.launch_kernel @target::@compute() : () -> ()
    catalyst.custom_call fn("fpga_trampoline_a_teardown") () {backend_config = {dispatch = ""}} : () -> ()
    return
  }
  module @target attributes {catalyst.object_file = "/tmp/target.o", catalyst.dispatch = {address = "ADDR:PORT"}} {
    func.func public @compute() {
      return
    }
  }
}

// -----

// When the program targets more than one executor, the custom_call should explicitly bind the 
// dispatch otherwise it would be ambiguous.
module @jit_ambiguous {
  func.func @setup() {
    return
  }
  func.func @main() {
    catalyst.launch_kernel @t1::@c1() : () -> ()
    catalyst.launch_kernel @t2::@c2() : () -> ()
    // expected-error @below {{ambiguous executor}}
    catalyst.custom_call fn("foo") () {backend_config = {dispatch = ""}} : () -> ()
    return
  }
  module @t1 attributes {catalyst.object_file = "/tmp/t1.o", catalyst.dispatch = {address = "host:1"}} {
    func.func public @c1() {
      return
    }
  }
  module @t2 attributes {catalyst.object_file = "/tmp/t2.o", catalyst.dispatch = {address = "host:2"}} {
    func.func public @c2() {
      return
    }
  }
}
