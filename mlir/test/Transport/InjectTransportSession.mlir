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

// RUN: quantum-opt --inject-transport-session --split-input-file %s | FileCheck %s

// Controller-only backline: synchronous bring-up before the launch, teardown after.

// CHECK-LABEL: func.func public @jit_circuit
// CHECK:         %[[S:.*]] = transport.create {{.*}} -> !transport.session<controller>
// CHECK:         transport.connect %[[S]] {oob_port = 18590 : i16, peer = "127.0.0.1"} : !transport.session<controller>
// CHECK:         transport.exchange_keys %[[S]] : !transport.session<controller>
// CHECK:         transport.establish_channel %[[S]] "cpu_verbs" : !transport.session<controller>
// CHECK:         transport.commit_work_item %[[S]] {{.*}} : !transport.session<controller>
// CHECK:         transport.start %[[S]] : !transport.session<controller>
// CHECK:         catalyst.launch_kernel
// CHECK:         transport.stop %[[S]] : !transport.session<controller>
// CHECK:         transport.destroy %[[S]] : !transport.session<controller>
// CHECK-NOT:     transport.unstash
// CHECK-LABEL: func.func @setup
// CHECK-NEXT:    quantum.init
// CHECK-NEXT:    return
module attributes {catalyst.backline = {
  controller = {
    backend_lib = "x", config = "c",
    peer = "127.0.0.1", oob_port = 18590 : i16, data_path = "cpu_verbs",
    in_bytes = 8 : i64, out_bytes = 8 : i64
  }
}} {
  func.func public @jit_circuit() -> tensor<4xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  module @module_circuit {
    func.func public @circuit() -> tensor<4xf64> {
      %c = arith.constant dense<0.0> : tensor<4xf64>
      return %c : tensor<4xf64>
    }
  }
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// Coprocessor backline: both roles brought up in the host -- async coprocessor + barrier,
// synchronous controller dial.

// CHECK-LABEL: func.func public @jit_circuit
// CHECK-DAG:     transport.create {{.*}} -> !transport.session<controller>
// CHECK-DAG:     transport.create {{.*}} -> !transport.session<coprocessor>
// CHECK:         transport.connect_async %{{.*}} : !transport.session<coprocessor> -> !transport.token
// CHECK:         transport.barrier
// CHECK:         transport.set_coprocessor_fn %{{.*}} {symbol = "coproc_fn"} : !transport.session<coprocessor>
// CHECK:         catalyst.launch_kernel
// CHECK:         transport.stop %{{.*}} : !transport.session<coprocessor>
// CHECK:         transport.destroy %{{.*}} : !transport.session<coprocessor>
// CHECK-NOT:     transport.unstash
module attributes {catalyst.backline = {
  controller = {backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16, in_bytes = 8 : i64, out_bytes = 8 : i64},
  coprocessor = {backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16, data_path = "gpu_engine", symbol = "coproc_fn"}
}} {
  func.func public @jit_circuit() -> tensor<4xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  module @module_circuit {
    func.func public @circuit() -> tensor<4xf64> {
      %c = arith.constant dense<0.0> : tensor<4xf64>
      return %c : tensor<4xf64>
    }
  }
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// No catalyst.backline: the module is left untouched.

// CHECK-LABEL: func.func @untouched
// CHECK-NOT:     transport.
module {
  func.func @untouched() -> tensor<4xf64> {
    %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  module @module_circuit {
    func.func public @circuit() -> tensor<4xf64> {
      %c = arith.constant dense<0.0> : tensor<4xf64>
      return %c : tensor<4xf64>
    }
  }
}

// -----

// Multiple coprocessors: one controller per coprocessor, keyed by the peer's name.

// CHECK-LABEL: func.func public @jit_circuit
// CHECK-DAG:     transport.create {{.*}}key = "cop0"{{.*}} -> !transport.session<controller>
// CHECK-DAG:     transport.create {{.*}}key = "cop1"{{.*}} -> !transport.session<controller>
// Coprocessor creates carry no key:
// CHECK-DAG:     transport.create {backend_lib = "x", config = "c"} -> !transport.session<coprocessor>
// CHECK-DAG:     transport.create {backend_lib = "x", config = "c"} -> !transport.session<coprocessor>
// CHECK:         catalyst.launch_kernel
module attributes {catalyst.backline = {
  controller = {backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16, in_bytes = 3 : i64, out_bytes = 8 : i64},
  coprocessors = [
    {backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16, data_path = "cpu_verbs", symbol = "coproc_fn", name = "cop0"},
    {backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18591 : i16, data_path = "cpu_verbs", symbol = "coproc_fn", name = "cop1"}
  ]
}} {
  func.func public @jit_circuit() -> tensor<4xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_circuit::@circuit() : () -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  module @module_circuit {
    func.func public @circuit() -> tensor<4xf64> {
      %c = arith.constant dense<0.0> : tensor<4xf64>
      return %c : tensor<4xf64>
    }
  }
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}
