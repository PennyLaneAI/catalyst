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

// A backline bring-up: marked catalyst.backline_bringup. The coprocessor's serve entry is
// launched asynchronously, then the serve entry is joined by executor.await.

// CHECK-LABEL: func.func public @jit_circuit
// CHECK:         executor.open("127.0.0.1:9000")
// CHECK-NEXT:    executor.launch %{{.*}}("circuit", "/tmp/circuit.o")
// CHECK-NOT:     executor.send_binary

// The bring-up: ship both objects, launch coproc_serve async, dial via setup_transport, then await
// the token that launch yielded.
// CHECK-LABEL: func.func @backline_setup
// CHECK-DAG:     %[[COP:.*]] = executor.open("127.0.0.1:9001")
// CHECK-DAG:     %[[CTRL:.*]] = executor.open("127.0.0.1:9000")
// CHECK-DAG:     executor.send_binary %[[COP]]("/tmp/coproc.o")
// CHECK-DAG:     executor.send_binary %[[CTRL]]("/tmp/circuit.o")
// CHECK:         %[[T:.*]] = executor.launch_async %[[COP]]("coproc_serve", "/tmp/coproc.o")
// CHECK-SAME:      : !executor.session -> !executor.token
// CHECK:         executor.launch %[[CTRL]]("setup_transport", "/tmp/circuit.o") ()
// CHECK:         executor.await %[[T]] : !executor.token

module @jit_test_async {
  func.func @setup() { return }
  func.func @teardown() { return }

  func.func public @jit_circuit() attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @module_circuit::@circuit() : () -> ()
    return
  }

  func.func @backline_setup() attributes {catalyst.backline_bringup, llvm.emit_c_interface} {
    catalyst.launch_kernel @module_coproc::@coproc_serve() {catalyst.nonblocking} : () -> ()
    catalyst.launch_kernel @module_circuit::@setup_transport() : () -> ()
    return
  }

  module @module_circuit attributes {catalyst.object_file = "/tmp/circuit.o", catalyst.dispatch = {address = "127.0.0.1:9000"}} {
    func.func public @circuit() { return }
    func.func public @setup_transport() { return }
  }
  module @module_coproc attributes {catalyst.object_file = "/tmp/coproc.o", catalyst.dispatch = {address = "127.0.0.1:9001"}} {
    func.func public @coproc_serve() { return }
  }
}
