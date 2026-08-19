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

// Controller-only: bring-up into @setup, release into @teardown, qnode entry untouched.

// CHECK-LABEL: func.func public @jit_circuit
// CHECK-NOT:     {{[^#]}}transport.
// CHECK:         catalyst.launch_kernel

// CHECK-LABEL: func.func @setup
// CHECK:         quantum.init
// CHECK:         %[[S:.*]] = transport.create {{.*}}key = "controller"{{.*}} -> !transport.session<controller>
// CHECK:         transport.connect %[[S]] {oob_port = 18590 : ui16, peer = "127.0.0.1"} : !transport.session<controller>
// CHECK:         transport.exchange_keys %[[S]] : !transport.session<controller>
// CHECK:         transport.establish_channel %[[S]] "rdma" : !transport.session<controller>
// CHECK:         transport.set_message_sizes %[[S]] {{.*}} : !transport.session<controller>
// CHECK:         transport.start %[[S]] : !transport.session<controller>

// CHECK-LABEL: func.func @teardown
// CHECK:         quantum.finalize
// CHECK:         %[[T:.*]] = transport.get_session {{.*}} : !transport.session<controller>
// CHECK:         transport.stop %[[T]] : !transport.session<controller>
// CHECK:         transport.destroy %[[T]] : !transport.session<controller>
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x", config = "c",
    peer = "127.0.0.1", oob_port = 18590 : i16,
    in_bytes = 8 : i64, out_bytes = 8 : i64>>} {
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

// Remote controller, no coprocessors, peer present: the controller self-dials (loopback).

// CHECK-LABEL: func.func @setup_transport
// CHECK:         %[[S:.*]] = transport.create {{.*}}key = "controller"{{.*}} -> !transport.session<controller>
// CHECK:         transport.connect %[[S]] {oob_port = 18590 : ui16, peer = "127.0.0.1"} : !transport.session<controller>
// CHECK:         transport.exchange_keys %[[S]] : !transport.session<controller>
// CHECK:         transport.establish_channel %[[S]] "rdma" : !transport.session<controller>
// CHECK:         transport.set_message_sizes %[[S]] {{.*}} : !transport.session<controller>
// CHECK:         transport.start %[[S]] : !transport.session<controller>

// CHECK-LABEL: func.func @teardown_transport
// CHECK:         %[[T:.*]] = transport.get_session {{.*}} : !transport.session<controller>
// CHECK:         transport.stop %[[T]] : !transport.session<controller>
// CHECK:         transport.destroy %[[T]] : !transport.session<controller>

// The host launches the controller lifecycle; with no coprocessor, no serve precedes it.
// CHECK-LABEL: func.func @setup() attributes {catalyst.backline_bringup}
// CHECK:         quantum.init
// CHECK-NEXT:    catalyst.launch_kernel @module_ctrl::@setup_transport()

// CHECK-LABEL: func.func @teardown()
// CHECK:         quantum.finalize
// CHECK-NEXT:    catalyst.launch_kernel @module_ctrl::@teardown_transport()
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x", config = "c",
    peer = "127.0.0.1", oob_port = 18590 : i16,
    triple = "aarch64-unknown-linux-gnu", address = "h:1", out_of_process = true,
    in_bytes = 8 : i64, out_bytes = 8 : i64>>} {
  func.func public @jit_circuit() -> tensor<4xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_ctrl::@circuit() : () -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  module @module_ctrl attributes {catalyst.target = {triple = "aarch64-unknown-linux-gnu"}, catalyst.dispatch = {address = "h:1"}, catalyst.backline_role = "controller"} {
    func.func public @circuit() -> tensor<4xf64> {
      %c = arith.constant dense<0.0> : tensor<4xf64>
      return %c : tensor<4xf64>
    }
  }
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// Remote controller, no coprocessors, no peer: nothing to dial, so no transport ops are generated.
// The controller module is still cross-compiled and dispatched, but @setup/@teardown are untouched

// CHECK-LABEL: module @module_ctrl
// CHECK-NOT:   setup_transport
// CHECK-NOT:   teardown_transport
// CHECK-NOT:   transport.

// CHECK:      func.func @setup() {
// CHECK-NEXT:   quantum.init
// CHECK-NEXT:   return
// CHECK:      func.func @teardown() {
// CHECK-NEXT:   quantum.finalize
// CHECK-NEXT:   return
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x", config = "c", triple = "aarch64-unknown-linux-gnu", address = "h:1", out_of_process = true, in_bytes = 8 : i64, out_bytes = 8 : i64>>} {
  func.func public @jit_circuit() -> tensor<4xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_ctrl::@circuit() : () -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  module @module_ctrl attributes {catalyst.target = {triple = "aarch64-unknown-linux-gnu"}, catalyst.dispatch = {address = "h:1"}, catalyst.backline_role = "controller"} {
    func.func public @circuit() -> tensor<4xf64> {
      %c = arith.constant dense<0.0> : tensor<4xf64>
      return %c : tensor<4xf64>
    }
  }
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// Co-located coprocessor: both roles brought up in @setup, released in @teardown.

// CHECK-LABEL: func.func public @jit_circuit
// CHECK-NOT:     {{[^#]}}transport.
// CHECK:         catalyst.launch_kernel

// CHECK-LABEL: func.func @setup
// CHECK:         quantum.init
// CHECK-DAG:     transport.create {{.*}} -> !transport.session<controller>
// CHECK-DAG:     transport.create {{.*}} -> !transport.session<coprocessor>
// CHECK:         transport.connect_async %{{.*}} : !transport.session<coprocessor> -> !transport.token
// CHECK:         transport.await
// CHECK:         transport.set_coprocessor_fn %{{.*}} {symbol = "coproc_fn"} : !transport.session<coprocessor>
// CHECK:         transport.start

// CHECK-LABEL: func.func @teardown
// CHECK:         quantum.finalize
// CHECK-DAG:     transport.get_session {{.*}} : !transport.session<controller>
// CHECK-DAG:     transport.get_session {{.*}} : !transport.session<coprocessor>
// CHECK:         transport.destroy
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16, in_bytes = 8 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16, symbol = "coproc_fn">]>} {
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

// Distributed: each role's lifecycle goes into its own target module, launched from
// @setup/@teardown in dependency order.

// The controller module.
// CHECK-DAG: func.func @setup_transport
// CHECK-DAG: func.func @teardown_transport
// CHECK-DAG: transport.create {{.*}} -> !transport.session<controller>

// The host funcs launch them.
// CHECK:      func.func @setup() attributes {catalyst.backline_bringup}
// CHECK:        quantum.init
// CHECK-NEXT:   catalyst.launch_kernel @module_coproc::@coproc_serve() {catalyst.nonblocking}
// CHECK-NEXT:   catalyst.launch_kernel @module_ctrl::@setup_transport()
// CHECK:      func.func @teardown()
// CHECK:        quantum.finalize
// CHECK-NEXT:   catalyst.launch_kernel @module_coproc::@coproc_stop()
// CHECK-NEXT:   catalyst.launch_kernel @module_ctrl::@teardown_transport()

// The coprocessor module.
// CHECK-DAG: module @module_coproc attributes {{.*}}catalyst.target = {triple = "x86_64-unknown-linux-gnu"}
// CHECK-DAG: func.func @coproc_serve
// CHECK-DAG: transport.set_coprocessor_fn {{.*}} {symbol = "foo"} : !transport.session<coprocessor>
// CHECK-DAG: func.func @coproc_stop
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x", config = "c", triple = "aarch64-unknown-linux-gnu", address = "h:1", out_of_process = true, in_bytes = 3 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "y", config = "c", peer = "10.0.0.3", oob_port = 18560 : i16, triple = "x86_64-unknown-linux-gnu", address = "h:2", out_of_process = true, symbol = "foo", name = "cop0">]>} {
  func.func public @jit_circuit() -> tensor<4xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_ctrl::@circuit() : () -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  module @module_ctrl attributes {catalyst.target = {triple = "aarch64-unknown-linux-gnu"}, catalyst.dispatch = {address = "h:1"}, catalyst.backline_role = "controller"} {
    func.func public @circuit() -> tensor<4xf64> {
      %c = arith.constant dense<0.0> : tensor<4xf64>
      return %c : tensor<4xf64>
    }
  }
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// Distributed with multiple remote coprocessors: each gets its own suffixed target module and
// lifecycle funcs, so the controller dials both from its module, and the host launches every role
// in order.

// The controller dials both coprocessors from its own module.
// CHECK:      transport.create {{.*}}key = "cop1"{{.*}} -> !transport.session<controller>

// @setup launches both serves (nonblocking) before the controller setup.
// CHECK:      catalyst.launch_kernel @module_coproc.0::@coproc_serve.0() {catalyst.nonblocking}
// CHECK-NEXT: catalyst.launch_kernel @module_coproc.1::@coproc_serve.1() {catalyst.nonblocking}
// CHECK-NEXT: catalyst.launch_kernel @module_ctrl::@setup_transport()

// @teardown stops both coprocessors before the controller teardown.
// CHECK:      catalyst.launch_kernel @module_coproc.0::@coproc_stop.0()
// CHECK-NEXT: catalyst.launch_kernel @module_coproc.1::@coproc_stop.1()
// CHECK-NEXT: catalyst.launch_kernel @module_ctrl::@teardown_transport()

// The second coprocessor's own module carries its distinct serve function.
// CHECK:      transport.set_coprocessor_fn %{{.*}} {symbol = "bar"} : !transport.session<coprocessor>
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x", config = "c", triple = "aarch64-unknown-linux-gnu", address = "h:1", out_of_process = true, in_bytes = 3 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "y", config = "c", peer = "10.0.0.3", oob_port = 18560 : i16, triple = "x86_64-unknown-linux-gnu", address = "h:2", out_of_process = true, symbol = "foo", name = "cop0">, #transport.node<backend_lib = "z", config = "c", peer = "10.0.0.4", oob_port = 18561 : i16, triple = "x86_64-unknown-linux-gnu", address = "h:3", out_of_process = true, symbol = "bar", name = "cop1">]>} {
  func.func public @jit_circuit() -> tensor<4xf64> attributes {llvm.emit_c_interface} {
    %0 = catalyst.launch_kernel @module_ctrl::@circuit() : () -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }
  module @module_ctrl attributes {catalyst.target = {triple = "aarch64-unknown-linux-gnu"}, catalyst.dispatch = {address = "h:1"}, catalyst.backline_role = "controller"} {
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
// CHECK-NOT:     {{[^#]}}transport.
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

// Multiple co-located coprocessors: one controller session each, keyed by the peer name.

// CHECK-LABEL: func.func public @jit_circuit
// CHECK-NOT:     {{[^#]}}transport.

// CHECK-LABEL: func.func @setup
// CHECK:         quantum.init
// CHECK-DAG:     transport.create {{.*}}key = "cop0"{{.*}} -> !transport.session<controller>
// CHECK-DAG:     transport.create {{.*}}key = "cop1"{{.*}} -> !transport.session<controller>
// CHECK-DAG:     transport.create {{.*}}key = "cop0"{{.*}} -> !transport.session<coprocessor>
// CHECK-DAG:     transport.create {{.*}}key = "cop1"{{.*}} -> !transport.session<coprocessor>
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16, in_bytes = 3 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16, symbol = "coproc_fn", name = "cop0">, #transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18591 : i16, symbol = "coproc_fn", name = "cop1">]>} {
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

// Memcpy backline transport selects the local channel. Memcpy pairs on the session key,
// so the emitted transport.connect / connect_async ops carry no peer or oob_port.

// CHECK-LABEL: func.func @setup
// CHECK:         quantum.init
// CHECK-DAG:     transport.connect_async %{{.*}} : !transport.session<coprocessor> -> !transport.token
// CHECK-DAG:     transport.connect %{{.*}} : !transport.session<controller>
// CHECK-NOT:     peer =
// CHECK-NOT:     oob_port =
// CHECK-DAG:     transport.establish_channel %{{.*}} "memcpy" : !transport.session<controller>
// CHECK-DAG:     transport.establish_channel %{{.*}} "memcpy" : !transport.session<coprocessor>
module attributes {catalyst.backline = #transport.backline<transport = "memcpy", controller = #transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18592 : i16, in_bytes = 3 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "y", config = "c", peer = "127.0.0.1", oob_port = 18592 : i16, symbol = "coproc_fn", name = "cop0">]>} {
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

// Remote controller with a local coprocessor: the roles are in different processes, so the
// coprocessor's handshake splits around the launch that dials it, and each session is released where
// it was created.

// CHECK-LABEL: func.func @setup_transport
// CHECK:         transport.connect %{{.*}} {oob_port = 18560 : ui16, peer = "10.0.0.3"}

// CHECK-LABEL: func.func @setup() attributes {catalyst.backline_bringup}
// CHECK:         %[[CO:.*]] = transport.create {{.*}} -> !transport.session<coprocessor>
// CHECK:         %[[TOK:.*]] = transport.connect_async %[[CO]]
// CHECK:         transport.set_coprocessor_fn %[[CO]]
// CHECK:         catalyst.launch_kernel @module_ctrl::@setup_transport()
// CHECK:         transport.await %[[TOK]]
// CHECK:         transport.start %[[CO]]

// The coprocessor is released in the host, the controller in its own module.
// CHECK-LABEL: func.func @teardown()
// CHECK:         transport.destroy %{{.*}} : !transport.session<coprocessor>
// CHECK:         catalyst.launch_kernel @module_ctrl::@teardown_transport()
module attributes {catalyst.backline = #transport.backline<transport = "rdma",
  controller = #transport.node<backend_lib = "x", out_of_process = true, address = "h:1", in_bytes = 8 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "y", peer = "10.0.0.3", oob_port = 18560 : i16, symbol = "foo", name = "cop0">]>} {
  func.func public @jit_circuit() attributes {llvm.emit_c_interface} {
    catalyst.launch_kernel @module_ctrl::@circuit() : () -> ()
    return
  }
  module @module_ctrl attributes {catalyst.target = {triple = "aarch64-unknown-linux-gnu"}, catalyst.dispatch = {address = "h:1"}, catalyst.backline_role = "controller"} {
    func.func public @circuit() { return }
  }
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// Local controller with a remote coprocessor: the controller dials inline from the host, so the
// coprocessor's serve entry is launched before that dial and its stop before the controller's.

// CHECK-LABEL: func.func @setup()
// CHECK:         quantum.init
// CHECK-NEXT:    catalyst.launch_kernel @module_coproc::@coproc_serve() {catalyst.nonblocking}
// CHECK-NEXT:    transport.create {{.*}} -> !transport.session<controller>
// CHECK:         transport.connect
// CHECK:         transport.start

// CHECK-LABEL: func.func @teardown()
// CHECK:         quantum.finalize
// CHECK-NEXT:    catalyst.launch_kernel @module_coproc::@coproc_stop()
// CHECK:         transport.destroy %{{.*}} : !transport.session<controller>
module attributes {catalyst.backline = #transport.backline<transport = "rdma",
  controller = #transport.node<backend_lib = "x", in_bytes = 8 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "y", peer = "10.0.0.3", oob_port = 18560 : i16, symbol = "foo", name = "cop0", out_of_process = true, address = "h:2", triple = "x86_64-unknown-linux-gnu">]>} {
  func.func public @jit_circuit() attributes {llvm.emit_c_interface} {
    return
  }
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}
