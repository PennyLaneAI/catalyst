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

// RUN: quantum-opt %s --inject-transport-session --split-input-file --verify-diagnostics

// A coprocessor is connected to, so it cannot be given without a peer. (A missing controller is
// rejected by the parser instead, being a required parameter.)

// expected-error @below {{coprocessor requires a 'peer'}}
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x">,
  coprocessors = [#transport.node<backend_lib = "y", symbol = "foo">]>} {
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// Nor without the symbol naming what to invoke on it.

// expected-error @below {{coprocessor requires a 'symbol'}}
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x">,
  coprocessors = [#transport.node<backend_lib = "y", peer = "10.0.0.3">]>} {
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// expected-error @below {{backline transport must be 'rdma' or 'memcpy'}}
module attributes {catalyst.backline = #transport.backline<transport = "bogus", controller = #transport.node<backend_lib = "x">>} {
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// memcpy is process-local, so a controller and coprocessor must land on the same node:
// a local controller cannot pair with a remote coprocessor.

// expected-error @below {{memcpy transport requires controller and coprocessor on the same node}}
module attributes {catalyst.backline = #transport.backline<transport = "memcpy", controller = #transport.node<backend_lib = "x">,
  coprocessors = [#transport.node<backend_lib = "y", peer = "10.0.0.3", symbol = "foo", out_of_process = true>]>} {
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// ... and symmetrically, a remote controller cannot pair with a local coprocessor.

// expected-error @below {{memcpy transport requires controller and coprocessor on the same node}}
module attributes {catalyst.backline = #transport.backline<transport = "memcpy", controller = #transport.node<backend_lib = "x", out_of_process = true>,
  coprocessors = [#transport.node<backend_lib = "y", peer = "10.0.0.3", symbol = "foo">]>} {
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// Message sizes are resolved by the frontend and must be explicit in the compiler configuration.

// expected-error @below {{controller requires 'in_bytes' and 'out_bytes'}}
module attributes {catalyst.backline = #transport.backline<transport = "rdma", controller = #transport.node<backend_lib = "x">>} {
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}
