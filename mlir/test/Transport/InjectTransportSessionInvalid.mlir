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
module attributes {catalyst.backline = #transport.backline<transport = "net", controller = #transport.node<backend_lib = "x">,
  coprocessors = [#transport.node<backend_lib = "y", symbol = "decode">]>} {
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}

// -----

// Nor without the symbol naming what to invoke on it.

// expected-error @below {{coprocessor requires a 'symbol'}}
module attributes {catalyst.backline = #transport.backline<transport = "net", controller = #transport.node<backend_lib = "x">,
  coprocessors = [#transport.node<backend_lib = "y", peer = "10.0.0.3">]>} {
  func.func @setup() { quantum.init  return }
  func.func @teardown() { quantum.finalize  return }
}
