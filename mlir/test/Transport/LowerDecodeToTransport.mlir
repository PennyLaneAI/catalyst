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

// RUN: quantum-opt --lower-decode-to-transport --split-input-file %s | FileCheck %s

// A bufferized qecp.decode_esm_css becomes a transport round to the coprocessor.

// CHECK-LABEL: func.func @qec_circuit
// CHECK-NOT:     qecp.decode_esm_css
// CHECK:         %[[S:.*]] = transport.get_session {key = "cop0"} : !transport.session<controller>
// CHECK:         transport.kick %[[S]], %{{.*}} {work_item_idx = 0 : i32} : !transport.session<controller>, memref<?xi1>
// CHECK:         transport.collect %[[S]], %{{.*}} : !transport.session<controller>, memref<?xindex>
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16>,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @qec_circuit(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<?xi1>, %erridx: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<?xindex>) : memref<?xi1>
    return
  }
}

// -----

// A backline with no coprocessors has no offload target: this module is the controller's own
// program, so the decode stays local.

// CHECK-LABEL: func.func @controller_only
// CHECK:         qecp.decode_esm_css
// CHECK-NOT:     {{[^#]}}transport.
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16>>} {
  func.func @controller_only(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<?xi1>, %erridx: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<?xindex>) : memref<?xi1>
    return
  }
}

// -----

// Without catalyst.backline, the decode is left untouched (no transport offload).

// CHECK-LABEL: func.func @plain
// CHECK:         qecp.decode_esm_css
// CHECK-NOT:     {{[^#]}}transport.
module {
  func.func @plain(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<?xi1>, %erridx: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<?xindex>) : memref<?xi1>
    return
  }
}

// -----

// A decode that is not bufferized yet has no buffers to transport, so it is left untouched even
// with a coprocessor to offload to.

// CHECK-LABEL: func.func @not_bufferized
// CHECK:         qecp.decode_esm_css
// CHECK-NOT:     {{[^#]}}transport.
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c">,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @not_bufferized(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: tensor<2xi1>) -> tensor<1xindex> {
    %erridx = qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm : tensor<2xi1> -> tensor<1xindex>
    return %erridx : tensor<1xindex>
  }
}

// -----

// Multiple coprocessors: decodes round-robin across the peers by key (cop0, then cop1).

// CHECK-LABEL: func.func @qec_circuit
// CHECK:         transport.get_session {key = "cop0"} : !transport.session<controller>
// CHECK:         transport.get_session {key = "cop1"} : !transport.session<controller>
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c">,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">,
                  #transport.node<backend_lib = "x", config = "c", name = "cop1", peer = "10.0.0.2", symbol = "decode">]>} {
  func.func @qec_circuit(%tanner: !qecp.tanner_graph<8, 6, i32>,
                         %e0: memref<?xi1>, %i0: memref<?xindex>,
                         %e1: memref<?xi1>, %i1: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %e0 in (%i0 : memref<?xindex>) : memref<?xi1>
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %e1 in (%i1 : memref<?xindex>) : memref<?xi1>
    return
  }
}

// -----

// Check decode check_type mapping to decoder_id

// CHECK-LABEL: func.func @qec_cycle_css
// CHECK:         %[[SX:.*]] = transport.get_session {key = "cop0"} : !transport.session<controller>
// CHECK:         transport.kick %[[SX]], %{{.*}} {work_item_idx = 0 : i32}
// CHECK:         %[[SZ:.*]] = transport.get_session {key = "cop0"} : !transport.session<controller>
// CHECK:         transport.kick %[[SZ]], %{{.*}} {decoder_id = 1 : i32, work_item_idx = 0 : i32}
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c">,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @qec_cycle_css(%tannerX: !qecp.tanner_graph<8, 6, i32>,
                           %tannerZ: !qecp.tanner_graph<8, 6, i32>,
                           %esmX: memref<?xi1>, %iX: memref<?xindex>,
                           %esmZ: memref<?xi1>, %iZ: memref<?xindex>) {
    qecp.decode_esm_css(%tannerX : !qecp.tanner_graph<8, 6, i32>) %esmX in (%iX : memref<?xindex>) {check_type = "x"} : memref<?xi1>
    qecp.decode_esm_css(%tannerZ : !qecp.tanner_graph<8, 6, i32>) %esmZ in (%iZ : memref<?xindex>) {check_type = "z"} : memref<?xi1>
    return
  }
}

// -----

// A decode that declares no check family takes decoder 0, so IR from before the attribute
// existed lowers unchanged.

// CHECK-LABEL: func.func @untagged
// CHECK:         transport.kick %{{.*}} {work_item_idx = 0 : i32}
// CHECK-NOT:     decoder_id
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c">,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @untagged(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<?xi1>, %erridx: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<?xindex>) : memref<?xi1>
    return
  }
}
