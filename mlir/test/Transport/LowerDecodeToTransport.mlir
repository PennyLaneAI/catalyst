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

// A bufferized qecp.decode_esm_css becomes a transport round to the coprocessor: the syndrome is
// staged and posted, and the correction is collected in the session's reply slot.

// CHECK-LABEL: func.func @qec_circuit
// CHECK-NOT:     qecp.decode_esm_css
// CHECK-NOT:     memref.alloc
// CHECK:         %[[S:.*]] = transport.get_session {key = "cop0"} : !transport.session<controller>
// CHECK:         transport.stage_payload %[[S]], %{{.*}} : !transport.session<controller>, memref<3xi1>
// CHECK:         transport.post %[[S]] : !transport.session<controller>
// CHECK:         %[[REP:.*]] = transport.reply_slot %[[S]] : !transport.session<controller> -> memref<1xindex>
// CHECK:         transport.collect %[[S]], %[[REP]] : !transport.session<controller>, memref<1xindex>
// CHECK:         memref.load %[[REP]]
// CHECK-NOT:     memref.dealloc
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16>,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @qec_circuit(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<3xi1>) -> index {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1xindex>
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%alloc : memref<1xindex>) : memref<3xi1>
    %v = memref.load %alloc[%c0] : memref<1xindex>
    memref.dealloc %alloc : memref<1xindex>
    return %v : index
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

// A correction the caller passed in is an out-parameter: the caller expects to find the reply in
// its own buffer, so the round collects there rather than in the reply slot.

// CHECK-LABEL: func.func @qec_circuit_out_param
// CHECK-NOT:     transport.reply_slot
// CHECK:         transport.collect %{{.*}}, %arg2 : !transport.session<controller>, memref<1xindex>
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c">,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @qec_circuit_out_param(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<3xi1>,
                                   %erridx: memref<1xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<1xindex>) : memref<3xi1>
    return
  }
}

// -----

// A dynamically shaped correction gives the reply slot no size to take, so the round falls back to
// the buffer the caller supplied and collect writes into that.

// CHECK-LABEL: func.func @qec_circuit_dynamic
// CHECK-NOT:     transport.reply_slot
// CHECK:         transport.collect %{{.*}}, %{{.*}} : !transport.session<controller>, memref<?xindex>
module attributes {catalyst.backline = #transport.backline<transport = "net",
  controller = #transport.node<backend_lib = "x", config = "c">,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @qec_circuit_dynamic(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<?xi1>, %erridx: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<?xindex>) : memref<?xi1>
    return
  }
}
