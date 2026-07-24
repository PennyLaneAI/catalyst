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

// A bufferized qecp.decode_esm_css becomes a transport round.

// CHECK-LABEL: func.func @qec_circuit
// CHECK-NOT:     qecp.decode_esm_css
// CHECK:         %[[S:.*]] = transport.get_session {key = "controller"} : !transport.session<controller>
// CHECK:         transport.kick %[[S]], %{{.*}} {work_item_idx = 0 : i32} : !transport.session<controller>, memref<?xi1>
// CHECK:         transport.collect %[[S]], %{{.*}} : !transport.session<controller>, memref<?xindex>
module attributes {catalyst.backline = {
  controller = {backend_lib = "x", config = "c", peer = "127.0.0.1", oob_port = 18590 : i16}
}} {
  func.func @qec_circuit(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<?xi1>, %erridx: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<?xindex>) : memref<?xi1>
    return
  }
}

// -----

// Without catalyst.backline, the decode is left untouched (no transport offload).

// CHECK-LABEL: func.func @plain
// CHECK:         qecp.decode_esm_css
// CHECK-NOT:     transport.
module {
  func.func @plain(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<?xi1>, %erridx: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<?xindex>) : memref<?xi1>
    return
  }
}

// -----

// Multiple coprocessors: decodes round-robin across the peers by key (cop0, then cop1).

// CHECK-LABEL: func.func @qec_circuit
// CHECK:         transport.get_session {key = "cop0"} : !transport.session<controller>
// CHECK:         transport.get_session {key = "cop1"} : !transport.session<controller>
module attributes {catalyst.backline = {
  controller = {backend_lib = "x", config = "c"},
  coprocessors = [
    {backend_lib = "x", config = "c", name = "cop0"},
    {backend_lib = "x", config = "c", name = "cop1"}
  ]
}} {
  func.func @qec_circuit(%tanner: !qecp.tanner_graph<8, 6, i32>,
                         %e0: memref<?xi1>, %i0: memref<?xindex>,
                         %e1: memref<?xi1>, %i1: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %e0 in (%i0 : memref<?xindex>) : memref<?xi1>
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %e1 in (%i1 : memref<?xindex>) : memref<?xi1>
    return
  }
}

// -----

// An explicit `transport.peer` tag overrides round-robin: this decode routes to cop1.

// CHECK-LABEL: func.func @qec_circuit
// CHECK:         transport.get_session {key = "cop1"} : !transport.session<controller>
module attributes {catalyst.backline = {
  controller = {backend_lib = "x", config = "c"},
  coprocessors = [
    {backend_lib = "x", config = "c", name = "cop0"},
    {backend_lib = "x", config = "c", name = "cop1"}
  ]
}} {
  func.func @qec_circuit(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<?xi1>, %erridx: memref<?xindex>) {
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<?xindex>) {transport.peer = "cop1"} : memref<?xi1>
    return
  }
}
