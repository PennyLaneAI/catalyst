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

// RUN: quantum-opt %s --lower-decode-to-transport --split-input-file --verify-diagnostics

// A syndrome is staged one byte per bit, so a 24-bit syndrome is a 24-byte request, which does not
// fit the 8 bytes the controller commits.

module attributes {catalyst.backline = #transport.backline<transport = "rdma",
  controller = #transport.node<backend_lib = "x", config = "c", in_bytes = 8 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @oversize_syndrome(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<24xi1>, %erridx: memref<1xindex>) {
    // expected-error @below {{syndrome of 24 bytes exceeds the backline controller's in_bytes of 8 bytes}}
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%erridx : memref<1xindex>) : memref<24xi1>
    return
  }
}

// -----

// A correction of two indices is a 16-byte reply, which does not fit the 8 bytes the controller
// commits.

module attributes {catalyst.backline = #transport.backline<transport = "rdma",
  controller = #transport.node<backend_lib = "x", config = "c", in_bytes = 8 : i64, out_bytes = 8 : i64>,
  coprocessors = [#transport.node<backend_lib = "x", config = "c", name = "cop0", peer = "10.0.0.1", symbol = "decode">]>} {
  func.func @oversize_correction(%tanner: !qecp.tanner_graph<8, 6, i32>, %esm: memref<3xi1>) -> index {
    %c1 = arith.constant 1 : index
    %buf = memref.alloc() : memref<2xindex>
    // expected-error @below {{correction of 16 bytes exceeds the backline controller's out_bytes of 8 bytes}}
    qecp.decode_esm_css(%tanner : !qecp.tanner_graph<8, 6, i32>) %esm in (%buf : memref<2xindex>) : memref<3xi1>
    %v = memref.load %buf[%c1] : memref<2xindex>
    memref.dealloc %buf : memref<2xindex>
    return %v : index
  }
}
