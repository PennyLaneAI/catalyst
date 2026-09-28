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

// RUN: quantum-opt --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s

// A payload whose bufferized form has a non-identity layout (here, a strided
// function argument) is copied into a fresh contiguous buffer before the kick,
// since the transport backend reads it as a contiguous block.
// CHECK-LABEL: func.func @round(
// CHECK-SAME:      %[[ARG:.*]]: memref<8xi8, strided<[?], offset: ?>>
// CHECK:         %[[COPY:.*]] = memref.alloc() : memref<8xi8>
// CHECK:         memref.copy %[[ARG]], %[[COPY]]
// CHECK:         transport.stage_payload %{{.*}}, %[[COPY]] : !transport.session<controller>, memref<8xi8>
// CHECK:         %[[DEST:.*]] = memref.alloc() {{.*}}: memref<8xi8>
// CHECK:         transport.collect %{{.*}}, %[[DEST]] : !transport.session<controller>, memref<8xi8>
func.func @round(%payload: tensor<8xi8>) -> tensor<8xi8> {
  %s = transport.get_session : !transport.session<controller>
  transport.stage_payload %s, %payload : !transport.session<controller>, tensor<8xi8>
  %corr = transport.collect %s : !transport.session<controller> -> tensor<8xi8>
  return %corr : tensor<8xi8>
}

// A payload that already bufferizes to an identity layout is passed to the kick
// directly, with no intervening copy.
// CHECK-LABEL: func.func @round_contiguous(
// CHECK:         %[[P:.*]] = memref.alloc() {{.*}}: memref<8xi8>
// CHECK:         linalg.fill ins(%{{.*}} : i8) outs(%[[P]] : memref<8xi8>)
// CHECK-NOT:     memref.copy
// CHECK:         transport.stage_payload %{{.*}}, %[[P]] : !transport.session<controller>, memref<8xi8>
func.func @round_contiguous(%v: i8) -> tensor<8xi8> {
  %s = transport.get_session : !transport.session<controller>
  %e = tensor.empty() : tensor<8xi8>
  %payload = linalg.fill ins(%v : i8) outs(%e : tensor<8xi8>) -> tensor<8xi8>
  transport.stage_payload %s, %payload : !transport.session<controller>, tensor<8xi8>
  %corr = transport.collect %s : !transport.session<controller> -> tensor<8xi8>
  return %corr : tensor<8xi8>
}
