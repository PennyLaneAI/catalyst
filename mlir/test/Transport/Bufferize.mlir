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

// CHECK-LABEL: func.func @round
// CHECK:         transport.kick %{{.*}}, %{{.*}} {work_item_idx = 0 : i32} : !transport.session<controller>, memref<8xi8>
// CHECK:         %[[A:.*]] = memref.alloc() : memref<8xi8>
// CHECK:         transport.collect %{{.*}}, %[[A]] : !transport.session<controller>, memref<8xi8>
func.func @round(%payload: tensor<8xi8>) -> tensor<8xi8> {
  %s = transport.get_session : !transport.session<controller>
  transport.kick %s, %payload {work_item_idx = 0 : i32} : !transport.session<controller>, tensor<8xi8>
  %corr = transport.collect %s : !transport.session<controller> -> tensor<8xi8>
  return %corr : tensor<8xi8>
}
