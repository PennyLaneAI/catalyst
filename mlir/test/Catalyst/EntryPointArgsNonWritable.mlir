// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt %s --mark-entry-point-args-non-writable -split-input-file | FileCheck %s --check-prefix=ATTR
// RUN: quantum-opt %s --mark-entry-point-args-non-writable --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -canonicalize -split-input-file | FileCheck %s --check-prefix=BUFFERIZE

// ATTR-LABEL: func.func public @entry(
// ATTR-SAME: %[[ARG0:.*]]: tensor<4xf32> {bufferization.writable = false}
// ATTR-SAME: %[[IDX:.*]]: index
func.func public @entry(%arg0: tensor<4xf32>, %arg1: index) -> tensor<4xf32> attributes {llvm.emit_c_interface} {
  return %arg0 : tensor<4xf32>
}

// ATTR-LABEL: func.func private @helper(
// ATTR-SAME: %[[ARG0:.*]]: tensor<4xf32>)
func.func private @helper(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}

// -----

// BUFFERIZE-LABEL: func.func public @entry(
// BUFFERIZE-SAME: %[[ARG0:.*]]: memref<4xf32>
// BUFFERIZE: %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// BUFFERIZE: memref.copy %[[ARG0]], %[[ALLOC]] : memref<4xf32> to memref<4xf32>
// BUFFERIZE: memref.store %{{.*}}, %[[ALLOC]][{{%.*}}] : memref<4xf32>
// BUFFERIZE: return %[[ALLOC]] : memref<4xf32>
func.func public @entry(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %0 = tensor.insert %cst into %arg0[%c0] : tensor<4xf32>
  return %0 : tensor<4xf32>
}
