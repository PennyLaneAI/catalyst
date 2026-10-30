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

// RUN: quantum-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:     one-shot-bufferize{unknown-type-conversion=identity-layout-map} \
// RUN:   )" %s | FileCheck %s

// CHECK-LABEL: func.func @host
// CHECK: %[[ARG:.*]] = bufferization.to_buffer %{{.*}} : tensor<4xf64> to memref<4xf64>
// CHECK: %[[RES:.*]] = catalyst.launch_kernel @target::@entry(%[[ARG]]) : (memref<4xf64>) -> memref<4xf64>
// CHECK: bufferization.to_tensor %[[RES]]
func.func @host(%arg0: tensor<4xf64>) -> tensor<4xf64> {
  %0 = catalyst.launch_kernel @target::@entry(%arg0) : (tensor<4xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

module @target {
  func.func public @entry(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    return %arg0 : tensor<4xf64>
  }
}

// -----

// A non-identity-layout (strided/offset) operand must be copied into a fresh contiguous buffer
// before the call.

// CHECK-LABEL: func.func @host_strided
// CHECK: %[[SUB:.*]] = memref.subview %{{.*}} : memref<8xf64> to memref<4xf64, strided<[1], offset: 2>>
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xf64>
// CHECK: memref.copy %[[SUB]], %[[ALLOC]]
// CHECK: catalyst.launch_kernel @target::@entry(%[[ALLOC]]) : (memref<4xf64>) -> memref<4xf64>
func.func @host_strided(%arg0: tensor<8xf64>) -> tensor<4xf64> {
  %slice = tensor.extract_slice %arg0[2] [4] [1] : tensor<8xf64> to tensor<4xf64>
  %0 = catalyst.launch_kernel @target::@entry(%slice) : (tensor<4xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

module @target {
  func.func public @entry(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    return %arg0 : tensor<4xf64>
  }
}
