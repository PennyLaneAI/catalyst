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
// RUN:     one-shot-bufferize{bufferize-function-boundaries \
// RUN:                         function-boundary-type-conversion=identity-layout-map} \
// RUN:   )" %s | FileCheck %s

// CHECK-LABEL: func.func @memspace_on_alloc_tensor_survives
// CHECK:         memref.alloc() {{.*}}: memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
func.func @memspace_on_alloc_tensor_survives() -> tensor<4xi32> {
    %t = bufferization.alloc_tensor() {memory_space = #catalyst.memspace<"d0", 1 : i64>} : tensor<4xi32>
    return %t : tensor<4xi32>
}

// -----

// CHECK-LABEL: func.func @memspace_with_store_load
// CHECK:         %[[BUF:.*]] = memref.alloc() {{.*}}: memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
// CHECK:         memref.store %{{.*}}, %[[BUF]][%{{.*}}] : memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
// CHECK:         memref.load %[[BUF]][%{{.*}}] : memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
func.func @memspace_with_store_load() -> i32 {
    %c0 = arith.constant 0 : index
    %v  = arith.constant 42 : i32
    %t  = bufferization.alloc_tensor() {memory_space = #catalyst.memspace<"d0", 1 : i64>} : tensor<4xi32>
    %t2 = tensor.insert %v into %t[%c0] : tensor<4xi32>
    %r  = tensor.extract %t2[%c0] : tensor<4xi32>
    return %r : i32
}

// -----

// CHECK-LABEL: func.func @two_memspaces_distinct
// CHECK-DAG:     memref.alloc() {{.*}}: memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
// CHECK-DAG:     memref.alloc() {{.*}}: memref<4xi32, #catalyst.memspace<"d1", 2 : i64>>
func.func @two_memspaces_distinct() -> (tensor<4xi32>, tensor<4xi32>) {
    %a = bufferization.alloc_tensor() {memory_space = #catalyst.memspace<"d0", 1 : i64>} : tensor<4xi32>
    %b = bufferization.alloc_tensor() {memory_space = #catalyst.memspace<"d1", 2 : i64>} : tensor<4xi32>
    return %a, %b : tensor<4xi32>, tensor<4xi32>
}
