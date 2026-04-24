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

// RUN: quantum-opt %s --memspace-to-address-space --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @explicit_addr_spaces
// CHECK-SAME:    (%arg0: memref<4xi8>
// CHECK-SAME:     %arg1: memref<4xi8, 1>
func.func @explicit_addr_spaces(
    %arg0: memref<4xi8,  #catalyst.memspace<"d0", 0 : i64>>,
    %arg1: memref<4xi8,  #catalyst.memspace<"d0", 1 : i64>>
) {
    return
}

// -----

// CHECK-LABEL: func.func @alloc_load_store
// CHECK:       %[[BUF:.*]] = memref.alloc() : memref<4xi32, 1>
// CHECK:       memref.store %{{.*}}, %[[BUF]][%{{.*}}] : memref<4xi32, 1>
// CHECK:       %{{.*}} = memref.load %[[BUF]][%{{.*}}] : memref<4xi32, 1>
// CHECK:       memref.dealloc %[[BUF]] : memref<4xi32, 1>
func.func @alloc_load_store() -> i32 {
    %c0 = arith.constant 0 : index
    %v  = arith.constant 42 : i32
    %buf = memref.alloc() : memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
    memref.store %v, %buf[%c0] : memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
    %out = memref.load %buf[%c0] : memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
    memref.dealloc %buf : memref<4xi32, #catalyst.memspace<"d0", 1 : i64>>
    return %out : i32
}

// -----

// CHECK-LABEL: func.func @discardable_attr
// CHECK-SAME:    attributes {catalyst.memspace = 1 : i64}
func.func @discardable_attr() attributes {
    catalyst.memspace = #catalyst.memspace<"d0", 1 : i64>
} {
    return
}
