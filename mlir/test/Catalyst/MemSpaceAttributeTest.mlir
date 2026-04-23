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

// RUN: quantum-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @memspace_in_memref
// CHECK-SAME:    (%arg0: memref<4x64xi8, #catalyst.memspace<"d0">>
// CHECK-SAME:     %arg1: memref<1xi32, #catalyst.memspace<"d1", 1 : i64>>)
func.func @memspace_in_memref(
    %arg0: memref<4x64xi8, #catalyst.memspace<"d0">>,
    %arg1: memref<1xi32,   #catalyst.memspace<"d1", 1 : i64>>
) {
    return
}

// -----

// CHECK-LABEL: func.func @memspace_on_op()
// CHECK-SAME:    attributes {catalyst.memspace = #catalyst.memspace<"d0", 3 : i64>}
func.func @memspace_on_op() attributes {
    catalyst.memspace = #catalyst.memspace<"d0", 3 : i64>
} {
    return
}

// -----

func.func @bad_domain(
    // expected-error@+1 {{catalyst.memspace: `domain` must be non-empty}}
    %arg0: memref<4xi8, #catalyst.memspace<"">>
) {
    return
}

// -----

func.func @bad_addr_space(
    // expected-error@+1 {{catalyst.memspace: addr_space must be non-negative}}
    %arg0: memref<4xi8, #catalyst.memspace<"d0", -1 : i64>>
) {
    return
}
