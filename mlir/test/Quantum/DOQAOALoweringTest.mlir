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

// RUN: quantum-opt %s --convert-quantum-to-llvm --split-input-file | FileCheck %s

// ============================================================
// !quantum.partition<N, m>  →  !llvm.struct<(i32, i32)>
// The struct carries (numQubits: i32, m: i32).
// ============================================================

// CHECK-LABEL: llvm.func @lower_partition
// CHECK-SAME:  !llvm.struct<(i32, i32)>
func.func @lower_partition(%arg0: !quantum.partition<6, 2>) {
    return
}

// -----

// ============================================================
// !quantum.cluster_map<K>  →  !llvm.struct<(i32)>
// The struct carries (k: i32).
// ============================================================

// CHECK-LABEL: llvm.func @lower_cluster_map
// CHECK-SAME:  !llvm.struct<(i32)>
func.func @lower_cluster_map(%arg0: !quantum.cluster_map<1>) {
    return
}

// -----

// ============================================================
// !quantum.circuit_ref  →  i64
// Opaque index into the representative sub-circuit table.
// ============================================================

// CHECK-LABEL: llvm.func @lower_circuit_ref
// CHECK-SAME:  i64
func.func @lower_circuit_ref(%arg0: !quantum.circuit_ref) {
    return
}

// -----

// ============================================================
// !quantum.params  →  !llvm.ptr
// Pointer to the f64 variational parameter buffer.
// ============================================================

// CHECK-LABEL: llvm.func @lower_params
// CHECK-SAME:  !llvm.ptr
func.func @lower_params(%arg0: !quantum.params) {
    return
}

// -----

// ============================================================
// !quantum.bitstring  →  !llvm.ptr
// Pointer to the i8 binary solution buffer.
// ============================================================

// CHECK-LABEL: llvm.func @lower_bitstring
// CHECK-SAME:  !llvm.ptr
func.func @lower_bitstring(%arg0: !quantum.bitstring) {
    return
}

// -----

// ============================================================
// All 5 types in one function signature
// ============================================================

// CHECK-LABEL: llvm.func @lower_all_types
// CHECK-SAME:  !llvm.struct<(i32, i32)>
// CHECK-SAME:  !llvm.struct<(i32)>
// CHECK-SAME:  i64
// CHECK-SAME:  !llvm.ptr
// CHECK-SAME:  !llvm.ptr
func.func @lower_all_types(
        %p: !quantum.partition<10, 3>,
        %c: !quantum.cluster_map<2>,
        %r: !quantum.circuit_ref,
        %params: !quantum.params,
        %bits: !quantum.bitstring) {
    return
}
