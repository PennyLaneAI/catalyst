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
//
// RUN: quantum-opt -resource-analysis=output-json --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s

// A million independent local ancillas do not need a million persistent clocks
// in the caller. Their completed paths are already represented by callee depth.
// CHECK-LABEL: "independent_local_ancillas": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @independent_local_ancillas() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count = arith.constant 1024 : index
    scf.for %i = %zero to %count step %one {
        func.call @z_independent_helper() : () -> ()
    }
    return
}
func.func private @z_independent_helper() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count = arith.constant 1024 : index
    scf.for %i = %zero to %count step %one {
        %ancilla = quantum.alloc_qb : !quantum.bit
        %out = pbc.ppr ["X"](2) %ancilla : !quantum.bit
        quantum.dealloc_qb %out : !quantum.bit
    }
    return
}

// -----

// Discarding internal clocks must retain every dependency carried by the data
// qubit. Its million interactions remain a serial path.
// CHECK-LABEL: "shared_data_local_ancillas": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1048576
func.func @shared_data_local_ancillas(%data: !quantum.bit) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count = arith.constant 1024 : index
    %out = scf.for %i = %zero to %count step %one iter_args(%q = %data) -> (!quantum.bit) {
        %next = func.call @z_shared_helper(%q) : (!quantum.bit) -> !quantum.bit
        scf.yield %next : !quantum.bit
    }
    return
}
func.func private @z_shared_helper(%data: !quantum.bit) -> !quantum.bit {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count = arith.constant 1024 : index
    %out = scf.for %i = %zero to %count step %one iter_args(%q = %data) -> (!quantum.bit) {
        %ancilla = quantum.alloc_qb : !quantum.bit
        %pair:2 = pbc.ppr ["X", "Z"](2) %q, %ancilla : !quantum.bit, !quantum.bit
        quantum.dealloc_qb %pair#1 : !quantum.bit
        scf.yield %pair#0 : !quantum.bit
    }
    return %out : !quantum.bit
}

// -----

// Inserting a register's own qubit only materializes its canonical mapping.
// Reusing the call summary must preserve the same physical qubit identities.
// CHECK-LABEL: "canonical_register_materialization": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @canonical_register_materialization() {
    %reg = quantum.alloc(2) : !quantum.reg
    %first = func.call @z_rotate_zero(%reg) : (!quantum.reg) -> !quantum.reg
    %q1 = quantum.extract %first[1] : !quantum.reg -> !quantum.bit
    %materialized = quantum.insert %first[1], %q1 : !quantum.reg, !quantum.bit
    %second = func.call @z_rotate_zero(%materialized) : (!quantum.reg) -> !quantum.reg
    return
}
func.func private @z_rotate_zero(%reg: !quantum.reg) -> !quantum.reg {
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["X"](2) %q : !quantum.bit
    %updated = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
    return %updated : !quantum.reg
}

// -----

// Swapping register positions changes which physical qubit the same helper uses.
// These two calls act on distinct qubits and must fit in one layer.
// CHECK-LABEL: "swapped_register_cache_key": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @swapped_register_cache_key() {
    %reg = quantum.alloc(2) : !quantum.reg
    %first = func.call @z_rotate_zero(%reg) : (!quantum.reg) -> !quantum.reg
    %q0 = quantum.extract %first[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %first[1] : !quantum.reg -> !quantum.bit
    %half = quantum.insert %first[0], %q1 : !quantum.reg, !quantum.bit
    %swapped = quantum.insert %half[1], %q0 : !quantum.reg, !quantum.bit
    %second = func.call @z_rotate_zero(%swapped) : (!quantum.reg) -> !quantum.reg
    return
}
func.func private @z_rotate_zero(%reg: !quantum.reg) -> !quantum.reg {
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["X"](2) %q : !quantum.bit
    %updated = quantum.insert %reg[0], %out : !quantum.reg, !quantum.bit
    return %updated : !quantum.reg
}

// -----

// A noncanonical alias makes the second helper's two operations sequential.
// It must not reuse the summary for the initially disjoint register.
// CHECK-LABEL: "aliased_register_cache_key": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 3
func.func @aliased_register_cache_key() {
    %reg = quantum.alloc(2) : !quantum.reg
    %first = func.call @z_rotate_pair(%reg) : (!quantum.reg) -> !quantum.reg
    %q0 = quantum.extract %first[0] : !quantum.reg -> !quantum.bit
    %aliased = quantum.insert %first[1], %q0 : !quantum.reg, !quantum.bit
    %second = func.call @z_rotate_pair(%aliased) : (!quantum.reg) -> !quantum.reg
    return
}
func.func private @z_rotate_pair(%reg: !quantum.reg) -> !quantum.reg {
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %out0 = pbc.ppr ["X"](2) %q0 : !quantum.bit
    %updated0 = quantum.insert %reg[0], %out0 : !quantum.reg, !quantum.bit
    %q1 = quantum.extract %updated0[1] : !quantum.reg -> !quantum.bit
    %out1 = pbc.ppr ["X"](2) %q1 : !quantum.bit
    %updated1 = quantum.insert %updated0[1], %out1 : !quantum.reg, !quantum.bit
    return %updated1 : !quantum.reg
}
