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

// RUN: quantum-opt -resource-analysis=output-json --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s

// The same callee has separate summaries for distinct physical arguments.
// Reusing a summary still starts at the current physical-qubit clock.
// CHECK-LABEL: "cached_qubit_arguments": {
// CHECK: "qubit_disjoint_depth_total": 4
func.func @cached_qubit_arguments(%q0: !quantum.bit, %q1: !quantum.bit) {
    %a = func.call @two_rotations(%q0) : (!quantum.bit) -> !quantum.bit
    %b = func.call @two_rotations(%q1) : (!quantum.bit) -> !quantum.bit
    %c = func.call @two_rotations(%a) : (!quantum.bit) -> !quantum.bit
    %d = func.call @two_rotations(%b) : (!quantum.bit) -> !quantum.bit
    return
}

func.func private @two_rotations(%q: !quantum.bit) -> !quantum.bit {
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %b = pbc.ppr ["Z"](8) %a : !quantum.bit
    return %b : !quantum.bit
}

// -----

// A permutation of a register's slots must change the cache key.
// The two helper calls act on different physical qubits.
// CHECK-LABEL: "cached_register_permutation": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @cached_register_permutation() {
    %reg = quantum.alloc(2) : !quantum.reg
    %r0 = func.call @slot_zero_rotations(%reg) : (!quantum.reg) -> !quantum.reg
    %q0 = quantum.extract %r0[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r0[1] : !quantum.reg -> !quantum.bit
    %r1 = quantum.insert %r0[0], %q1 : !quantum.reg, !quantum.bit
    %r2 = quantum.insert %r1[1], %q0 : !quantum.reg, !quantum.bit
    %r3 = func.call @slot_zero_rotations(%r2) : (!quantum.reg) -> !quantum.reg
    quantum.dealloc %r3 : !quantum.reg
    return
}

func.func private @slot_zero_rotations(%reg: !quantum.reg) -> !quantum.reg {
    %q = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %b = pbc.ppr ["Z"](8) %a : !quantum.bit
    %out = quantum.insert %reg[0], %b : !quantum.reg, !quantum.bit
    return %out : !quantum.reg
}

// -----

// Aliasing two formal registers changes both the footprint and the duration.
// CHECK-LABEL: "cached_register_aliases": {
// CHECK: "qubit_disjoint_depth_total": 3
func.func @cached_register_aliases() {
    %reg0 = quantum.alloc(1) : !quantum.reg
    %reg1 = quantum.alloc(1) : !quantum.reg
    %a:2 = func.call @two_register_rotations(%reg0, %reg1) : (!quantum.reg, !quantum.reg) -> (!quantum.reg, !quantum.reg)
    %b:2 = func.call @two_register_rotations(%a#0, %a#0) : (!quantum.reg, !quantum.reg) -> (!quantum.reg, !quantum.reg)
    %c:2 = func.call @two_register_rotations(%a#1, %a#1) : (!quantum.reg, !quantum.reg) -> (!quantum.reg, !quantum.reg)
    quantum.dealloc %b#1 : !quantum.reg
    quantum.dealloc %c#1 : !quantum.reg
    return
}

func.func private @two_register_rotations(%reg0: !quantum.reg, %reg1: !quantum.reg) -> (!quantum.reg, !quantum.reg) {
    %q0 = quantum.extract %reg0[0] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %r0 = quantum.insert %reg0[0], %a : !quantum.reg, !quantum.bit
    %q1 = quantum.extract %reg1[0] : !quantum.reg -> !quantum.bit
    %b = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    %r1 = quantum.insert %reg1[0], %b : !quantum.reg, !quantum.bit
    return %r0, %r1 : !quantum.reg, !quantum.reg
}

// -----

// Cached calls keep the qubit-only cost when their classical arguments change.
// CHECK-LABEL: "cached_classical_readiness": {
// CHECK: "qubit_disjoint_depth_total": 4
func.func @cached_classical_readiness(%q0: !quantum.bit, %q1: !quantum.bit, %condition: i1) {
    %a = func.call @conditional_rotation(%q0, %condition) : (!quantum.bit, i1) -> !quantum.bit
    %b0 = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    %b2 = pbc.ppr ["X"](8) %b1 : !quantum.bit
    %m, %out = pbc.ppm ["Z"] %b2 : i1, !quantum.bit
    %c = func.call @conditional_rotation(%a, %m) : (!quantum.bit, i1) -> !quantum.bit
    return
}

func.func private @conditional_rotation(%q: !quantum.bit, %condition: i1) -> !quantum.bit {
    %out = pbc.ppr ["X"](2) %q cond(%condition) : !quantum.bit
    return %out : !quantum.bit
}

// -----

// Classical memory does not synchronize disjoint qubits across cached calls.
// CHECK-LABEL: "cached_memory_readiness": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @cached_memory_readiness(%q0: !quantum.bit, %q1: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %memory = memref.alloc() : memref<1xi1>
    %a = func.call @store_measurement(%q0, %memory) : (!quantum.bit, memref<1xi1>) -> !quantum.bit
    %b = func.call @store_measurement(%a, %memory) : (!quantum.bit, memref<1xi1>) -> !quantum.bit
    %condition = memref.load %memory[%c0] : memref<1xi1>
    %out = pbc.ppr ["X"](2) %q1 cond(%condition) : !quantum.bit
    memref.dealloc %memory : memref<1xi1>
    return
}

func.func private @store_measurement(%q: !quantum.bit, %memory: memref<1xi1>) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %m, %out = pbc.ppm ["Z"] %q : i1, !quantum.bit
    memref.store %m, %memory[%c0] : memref<1xi1>
    return %out : !quantum.bit
}

// -----

// Fresh allocations returned by separate calls have distinct physical identities.
// CHECK-LABEL: "cached_allocated_outputs": {
// CHECK: "qubit_disjoint_depth_total": 1
func.func @cached_allocated_outputs() {
    %a = func.call @allocate_rotated_qubit() : () -> !quantum.bit
    %b = func.call @allocate_rotated_qubit() : () -> !quantum.bit
    quantum.dealloc_qb %a : !quantum.bit
    quantum.dealloc_qb %b : !quantum.bit
    return
}

func.func private @allocate_rotated_qubit() -> !quantum.bit {
    %q = quantum.alloc_qb : !quantum.bit
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    return %out : !quantum.bit
}

// -----

// Changing an unused classical argument does not change a helper's transfer.
// Even an unknown value can reuse it; all six gates still share the same wire.
// CHECK-LABEL: "unused_classical_arguments": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 6
func.func @unused_classical_arguments(%q: !quantum.bit, %unknown: i64) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %a = func.call @ignored_classical_argument(%q, %zero) : (!quantum.bit, i64) -> !quantum.bit
    %b = func.call @ignored_classical_argument(%a, %one) : (!quantum.bit, i64) -> !quantum.bit
    %c = func.call @ignored_classical_argument(%b, %unknown) : (!quantum.bit, i64) -> !quantum.bit
    return
}

func.func private @ignored_classical_argument(%q: !quantum.bit, %unused: i64) -> !quantum.bit {
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %b = pbc.ppr ["Z"](8) %a : !quantum.bit
    return %b : !quantum.bit
}

// -----

// A classical argument used as a wire index must still distinguish summaries.
// Reusing the first call's summary would incorrectly serialize both rotations.
// CHECK-LABEL: "used_classical_wire_arguments": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @used_classical_wire_arguments() {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %reg = quantum.alloc(2) : !quantum.reg
    %a = func.call @indexed_rotation(%reg, %zero) : (!quantum.reg, i64) -> !quantum.reg
    %b = func.call @indexed_rotation(%a, %one) : (!quantum.reg, i64) -> !quantum.reg
    quantum.dealloc %b : !quantum.reg
    return
}

func.func private @indexed_rotation(%reg: !quantum.reg, %index: i64) -> !quantum.reg {
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %rotated = pbc.ppr ["X"](8) %q : !quantum.bit
    %out = quantum.insert %reg[%index], %rotated : !quantum.reg, !quantum.bit
    return %out : !quantum.reg
}

// -----

// A classical loop bound also remains part of the key. The same physical wire
// receives two gates in the first call and three in the second, for five layers.
// CHECK-LABEL: "used_classical_depth_arguments": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 5
func.func @used_classical_depth_arguments(%q: !quantum.bit) {
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %a = func.call @repeated_rotation(%q, %two) : (!quantum.bit, index) -> !quantum.bit
    %b = func.call @repeated_rotation(%a, %three) : (!quantum.bit, index) -> !quantum.bit
    return
}

func.func private @repeated_rotation(%q: !quantum.bit, %count: index) -> !quantum.bit {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %out = scf.for %i = %zero to %count step %one iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return %out : !quantum.bit
}
