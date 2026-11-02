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

// A callee may return a locally allocated qubit. Cached allocations are shared
// conservatively, retaining dependencies instead of inventing parallel wires.
// CHECK-LABEL: "cached_allocated_outputs": {
// CHECK: "qubit_disjoint_depth_total": 2
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
