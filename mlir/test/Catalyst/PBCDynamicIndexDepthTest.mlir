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

// An unresolved slot leaves the exact overlap graph unknown.
// CHECK-LABEL: "dynamic_serializes_static_slots": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @dynamic_serializes_static_slots(%index: i64) {
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %r0 = quantum.insert %reg[0], %a1 : !quantum.reg, !quantum.bit
    %qd = quantum.extract %r0[%index] : !quantum.reg -> !quantum.bit
    %b0 = pbc.ppr ["X"](8) %qd : !quantum.bit
    %rd = quantum.insert %r0[%index], %b0 : !quantum.reg, !quantum.bit
    %q1 = quantum.extract %rd[1] : !quantum.reg -> !quantum.bit
    %c0 = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %c1 = pbc.ppr ["Z"](8) %c0 : !quantum.bit
    %r1 = quantum.insert %rd[1], %c1 : !quantum.reg, !quantum.bit
    quantum.dealloc %r1 : !quantum.reg
    return
}

// -----

// Even separate registers require a resolved graph for this exact analysis.
// CHECK-LABEL: "independent_dynamic_registers": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @independent_dynamic_registers(%index: i64) {
    %reg0 = quantum.alloc(2) : !quantum.reg
    %reg1 = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg0[%index] : !quantum.reg -> !quantum.bit
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %r0 = quantum.insert %reg0[%index], %a1 : !quantum.reg, !quantum.bit
    %q1 = quantum.extract %reg1[%index] : !quantum.reg -> !quantum.bit
    %b0 = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    %r1 = quantum.insert %reg1[%index], %b1 : !quantum.reg, !quantum.bit
    quantum.dealloc %r0 : !quantum.reg
    quantum.dealloc %r1 : !quantum.reg
    return
}

// -----

// A bounded static loop resolves each induction-dependent slot independently.
// CHECK-LABEL: "dynamic_register_loop": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @dynamic_register_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %reg = quantum.alloc(4) : !quantum.reg
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%r = %reg) -> (!quantum.reg) {
        %index = arith.index_cast %i : index to i64
        %q = quantum.extract %r[%index] : !quantum.reg -> !quantum.bit
        %out = pbc.ppr ["X"](8) %q : !quantum.bit
        %rout = quantum.insert %r[%index], %out : !quantum.reg, !quantum.bit
        scf.yield %rout : !quantum.reg
    }
    %q0 = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["Z"](8) %q0 : !quantum.bit
    %rout = quantum.insert %r[0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %rout : !quantum.reg
    return
}

// -----

// Carrying an unresolved extraction through a loop does not resolve its identity.
// CHECK-LABEL: "dynamic_qubit_loop": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @dynamic_qubit_loop(%index: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %reg = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %out = scf.for %i = %c0 to %c3 step %c1 iter_args(%carried = %q) -> (!quantum.bit) {
        %next = pbc.ppr ["X"](8) %carried : !quantum.bit
        scf.yield %next : !quantum.bit
    }
    %rout = quantum.insert %reg[%index], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %rout : !quantum.reg
    return
}

// -----

// Register sizes supplied by a caller do not determine unresolved indices.
// CHECK-LABEL: "dynamic_register_calls": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
// CHECK-LABEL: "dynamic_register_helper": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @dynamic_register_calls(%index: i64) {
    %reg0 = quantum.alloc(2) : !quantum.reg
    %reg1 = quantum.alloc(2) : !quantum.reg
    %a = func.call @dynamic_register_helper(%reg0, %index) : (!quantum.reg, i64) -> !quantum.reg
    %b = func.call @dynamic_register_helper(%a, %index) : (!quantum.reg, i64) -> !quantum.reg
    %c = func.call @dynamic_register_helper(%reg1, %index) : (!quantum.reg, i64) -> !quantum.reg
    quantum.dealloc %b : !quantum.reg
    quantum.dealloc %c : !quantum.reg
    return
}

func.func private @dynamic_register_helper(%reg: !quantum.reg, %index: i64) -> !quantum.reg {
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %b = pbc.ppr ["Z"](8) %a : !quantum.bit
    %rout = quantum.insert %reg[%index], %b : !quantum.reg, !quantum.bit
    return %rout : !quantum.reg
}

// -----

// Measurement-dependent indices leave the exact overlap graph unavailable.
// CHECK-LABEL: "measurement_dependent_dynamic_index": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @measurement_dependent_dynamic_index(%q0: !quantum.bit) {
    %reg = quantum.alloc(2) : !quantum.reg
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %m, %out = pbc.ppm ["Z"] %a2 : i1, !quantum.bit
    %index = arith.extui %m : i1 to i64
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %b = func.call @dynamic_qubit_helper(%q) : (!quantum.bit) -> !quantum.bit
    %rout = quantum.insert %reg[%index], %b : !quantum.reg, !quantum.bit
    quantum.dealloc %rout : !quantum.reg
    return
}

func.func private @dynamic_qubit_helper(%q: !quantum.bit) -> !quantum.bit {
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    return %out : !quantum.bit
}

// -----

// Two unresolved indices cannot be treated as known physical identities.
// CHECK-LABEL: "different_dynamic_insert_index": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @different_dynamic_insert_index(%index0: i64, %index1: i64) {
    %reg = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %reg[%index0] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    %rout = quantum.insert %reg[%index1], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %rout : !quantum.reg
    return
}

// -----

// Inserting an unrelated physical qubit has no proven register identity.
// CHECK-LABEL: "dynamic_insert_replacement": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @dynamic_insert_replacement(%index: i64, %q: !quantum.bit) {
    %reg = quantum.alloc(2) : !quantum.reg
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    %rout = quantum.insert %reg[%index], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %rout : !quantum.reg
    return
}

// -----

// A register permutation after extraction invalidates the matching insertion.
// CHECK-LABEL: "dynamic_insert_after_permutation": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @dynamic_insert_after_permutation(%index: i64) {
    %reg = quantum.alloc(2) : !quantum.reg
    %qd = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["X"](8) %qd : !quantum.bit
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %r0 = quantum.insert %reg[0], %q1 : !quantum.reg, !quantum.bit
    %r1 = quantum.insert %r0[1], %q0 : !quantum.reg, !quantum.bit
    %rout = quantum.insert %r1[%index], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %rout : !quantum.reg
    return
}

// -----

// Loop-carried permutations do not resolve unknown initial qubit identities.
// CHECK-LABEL: "dynamic_loop_carried_permutation": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @dynamic_loop_carried_permutation(%index0: i64, %index1: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[%index0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[%index1] : !quantum.reg -> !quantum.bit
    %out:2 = scf.for %i = %c0 to %c3 step %c1
        iter_args(%a = %q0, %b = %q1) -> (!quantum.bit, !quantum.bit) {
        %next = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b, %next : !quantum.bit, !quantum.bit
    }
    quantum.dealloc %reg : !quantum.reg
    return
}
