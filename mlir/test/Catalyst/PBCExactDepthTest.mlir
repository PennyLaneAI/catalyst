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

// A helper must retain separate entry and exit times for its qubits.
// q0 has four gates altogether; q1 has three independent gates.
// Reserving both inputs for the helper's longest path would incorrectly give six.
// CHECK-LABEL: "argument_ready_times": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 4
func.func @argument_ready_times(%q0: !quantum.bit, %q1: !quantum.bit) {
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %out:2 = func.call @z_asymmetric_helper(%a2, %q1) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    return
}
func.func private @z_asymmetric_helper(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %a = pbc.ppr ["Z"](8) %q0 : !quantum.bit
    %b0 = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    %b2 = pbc.ppr ["X"](8) %b1 : !quantum.bit
    return %a, %b2 : !quantum.bit, !quantum.bit
}

// -----

// Direct-gate oracle for argument_ready_times.
// CHECK-LABEL: "argument_ready_times_flat": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 4
func.func @argument_ready_times_flat(%q0: !quantum.bit, %q1: !quantum.bit) {
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %a3 = pbc.ppr ["Z"](8) %a2 : !quantum.bit
    %b0 = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    %b2 = pbc.ppr ["X"](8) %b1 : !quantum.bit
    return
}

// -----

// Both calls touch q1, but only once. Their three-gate paths on q0 and q2 overlap.
// Whole-call footprint reservations would incorrectly serialize both calls.
// CHECK-LABEL: "partially_overlapping_calls": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 3
func.func @partially_overlapping_calls(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) {
    %a:2 = func.call @z_three_and_one(%q0, %q1) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    %b:2 = func.call @z_three_and_one(%q2, %a#1) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    return
}
func.func private @z_three_and_one(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %b = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    return %a2, %b : !quantum.bit, !quantum.bit
}

// -----

// Direct-gate oracle for partially_overlapping_calls: lengths are 3, 2, and 3.
// CHECK-LABEL: "partially_overlapping_calls_flat": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 3
func.func @partially_overlapping_calls_flat(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) {
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %b0 = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    %c0 = pbc.ppr ["X"](8) %q2 : !quantum.bit
    %c1 = pbc.ppr ["Z"](8) %c0 : !quantum.bit
    %c2 = pbc.ppr ["X"](8) %c1 : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    return
}

// -----

// Iterations need not wait for the longest path on an unrelated carried qubit.
// The two paths have lengths 2 + 3 = 5 and 3 * 3 = 9.
// CHECK-LABEL: "loop_interleaving": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 9
func.func @loop_interleaving(%q0: !quantum.bit, %q1: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %out:2 = scf.for %i = %c0 to %c3 step %c1
        iter_args(%a = %a1, %b = %q1) -> (!quantum.bit, !quantum.bit) {
        %a2 = pbc.ppr ["X"](8) %a : !quantum.bit
        %b0 = pbc.ppr ["X"](8) %b : !quantum.bit
        %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
        %b2 = pbc.ppr ["X"](8) %b1 : !quantum.bit
        scf.yield %a2, %b2 : !quantum.bit, !quantum.bit
    }
    return
}

// -----

// Direct-gate oracle for loop_interleaving.
// CHECK-LABEL: "loop_interleaving_flat": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 9
func.func @loop_interleaving_flat(%q0: !quantum.bit, %q1: !quantum.bit) {
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %b0 = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    %b2 = pbc.ppr ["X"](8) %b1 : !quantum.bit
    %a3 = pbc.ppr ["X"](8) %a2 : !quantum.bit
    %b3 = pbc.ppr ["X"](8) %b2 : !quantum.bit
    %b4 = pbc.ppr ["Z"](8) %b3 : !quantum.bit
    %b5 = pbc.ppr ["X"](8) %b4 : !quantum.bit
    %a4 = pbc.ppr ["X"](8) %a3 : !quantum.bit
    %b6 = pbc.ppr ["X"](8) %b5 : !quantum.bit
    %b7 = pbc.ppr ["Z"](8) %b6 : !quantum.bit
    %b8 = pbc.ppr ["X"](8) %b7 : !quantum.bit
    return
}

// -----

// Repeated invocation of one allocator returns distinct physical qubits.
// CHECK-LABEL: "fresh_call_outputs": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @fresh_call_outputs() {
    %a = func.call @z_allocate() : () -> !quantum.bit
    %b = func.call @z_allocate() : () -> !quantum.bit
    quantum.dealloc_qb %a : !quantum.bit
    quantum.dealloc_qb %b : !quantum.bit
    return
}
func.func private @z_allocate() -> !quantum.bit {
    %q = quantum.alloc_qb : !quantum.bit
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    return %out : !quantum.bit
}

// -----

// Separate loop iterations allocate separate qubits, with no carried dependency.
// CHECK-LABEL: "fresh_loop_ancillas": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @fresh_loop_ancillas() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c3 step %c1 {
        %q = quantum.alloc_qb : !quantum.bit
        %out = pbc.ppr ["X"](8) %q : !quantum.bit
        quantum.dealloc_qb %out : !quantum.bit
    }
    return
}

// -----

// A runtime index must not silently become a conservative alias reservation.
// CHECK-LABEL: "unresolved_wire_index": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "function_calls"
func.func @unresolved_wire_index(%index: i64) {
    %reg = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    %reg_out = quantum.insert %reg[%index], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %reg_out : !quantum.reg
    return
}

// -----

// Fractional expected iterations cannot establish an exact circuit depth.
// CHECK-LABEL: "estimated_loop_count": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "function_calls"
func.func @estimated_loop_count(%q: !quantum.bit, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    } {catalyst.estimated_iterations = 2.5 : f64}
    return
}

// -----

// Even an integral estimate is not an exact trip count.
// CHECK-LABEL: "integer_estimated_loop_count": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "function_calls"
func.func @integer_estimated_loop_count(%q: !quantum.bit, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    } {catalyst.estimated_iterations = 3 : i64}
    return
}

// -----

// Resolvable bounds take precedence over an unrelated estimate.
// CHECK-LABEL: "static_loop_ignores_estimate": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 3
func.func @static_loop_ignores_estimate(%q: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %out = scf.for %i = %c0 to %c3 step %c1 iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    } {catalyst.estimated_iterations = 2.5 : f64}
    return
}

// -----

// A known branch has the chosen branch's depth, not the maximum of both.
// CHECK-LABEL: "constant_branch": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @constant_branch(%q: !quantum.bit) {
    %false = arith.constant false
    %out = scf.if %false -> (!quantum.bit) {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        %b = pbc.ppr ["Z"](8) %a : !quantum.bit
        %c = pbc.ppr ["X"](8) %b : !quantum.bit
        scf.yield %c : !quantum.bit
    } else {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    }
    return
}

// -----

// The chosen circuit graph cannot be determined from an unknown branch selector.
// CHECK-LABEL: "unresolved_branch": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "function_calls"
func.func @unresolved_branch(%q: !quantum.bit, %condition: i1) {
    %out = scf.if %condition -> (!quantum.bit) {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        %b = pbc.ppr ["Z"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    } else {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    }
    return
}

// -----

// A two-qubit gate inside a call transmits the later input time to both outputs.
// The path is q0's three gates, the joint gate, and the last q1 gate.
// CHECK-LABEL: "call_entangling_dependency": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 5
func.func @call_entangling_dependency(%q0: !quantum.bit, %q1: !quantum.bit) {
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %out:2 = func.call @z_entangling_helper(%a2, %q1) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    return
}
func.func private @z_entangling_helper(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %b = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %pair:2 = pbc.ppr ["X", "X"](8) %q0, %b : !quantum.bit, !quantum.bit
    %last = pbc.ppr ["Z"](8) %pair#1 : !quantum.bit
    return %pair#0, %last : !quantum.bit, !quantum.bit
}

// -----

// Each statically known iteration addresses a different physical register slot.
// CHECK-LABEL: "static_induction_wire_indices": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @static_induction_wire_indices() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %reg = quantum.alloc(4) : !quantum.reg
    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%r = %reg) -> (!quantum.reg) {
        %wire = arith.index_cast %i : index to i64
        %q = quantum.extract %r[%wire] : !quantum.reg -> !quantum.bit
        %rotated = pbc.ppr ["X"](8) %q : !quantum.bit
        %rout = quantum.insert %r[%wire], %rotated : !quantum.reg, !quantum.bit
        scf.yield %rout : !quantum.reg
    }
    quantum.dealloc %out : !quantum.reg
    return
}

// -----

// A carried permutation changes which physical qubit receives each iteration.
// q0 has one gate before the loop and two in it; q1 receives one loop gate.
// CHECK-LABEL: "loop_carried_permutation": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 3
func.func @loop_carried_permutation(%q0: !quantum.bit, %q1: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %before = pbc.ppr ["Z"](8) %q0 : !quantum.bit
    %out:2 = scf.for %i = %c0 to %c3 step %c1
        iter_args(%a = %before, %b = %q1) -> (!quantum.bit, !quantum.bit) {
        %rotated = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b, %rotated : !quantum.bit, !quantum.bit
    }
    return
}

// -----

// Exact depths must retain integers above the exact range of a double.
// CHECK-LABEL: "large_exact_loop": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 9007199254740993
func.func @large_exact_loop(%q: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 9007199254740993 : index
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b : !quantum.bit
    }
    return
}

// -----

// A depth exceeding the supported integer range is unavailable, never wrapped.
// CHECK-LABEL: "depth_integer_overflow": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "function_calls"
func.func @depth_integer_overflow(%q: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 9223372036854775807 : index
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%a = %q) -> (!quantum.bit) {
        %b = pbc.ppr ["X"](8) %a : !quantum.bit
        %c = pbc.ppr ["Z"](8) %b : !quantum.bit
        scf.yield %c : !quantum.bit
    }
    return
}

// -----

// Equal branch depths alone do not establish an exact continuation: the branch
// touching q0 serializes with the last gate, whereas the other branch does not.
// CHECK-LABEL: "equal_branch_depth_different_dependencies": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "function_calls"
func.func @equal_branch_depth_different_dependencies(%q0: !quantum.bit, %q1: !quantum.bit, %condition: i1) {
    %out:2 = scf.if %condition -> (!quantum.bit, !quantum.bit) {
        %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
        scf.yield %a, %q1 : !quantum.bit, !quantum.bit
    } else {
        %b = pbc.ppr ["X"](8) %q1 : !quantum.bit
        scf.yield %q0, %b : !quantum.bit, !quantum.bit
    }
    %last = pbc.ppr ["Z"](8) %out#0 : !quantum.bit
    return
}

// -----

// The precise gate name differs, but both alternatives have the same dependency
// transform. The continuation therefore has a branch-independent exact depth.
// CHECK-LABEL: "equal_branch_dependencies": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @equal_branch_dependencies(%q: !quantum.bit, %condition: i1) {
    %out = scf.if %condition -> (!quantum.bit) {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    } else {
        %a = pbc.ppr ["Z"](8) %q : !quantum.bit
        scf.yield %a : !quantum.bit
    }
    %last = pbc.ppr ["X"](8) %out : !quantum.bit
    return
}

// -----

// A statically empty PBC loop has exact depth zero even when its skipped body
// would otherwise require resolving a runtime wire index.
// CHECK-LABEL: "zero_iteration_pbc_loop": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 0
func.func @zero_iteration_pbc_loop(%index: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %reg = quantum.alloc(2) : !quantum.reg
    %out = scf.for %i = %c0 to %c0 step %c1 iter_args(%r = %reg) -> (!quantum.reg) {
        %q = quantum.extract %r[%index] : !quantum.reg -> !quantum.bit
        %rotated = pbc.ppr ["X"](8) %q : !quantum.bit
        %rout = quantum.insert %r[%index], %rotated : !quantum.reg, !quantum.bit
        scf.yield %rout : !quantum.reg
    }
    quantum.dealloc %out : !quantum.reg
    return
}

// -----

// A known branch that skips all PBC instructions likewise has exact depth zero.
// CHECK-LABEL: "inactive_pbc_branch": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 0
func.func @inactive_pbc_branch(%q: !quantum.bit) {
    %false = arith.constant false
    scf.if %false {
        %a = pbc.ppr ["X"](8) %q : !quantum.bit
    }
    return
}

// -----

// Exercise transfer exponentiation beyond the bounded-unrolling path.
// After the first iteration the clocks are (4, 6). Each subsequent iteration
// adds three to both clocks, so the final depth is 6 + 1999 * 3.
// CHECK-LABEL: "large_entangling_loop": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 6003
func.func @large_entangling_loop(%q0: !quantum.bit, %q1: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 2000 : index
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %out:2 = scf.for %i = %c0 to %n step %c1
        iter_args(%a = %a1, %b = %q1) -> (!quantum.bit, !quantum.bit) {
        %next = pbc.ppr ["X"](8) %a : !quantum.bit
        %pair:2 = pbc.ppr ["X", "X"](8) %next, %b : !quantum.bit, !quantum.bit
        %b0 = pbc.ppr ["Z"](8) %pair#1 : !quantum.bit
        %b1 = pbc.ppr ["X"](8) %b0 : !quantum.bit
        scf.yield %pair#0, %b1 : !quantum.bit, !quantum.bit
    }
    return
}

// -----

// Local ancillas are fresh on every iteration, including the exponentiation
// path. Their completed paths do not become artificial loop-carried clocks.
// CHECK-LABEL: "large_fresh_ancilla_loop": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @large_fresh_ancilla_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = arith.constant 2000 : index
    scf.for %i = %c0 to %n step %c1 {
        %q = quantum.alloc_qb : !quantum.bit
        %out = pbc.ppr ["X"](8) %q : !quantum.bit
        quantum.dealloc_qb %out : !quantum.bit
    }
    return
}
