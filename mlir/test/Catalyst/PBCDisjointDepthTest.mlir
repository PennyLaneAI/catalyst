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

// Independent sequences can overlap even when their operations are not adjacent.
// Keep the existing local, greedy depth available to older consumers.
// CHECK-LABEL: "flat_independent_sequences": {
// CHECK: "any_commuting_depth": 5
// CHECK: "qubit_disjoint_depth": 5
// CHECK: "qubit_disjoint_depth_total": 3
func.func @flat_independent_sequences(%q0: !quantum.bit, %q1: !quantum.bit) {
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %b0 = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    %b2 = pbc.ppr ["X"](8) %b1 : !quantum.bit
    return
}

// -----

// Both three-iteration loops fit into the same three layers.
// CHECK-LABEL: "disjoint_loops": {
// CHECK: "qubit_disjoint_depth": 0
// CHECK: "qubit_disjoint_depth_total": 3
// CHECK: "function_calls"
func.func @disjoint_loops(%q0: !quantum.bit, %q1: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a = scf.for %i = %c0 to %c3 step %c1 iter_args(%q = %q0) -> (!quantum.bit) {
        %r = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %r : !quantum.bit
    }
    %b = scf.for %i = %c0 to %c3 step %c1 iter_args(%q = %q1) -> (!quantum.bit) {
        %r = pbc.ppr ["Z"](8) %q : !quantum.bit
        scf.yield %r : !quantum.bit
    }
    return
}

// -----

// Reusing the output qubit makes the same two loops sequential.
// CHECK-LABEL: "same_qubit_loops": {
// CHECK: "qubit_disjoint_depth_total": 6
func.func @same_qubit_loops(%q0: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a = scf.for %i = %c0 to %c3 step %c1 iter_args(%q = %q0) -> (!quantum.bit) {
        %r = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %r : !quantum.bit
    }
    %b = scf.for %i = %c0 to %c3 step %c1 iter_args(%q = %a) -> (!quantum.bit) {
        %r = pbc.ppr ["Z"](8) %q : !quantum.bit
        scf.yield %r : !quantum.bit
    }
    return
}

// -----

// Distinct data qubits do not permit overlap when both loops use one ancilla.
// CHECK-LABEL: "shared_ancilla_loops": {
// CHECK: "qubit_disjoint_depth_total": 6
func.func @shared_ancilla_loops(%q0: !quantum.bit, %q1: !quantum.bit,
                              %ancilla: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a:2 = scf.for %i = %c0 to %c3 step %c1
        iter_args(%q = %q0, %aux = %ancilla) -> (!quantum.bit, !quantum.bit) {
        %r:2 = pbc.ppr ["X", "Z"](8) %q, %aux : !quantum.bit, !quantum.bit
        scf.yield %r#0, %r#1 : !quantum.bit, !quantum.bit
    }
    %b:2 = scf.for %i = %c0 to %c3 step %c1
        iter_args(%q = %q1, %aux = %a#1) -> (!quantum.bit, !quantum.bit) {
        %r:2 = pbc.ppr ["X", "Z"](8) %q, %aux : !quantum.bit, !quantum.bit
        scf.yield %r#0, %r#1 : !quantum.bit, !quantum.bit
    }
    return
}

// -----

// The two-qubit PPR waits for the longer input path and advances both qubits.
// CHECK-LABEL: "entangling_synchronization": {
// CHECK: "qubit_disjoint_depth_total": 5
func.func @entangling_synchronization(%q0: !quantum.bit, %q1: !quantum.bit) {
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %b0 = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    %pair:2 = pbc.ppr ["X", "X"](8) %a2, %b0 : !quantum.bit, !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %pair#1 : !quantum.bit
    return
}

// -----

// The measurement-dependent branch has an unresolved quantum graph.
// CHECK-LABEL: "measurement_control_dependency": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @measurement_control_dependency(%q0: !quantum.bit, %q1: !quantum.bit) {
    %false = arith.constant false
    %a0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %a2 = pbc.ppr ["X"](8) %a1 : !quantum.bit
    %m, %out = pbc.ppm ["Z"] %a2 : i1, !quantum.bit
    %condition = arith.xori %m, %false : i1
    %b = scf.if %condition -> (!quantum.bit) {
        %r = pbc.ppr ["X"](8) %q1 : !quantum.bit
        scf.yield %r : !quantum.bit
    } else {
        scf.yield %q1 : !quantum.bit
    }
    return
}

// -----

// Loading an unknown branch choice from memory leaves depth unavailable.
// CHECK-LABEL: "measurement_memory_dependency": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @measurement_memory_dependency(%q0: !quantum.bit, %q1: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %memory = memref.alloc() : memref<1xi1>
    %m, %out = pbc.ppm ["Z"] %q0 : i1, !quantum.bit
    memref.store %m, %memory[%c0] : memref<1xi1>
    %condition = memref.load %memory[%c0] : memref<1xi1>
    %b = scf.if %condition -> (!quantum.bit) {
        %r = pbc.ppr ["X"](8) %q1 : !quantum.bit
        scf.yield %r : !quantum.bit
    } else {
        scf.yield %q1 : !quantum.bit
    }
    memref.dealloc %memory : memref<1xi1>
    return
}

// -----

// A register's SSA dependency must not serialize operations on distinct slots.
// CHECK-LABEL: "disjoint_register_loops": {
// CHECK: "qubit_disjoint_depth_total": 3
func.func @disjoint_register_loops() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %reg = quantum.alloc(2) : !quantum.reg
    %a = scf.for %i = %c0 to %c3 step %c1 iter_args(%r = %reg) -> (!quantum.reg) {
        %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
        %qout = pbc.ppr ["X"](8) %q : !quantum.bit
        %rout = quantum.insert %r[0], %qout : !quantum.reg, !quantum.bit
        scf.yield %rout : !quantum.reg
    }
    %b = scf.for %i = %c0 to %c3 step %c1 iter_args(%r = %a) -> (!quantum.reg) {
        %q = quantum.extract %r[1] : !quantum.reg -> !quantum.bit
        %qout = pbc.ppr ["Z"](8) %q : !quantum.bit
        %rout = quantum.insert %r[1], %qout : !quantum.reg, !quantum.bit
        scf.yield %rout : !quantum.reg
    }
    quantum.dealloc %b : !quantum.reg
    return
}

// -----

// Re-extracting slot zero still aliases the qubit modified by the first loop.
// CHECK-LABEL: "overlapping_register_loops": {
// CHECK: "qubit_disjoint_depth_total": 6
func.func @overlapping_register_loops() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %reg = quantum.alloc(2) : !quantum.reg
    %a = scf.for %i = %c0 to %c3 step %c1 iter_args(%r = %reg) -> (!quantum.reg) {
        %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
        %qout = pbc.ppr ["X"](8) %q : !quantum.bit
        %rout = quantum.insert %r[0], %qout : !quantum.reg, !quantum.bit
        scf.yield %rout : !quantum.reg
    }
    %b = scf.for %i = %c0 to %c3 step %c1 iter_args(%r = %a) -> (!quantum.reg) {
        %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
        %qout = pbc.ppr ["Z"](8) %q : !quantum.bit
        %rout = quantum.insert %r[0], %qout : !quantum.reg, !quantum.bit
        scf.yield %rout : !quantum.reg
    }
    quantum.dealloc %b : !quantum.reg
    return
}

// -----

// Calls include nested loop depth and map formal qubits to the actual operands.
// The first two calls overlap; the third follows the first on the same qubit.
// CHECK-LABEL: "call_nested_loops": {
// CHECK: "qubit_disjoint_depth_total": 12
func.func @call_nested_loops(%q0: !quantum.bit, %q1: !quantum.bit) {
    %a = func.call @nested_loop_helper(%q0) : (!quantum.bit) -> !quantum.bit
    %b = func.call @nested_loop_helper(%q1) : (!quantum.bit) -> !quantum.bit
    %c = func.call @nested_loop_helper(%a) : (!quantum.bit) -> !quantum.bit
    return
}

func.func private @nested_loop_helper(%q0: !quantum.bit) -> !quantum.bit {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %outer = scf.for %i = %c0 to %c3 step %c1 iter_args(%q = %q0) -> (!quantum.bit) {
        %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%r = %q) -> (!quantum.bit) {
            %out = pbc.ppr ["X"](8) %r : !quantum.bit
            scf.yield %out : !quantum.bit
        }
        scf.yield %inner : !quantum.bit
    }
    return %outer : !quantum.bit
}

// -----

// Fractional iteration hints cannot establish an exact executed graph.
// CHECK-LABEL: "fractional_iteration_hint": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @fractional_iteration_hint(%q0: !quantum.bit, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%q = %q0) -> (!quantum.bit) {
        %r = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %r : !quantum.bit
    } {catalyst.estimated_iterations = 2.5 : f64}
    return
}

// -----

// Unequal switch branches cannot establish one exact dependency graph.
// CHECK-LABEL: "index_switch_depth": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @index_switch_depth(%q0: !quantum.bit, %selector: index) {
    %out = scf.index_switch %selector -> !quantum.bit
    case 0 {
        %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
        scf.yield %a : !quantum.bit
    }
    default {
        %b = pbc.ppr ["Z"](8) %q0 : !quantum.bit
        %c = pbc.ppr ["X"](8) %b : !quantum.bit
        scf.yield %c : !quantum.bit
    }
    %last = pbc.ppr ["Z"](8) %out : !quantum.bit
    return
}

// -----

// Missing trip counts leave the legacy partial depth but omit the new total.
// CHECK-LABEL: "unknown_iteration_count": {
// CHECK: "qubit_disjoint_depth": 1
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @unknown_iteration_count(%q0: !quantum.bit, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %initial = pbc.ppr ["Z"](8) %q0 : !quantum.bit
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%q = %initial) -> (!quantum.bit) {
        %r = pbc.ppr ["X"](8) %q : !quantum.bit
        scf.yield %r : !quantum.bit
    }
    return
}

// -----

// An unknown register size prevents enumerating the possible aliases.
// CHECK-LABEL: "unknown_register_index": {
// CHECK: "qubit_disjoint_depth": 1
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @unknown_register_index(%index: i64, %size: i64) {
    %reg = quantum.alloc(%size) : !quantum.reg
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    %rout = quantum.insert %reg[%index], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %rout : !quantum.reg
    return
}

// -----

// A bounded loop follows each physical-wire permutation exactly.
// CHECK-LABEL: "loop_carried_permutation": {
// CHECK: "qubit_disjoint_depth_total": 2
// CHECK: "num_qubits"
func.func @loop_carried_permutation(%q0: !quantum.bit, %q1: !quantum.bit) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %out:2 = scf.for %i = %c0 to %c3 step %c1
        iter_args(%a = %q0, %b = %q1) -> (!quantum.bit, !quantum.bit) {
        %next = pbc.ppr ["X"](8) %a : !quantum.bit
        scf.yield %b, %next : !quantum.bit, !quantum.bit
    }
    return
}

// -----

// Until CFG traversal is supported, a multi-block function must not report only
// its first block's depth as the total for the whole function.
// CHECK-LABEL: "multiple_blocks": {
// CHECK: "qubit_disjoint_depth": 1
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @multiple_blocks(%q0: !quantum.bit) -> !quantum.bit {
    %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
    cf.br ^next(%a : !quantum.bit)
^next(%q: !quantum.bit):
    %b = pbc.ppr ["Z"](8) %q : !quantum.bit
    return %b : !quantum.bit
}

// -----

// A zero-trip loop preserves inserted qubit identities even when binding its
// result grows the register map. Extra arguments exercise that map growth.
// CHECK-LABEL: "zero_iteration_register_alias": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @zero_iteration_register_alias(
    %r0: !quantum.reg, %r1: !quantum.reg, %r2: !quantum.reg, %r3: !quantum.reg,
    %r4: !quantum.reg, %r5: !quantum.reg, %r6: !quantum.reg, %r7: !quantum.reg,
    %r8: !quantum.reg, %r9: !quantum.reg, %r10: !quantum.reg, %r11: !quantum.reg,
    %r12: !quantum.reg, %r13: !quantum.reg, %r14: !quantum.reg, %r15: !quantum.reg,
    %r16: !quantum.reg, %r17: !quantum.reg, %r18: !quantum.reg, %r19: !quantum.reg,
    %r20: !quantum.reg, %r21: !quantum.reg, %r22: !quantum.reg, %r23: !quantum.reg,
    %r24: !quantum.reg, %r25: !quantum.reg, %r26: !quantum.reg, %r27: !quantum.reg,
    %r28: !quantum.reg, %r29: !quantum.reg, %r30: !quantum.reg, %r31: !quantum.reg,
    %r32: !quantum.reg, %r33: !quantum.reg, %r34: !quantum.reg, %r35: !quantum.reg,
    %r36: !quantum.reg, %r37: !quantum.reg, %r38: !quantum.reg, %r39: !quantum.reg,
    %r40: !quantum.reg, %r41: !quantum.reg, %r42: !quantum.reg, %r43: !quantum.reg,
    %r44: !quantum.reg, %r45: !quantum.reg) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q = quantum.alloc_qb : !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %r = quantum.insert %r0[0], %a : !quantum.reg, !quantum.bit
    %empty = scf.for %i = %c0 to %c0 step %c1 iter_args(%s = %r) -> (!quantum.reg) {
        scf.yield %s : !quantum.reg
    }
    %b = quantum.extract %empty[0] : !quantum.reg -> !quantum.bit
    %c = pbc.ppr ["Z"](8) %b : !quantum.bit
    return
}

// -----

// An operation outside the PBC basis must not be silently assigned zero depth.
// CHECK-LABEL: "unsupported_quantum_operation": {
// CHECK: "qubit_disjoint_depth": 1
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @unsupported_quantum_operation(%q0: !quantum.bit) {
    %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %b = quantum.custom "Hadamard"() %a : !quantum.bit
    return
}

// -----

// A compact call DAG represents exponentially many operations. Reuse summaries
// instead of expanding all 131072 rotations or exhausting the traversal budget.
// CHECK-LABEL: "bounded_call_dag": {
// CHECK: "qubit_disjoint_depth_total": 131072
func.func @bounded_call_dag(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_16(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_16(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_0(%q: !quantum.bit) -> !quantum.bit {
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    return %a : !quantum.bit
}

func.func private @repeat_1(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_0(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_0(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_2(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_1(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_1(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_3(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_2(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_2(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_4(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_3(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_3(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_5(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_4(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_4(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_6(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_5(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_5(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_7(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_6(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_6(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_8(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_7(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_7(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_9(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_8(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_8(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_10(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_9(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_9(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_11(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_10(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_10(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_12(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_11(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_11(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_13(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_12(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_12(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_14(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_13(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_13(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_15(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_14(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_14(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}

func.func private @repeat_16(%q: !quantum.bit) -> !quantum.bit {
    %a = func.call @repeat_15(%q) : (!quantum.bit) -> !quantum.bit
    %b = func.call @repeat_15(%a) : (!quantum.bit) -> !quantum.bit
    return %b : !quantum.bit
}
