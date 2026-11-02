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
// RUN: quantum-opt -lower-pbc-init-ops -resource-analysis=output-json --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s

// Lowering each supported prepare/fabricate state must leave its PBC-layer cost
// unchanged. Only the measurement and following rotation contribute layers.
// CHECK-LABEL: "initialization_states": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @initialization_states() {
    %zero = pbc.prepare zero : !quantum.bit
    %one = pbc.prepare one : !quantum.bit
    %plus = pbc.prepare plus : !quantum.bit
    %minus = pbc.prepare minus : !quantum.bit
    %plus_i = pbc.prepare plus_i : !quantum.bit
    %minus_i = pbc.prepare minus_i : !quantum.bit
    %magic = pbc.fabricate magic : !quantum.bit
    %magic_conj = pbc.fabricate magic_conj : !quantum.bit
    %m, %out:8 = pbc.ppm ["Z", "Z", "Z", "Z", "Z", "Z", "Z", "Z"]
        %zero, %one, %plus, %minus, %plus_i, %minus_i, %magic, %magic_conj
        : i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit,
          !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %r = pbc.ppr ["X"](8) %out#0 : !quantum.bit
    return
}

// -----

// The ancilla measurement and correction act on disjoint qubits and can share
// a layer under the qubit-only metric, as with ordinary PPMs.
// CHECK-LABEL: "magic_feedforward": {
// CHECK: "qubit_disjoint_depth_total": 3
func.func @magic_feedforward(%data: !quantum.bit) {
    %a = pbc.ppr ["X"](8) %data : !quantum.bit
    %magic = pbc.fabricate magic : !quantum.bit
    %m, %out:2 = pbc.ppm ["Z", "Z"] %a, %magic : i1, !quantum.bit, !quantum.bit
    %m2, %ancilla = pbc.select.ppm (%m ? ["Y"] : ["X"]) %out#1 : i1, !quantum.bit
    %corrected = pbc.ppr ["Z"](2) %out#0 cond(%m2) : !quantum.bit
    quantum.dealloc_qb %ancilla : !quantum.bit
    return
}

// -----

// Preserve the existing leaf cost of an S-gate injection: its final measurement
// and conditional correction act on disjoint qubits, even though the correction
// uses the measurement result through classical arithmetic.
// CHECK-LABEL: "s_injection_leaf_cost": {
// CHECK: "any_commuting_depth": 2
// CHECK: "qubit_disjoint_depth": 2
// CHECK: "qubit_disjoint_depth_total": 2
func.func @s_injection_leaf_cost(%data: !quantum.bit) {
    %ancilla = pbc.prepare plus_i : !quantum.bit
    %m, %out:2 = pbc.ppm ["Z", "Y"] %data, %ancilla
        : i1, !quantum.bit, !quantum.bit
    %m2, %ancilla_out = pbc.ppm ["X"] %out#1 : i1, !quantum.bit
    %condition = arith.xori %m, %m2 : i1
    %corrected = pbc.ppr ["Z"](2) %out#0 cond(%condition) : !quantum.bit
    quantum.dealloc_qb %ancilla_out : !quantum.bit
    return
}

// -----

// Initializations on separate fresh qubits do not serialize independent work.
// CHECK-LABEL: "independent_initializations": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @independent_initializations() {
    %magic = pbc.fabricate magic : !quantum.bit
    %a0 = pbc.ppr ["X"](8) %magic : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %plus = pbc.prepare plus : !quantum.bit
    %b0 = pbc.ppr ["X"](8) %plus : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    return
}

// -----

// A Hadamard on a computational qubit is not an initialization exemption.
// CHECK-LABEL: "computational_hadamard": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @computational_hadamard(%q: !quantum.bit) {
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %h = quantum.custom "Hadamard"() %a : !quantum.bit
    %m, %out = pbc.ppm ["Z"] %h : i1, !quantum.bit
    return
}

// -----

// The same restriction applies to T gates on computational qubits.
// CHECK-LABEL: "computational_t": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @computational_t(%q: !quantum.bit) {
    %t = quantum.custom "T"() %q : !quantum.bit
    %m, %out = pbc.ppm ["Z"] %t : i1, !quantum.bit
    return
}

// -----

// An arbitrary chain after a fresh allocation is not a recognized preparation.
// CHECK-LABEL: "unrecognized_initialization_chain": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @unrecognized_initialization_chain() {
    %q = quantum.alloc_qb : !quantum.bit
    %h = quantum.custom "Hadamard"() %q : !quantum.bit
    %t = quantum.custom "T"() %h : !quantum.bit
    %h2 = quantum.custom "Hadamard"() %t : !quantum.bit
    %m, %out = pbc.ppm ["Z"] %h2 : i1, !quantum.bit
    return
}

// -----

// Controlled gates couple to computational qubits and are not preparations.
// CHECK-LABEL: "controlled_initialization": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @controlled_initialization(%ctrl: !quantum.bit) {
    %true = arith.constant true
    %q = quantum.alloc_qb : !quantum.bit
    %h, %ctrl_out = quantum.custom "Hadamard"() %q ctrls(%ctrl) ctrlvals(%true)
        : !quantum.bit ctrls !quantum.bit
    %m, %out = pbc.ppm ["Z"] %h : i1, !quantum.bit
    return
}

// -----

// The allocation must belong exclusively to the preparation chain.
// CHECK-LABEL: "shared_initialization_input": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @shared_initialization_input() {
    %q = quantum.alloc_qb : !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %h = quantum.custom "Hadamard"() %q : !quantum.bit
    %m, %out = pbc.ppm ["Z"] %h : i1, !quantum.bit
    return
}
