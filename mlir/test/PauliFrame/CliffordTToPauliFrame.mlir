// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --to-pauli-frame --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: test_to_pauli_frame_pauli_gates
func.func @test_to_pauli_frame_pauli_gates(%arg0 : !quantum.bit) -> !quantum.bit {
    // CHECK: [[q1:%.+]] = pauli_frame.update{{\s*}}[false, false] %arg0
    %q1 = quantum.custom "I"() %arg0 : !quantum.bit
    // CHECK: [[q2:%.+]] = pauli_frame.update{{\s*}}[true, false] [[q1]]
    %q2 = quantum.custom "X"() %q1 : !quantum.bit
    // CHECK: [[q3:%.+]] = pauli_frame.update{{\s*}}[true, true] [[q2]]
    %q3 = quantum.custom "Y"() %q2 : !quantum.bit
    // CHECK: [[q4:%.+]] = pauli_frame.update{{\s*}}[false, true] [[q3]]
    %q4 = quantum.custom "Z"() %q3 : !quantum.bit
    // CHECK-NOT: quantum.custom
    // CHECK: return [[q4]]
    func.return %q4 : !quantum.bit
}

// -----

// CHECK-LABEL: test_to_pauli_frame_clifford_gates_single_qubit
func.func @test_to_pauli_frame_clifford_gates_single_qubit(%arg0 : !quantum.bit) -> !quantum.bit {
    // CHECK: [[q1a:%.+]] = pauli_frame.update_with_clifford[ Hadamard] %arg0
    // CHECK-NEXT: [[q1b:%.+]] = quantum.custom "Hadamard"() [[q1a]]
    %q1 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    // CHECK: [[q2a:%.+]] = pauli_frame.update_with_clifford[ S] [[q1b]]
    // CHECK-NEXT: [[q2b:%.+]] =  quantum.custom "S"() [[q2a]]
    %q2 = quantum.custom "S"() %q1 : !quantum.bit
    // CHECK: return [[q2b]]
    func.return %q2 : !quantum.bit
}

// -----

// CHECK-LABEL: test_to_pauli_frame_clifford_gates_two_qubit
func.func @test_to_pauli_frame_clifford_gates_two_qubit(%arg0 : !quantum.bit, %arg1 : !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    // CHECK:      [[q1a:%.+]]:2 = pauli_frame.update_with_clifford[ CNOT] %arg0, %arg1
    // CHECK-NEXT: [[q1b:%.+]]:2 = quantum.custom "CNOT"() [[q1a]]#0, [[q1a]]#1
    %q1:2 = quantum.custom "CNOT"() %arg0, %arg1 : !quantum.bit, !quantum.bit
    // CHECK: return [[q1b]]#0, [[q1b]]#1
    func.return %q1#0, %q1#1 : !quantum.bit, !quantum.bit
}

// -----

// CHECK-LABEL: test_to_pauli_frame_non_clifford_gate
func.func @test_to_pauli_frame_non_clifford_gate(%arg0 : !quantum.bit) -> (!quantum.bit) {
    // CHECK: [[xbit:%.+]], [[zbit:%.+]], [[q1:%.+]] = pauli_frame.flush %arg0
    // CHECK: [[q2:%.+]] = scf.if [[xbit]] {{.*}} {
    // CHECK:   [[x_outq:%.+]] = quantum.custom "X"() [[q1]]
    // CHECK:   yield [[x_outq]]
    // CHECK: } else {
    // CHECK:   yield [[q1]]
    // CHECK: }
    // CHECK: [[q3:%.+]] = scf.if [[zbit]] {{.*}} {
    // CHECK:   [[z_outq:%.+]] = quantum.custom "Z"() [[q2]]
    // CHECK:   yield [[z_outq]]
    // CHECK: } else {
    // CHECK:   yield [[q2]]
    // CHECK: }
    // CHECK: [[q4:%.+]] = quantum.custom "T"() [[q3]]
    %q1 = quantum.custom "T"() %arg0 : !quantum.bit
    // CHECK: return [[q4]]
    func.return %q1 : !quantum.bit
}

// -----

// CHECK-LABEL: test_to_pauli_frame_init_qubit
func.func @test_to_pauli_frame_init_qubit() -> !quantum.bit {
    // CHECK:      [[q1:%.+]] = quantum.alloc_qb
    // CHECK-NEXT: [[q2:%.+]] = pauli_frame.init [[q1]]
    %q = quantum.alloc_qb : !quantum.bit
    // return [[q2]]
    func.return %q : !quantum.bit
}

// -----

// CHECK-LABEL: test_to_pauli_frame_init_qreg
func.func @test_to_pauli_frame_init_qreg() {
    // CHECK:      [[qreg1:%.+]] = quantum.alloc( 1)
    // CHECK-NEXT: [[qreg2:%.+]] = pauli_frame.init_qreg [[qreg1]]
    // CHECK-NEXT: quantum.extract [[qreg2]][ 0]
    %qreg = quantum.alloc( 1) : !quantum.reg
    %q = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    func.return
}

// -----

// CHECK-LABEL: test_to_pauli_frame_correct_meas
func.func @test_to_pauli_frame_correct_meas(%arg0 : !quantum.bit) -> (i1, !quantum.bit) {
    // CHECK:      [[mres1:%.+]], [[q1:%.+]] = quantum.measure %arg0
    // CHECK-NEXT: [[mres2:%.+]], [[q2:%.+]] = pauli_frame.correct_measurement [[mres1]], [[q1]]
    %mres, %q = quantum.measure %arg0 : i1, !quantum.bit
    // CHECK: return [[mres2]], [[q2]]
    func.return %mres, %q : i1, !quantum.bit
}

// -----

// CHECK-LABEL: test_to_pauli_frame_expval_single_qubit
func.func @test_to_pauli_frame_expval_single_qubit(%arg0 : !quantum.bit) -> f64 {
    // CHECK: [[xbit:%.+]], [[zbit:%.+]], [[q1:%.+]] = pauli_frame.flush %arg0
    // CHECK: [[q2:%.+]] = scf.if [[xbit]]
    // CHECK:   quantum.custom "X"() [[q1]]
    // CHECK: else
    // CHECK: [[q3:%.+]] = scf.if [[zbit]]
    // CHECK:   quantum.custom "Z"() [[q2]]
    // CHECK: else
    // CHECK: [[obs:%.+]] = quantum.namedobs [[q3]]
    // CHECK: [[res:%.+]] = quantum.expval [[obs]]
    %obs = quantum.namedobs %arg0 [PauliZ] : !quantum.obs
    %res = quantum.expval %obs : f64
    // CHECK: return [[res]]
    func.return %res : f64
}

// -----

// CHECK-LABEL: test_to_pauli_frame_var_single_qubit
func.func @test_to_pauli_frame_var_single_qubit(%arg0 : !quantum.bit) -> f64 {
    // CHECK: [[xbit:%.+]], [[zbit:%.+]], [[q1:%.+]] = pauli_frame.flush %arg0
    // CHECK: [[q2:%.+]] = scf.if [[xbit]]
    // CHECK:   quantum.custom "X"() [[q1]]
    // CHECK: else
    // CHECK: [[q3:%.+]] = scf.if [[zbit]]
    // CHECK:   quantum.custom "Z"() [[q2]]
    // CHECK: else
    // CHECK: [[obs:%.+]] = quantum.namedobs [[q3]]
    // CHECK: [[res:%.+]] = quantum.var [[obs]]
    %obs = quantum.namedobs %arg0 [PauliZ] : !quantum.obs
    %res = quantum.var %obs : f64
    // CHECK: return [[res]]
    func.return %res : f64
}

// -----

// CHECK-LABEL: test_to_pauli_frame_sample_single_qubit
func.func @test_to_pauli_frame_sample_single_qubit(%arg0 : !quantum.bit) -> tensor<1000x1xf64> {
    // CHECK: [[xbit:%.+]], [[zbit:%.+]], [[q1:%.+]] = pauli_frame.flush %arg0
    // CHECK: [[q2:%.+]] = scf.if [[xbit]]
    // CHECK:   quantum.custom "X"() [[q1]]
    // CHECK: else
    // CHECK: [[q3:%.+]] = scf.if [[zbit]]
    // CHECK:   quantum.custom "Z"() [[q2]]
    // CHECK: else
    // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q3]] : !quantum.obs
    // CHECK: [[samples:%.+]] = quantum.sample [[obs]]
    %obs = quantum.compbasis qubits %arg0 : !quantum.obs
    %samples = quantum.sample %obs : tensor<1000x1xf64>
    // CHECK: return [[samples]]
    func.return %samples : tensor<1000x1xf64>
}

// -----

// CHECK-LABEL: test_to_pauli_frame_sample_two_qubits
func.func @test_to_pauli_frame_sample_two_qubits(%arg0 : !quantum.bit, %arg1 : !quantum.bit) -> tensor<1000x2xf64> {
    // CHECK: [[xbit0:%.+]], [[zbit0:%.+]], [[q00:%.+]] = pauli_frame.flush %arg0
    // CHECK: [[q01:%.+]] = scf.if [[xbit0]]
    // CHECK:   quantum.custom "X"() [[q00]]
    // CHECK: else
    // CHECK: [[q02:%.+]] = scf.if [[zbit0]]
    // CHECK:   quantum.custom "Z"() [[q01]]
    // CHECK: else
    // CHECK: [[xbit1:%.+]], [[zbit1:%.+]], [[q10:%.+]] = pauli_frame.flush %arg1
    // CHECK: [[q11:%.+]] = scf.if [[xbit1]]
    // CHECK:   quantum.custom "X"() [[q10]]
    // CHECK: else
    // CHECK: [[q12:%.+]] = scf.if [[zbit]]
    // CHECK:   quantum.custom "Z"() [[q11]]
    // CHECK: else
    // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q02]], [[q12]] : !quantum.obs
    // CHECK: [[samples:%.+]] = quantum.sample [[obs]]
    %obs = quantum.compbasis qubits %arg0, %arg1 : !quantum.obs
    %samples = quantum.sample %obs : tensor<1000x2xf64>
    // CHECK: return [[samples]]
    func.return %samples : tensor<1000x2xf64>
}

// -----

// CHECK-LABEL: test_to_pauli_frame_counts_single_qubit
func.func @test_to_pauli_frame_counts_single_qubit(%arg0 : !quantum.bit) -> tensor<2xi64> {
    // CHECK: [[xbit:%.+]], [[zbit:%.+]], [[q1:%.+]] = pauli_frame.flush %arg0
    // CHECK: [[q2:%.+]] = scf.if [[xbit]]
    // CHECK:   quantum.custom "X"() [[q1]]
    // CHECK: else
    // CHECK: [[q3:%.+]] = scf.if [[zbit]]
    // CHECK:   quantum.custom "Z"() [[q2]]
    // CHECK: else
    // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q3]] : !quantum.obs
    // CHECK: {{%.+}}, [[counts:%.+]] = quantum.counts [[obs]]
    %obs = quantum.compbasis qubits %arg0 : !quantum.obs
    %eigvals, %counts = quantum.counts %obs : tensor<2xf64>, tensor<2xi64>
    // CHECK: return [[counts]]
    func.return %counts : tensor<2xi64>
}

// -----

// CHECK-LABEL: test_to_pauli_frame_probs_single_qubit
func.func @test_to_pauli_frame_probs_single_qubit(%arg0 : !quantum.bit) -> tensor<2xf64> {
    // CHECK: [[xbit:%.+]], [[zbit:%.+]], [[q1:%.+]] = pauli_frame.flush %arg0
    // CHECK: [[q2:%.+]] = scf.if [[xbit]]
    // CHECK:   quantum.custom "X"() [[q1]]
    // CHECK: else
    // CHECK: [[q3:%.+]] = scf.if [[zbit]]
    // CHECK:   quantum.custom "Z"() [[q2]]
    // CHECK: else
    // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[q3]] : !quantum.obs
    // CHECK: [[probs:%.+]] = quantum.probs [[obs]]
    %obs = quantum.compbasis qubits %arg0 : !quantum.obs
    %probs = quantum.probs %obs : tensor<2xf64>
    // CHECK: return [[probs]]
    func.return %probs : tensor<2xf64>
}

// -----

// COM: This program represents the following circuit:
// COM: 0: ──H──X─╭●──T──┤↗├─┤
// COM: 1: ──S──Z─╰X──Y──┤↗├─┤
// CHECK-LABEL: test_to_pauli_frame_integration
func.func @test_to_pauli_frame_integration() -> (i1, i1) {
    // CHECK: quantum.alloc( 2) : !quantum.reg
    // CHECK: pauli_frame.init_qreg {{%.+}} : !quantum.reg
    %qreg = quantum.alloc( 2) : !quantum.reg
    // CHECK: quantum.extract {{%.+}}[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.extract {{%.+}}[ 1] : !quantum.reg -> !quantum.bit
    %q00 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q10 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: pauli_frame.update_with_clifford[ Hadamard] {{%.+}} : !quantum.bit
    // CHECK: quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
    %q01 = quantum.custom "Hadamard"() %q00 : !quantum.bit
    // CHECK: pauli_frame.update_with_clifford[ S] {{%.+}} : !quantum.bit
    // CHECK: quantum.custom "S"() {{%.+}} : !quantum.bit
    %q11 = quantum.custom "S"() %q10 : !quantum.bit
    // CHECK: pauli_frame.update[true, false] {{%.+}} : !quantum.bit
    %q02 = quantum.custom "X"() %q01 : !quantum.bit
    // CHECK: pauli_frame.update[false, true] {{%.+}} : !quantum.bit
    %q12 = quantum.custom "Z"() %q11 : !quantum.bit
    // CHECK: pauli_frame.update_with_clifford[ CNOT] {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit
    // CHECK: quantum.custom "CNOT"() {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit
    %q03, %q13 = quantum.custom "CNOT"() %q02, %q12 : !quantum.bit, !quantum.bit
    // CHECK: pauli_frame.flush {{%.+}} : i1, i1, !quantum.bit
    // CHECK: scf.if {{%.+}} -> (!quantum.bit) {
    // CHECK:   quantum.custom "X"() {{%.+}} : !quantum.bit
    // CHECK:   scf.yield {{%.+}} : !quantum.bit
    // CHECK: } else {
    // CHECK:   scf.yield {{%.+}} : !quantum.bit
    // CHECK: }
    // CHECK: scf.if {{%.+}} -> (!quantum.bit) {
    // CHECK:   quantum.custom "Z"() {{%.+}} : !quantum.bit
    // CHECK:   scf.yield {{%.+}} : !quantum.bit
    // CHECK: } else {
    // CHECK:   scf.yield {{%.+}} : !quantum.bit
    // CHECK: }
    // CHECK: quantum.custom "T"() {{%.+}} : !quantum.bit
    %q04 = quantum.custom "T"() %q03 : !quantum.bit
    // CHECK: pauli_frame.update[true, true] {{%.+}} : !quantum.bit
    %q14 = quantum.custom "Y"() %q13 : !quantum.bit
    // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
    // CHECK: pauli_frame.correct_measurement {{%.+}}, {{%.+}} : i1, !quantum.bit
    %mres0, %q05 = quantum.measure %q04 : i1, !quantum.bit
    // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
    // CHECK: pauli_frame.correct_measurement {{%.+}}, {{%.+}} : i1, !quantum.bit
    %mres1, %q15 = quantum.measure %q14 : i1, !quantum.bit
    // CHECK: return {{%.+}}, {{%.+}} : i1, i1
    func.return %mres0, %mres1 : i1, i1
}

// -----

// COM: This test checks that we correctly handle programs with multiple compbasis observables
// COM: acting on the same qubit and ensure that we only flush the Pauli record for this qubit once.
// CHECK-LABEL: test_to_pauli_frame_multiple_compbasis_obs
func.func @test_to_pauli_frame_multiple_compbasis_obs(%arg0 : !quantum.bit) -> () {
    // CHECK: [[q0:%.+]] = quantum.custom "Hadamard"()
    %q0 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    // CHECK: pauli_frame.flush [[q0]]
    // CHECK: scf.if {{%.+}} -> (!quantum.bit) {
    // CHECK:   quantum.custom "X"()
    // CHECK: }
    // CHECK: [[q1:%.+]] = scf.if {{%.+}} -> (!quantum.bit) {
    // CHECK:   quantum.custom "Z"()
    // CHECK: }
    // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
    // CHECK: quantum.sample [[obs0]] : tensor<1000x1xf64>
    %obs0 = quantum.compbasis qubits %q0 : !quantum.obs
    %samples = quantum.sample %obs0 : tensor<1000x1xf64>
    // CHECK-NOT: pauli_frame.flush
    // CHECK: [[obs1:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
    // CHECK: quantum.probs [[obs1]] : tensor<2xf64>
    %obs1 = quantum.compbasis qubits %q0 : !quantum.obs
    %probs = quantum.probs %obs1 : tensor<2xf64>
    // CHECK: return
    func.return
}

// -----

// COM: This test checks that we correctly handle programs with multiple namedobs observables
// COM: acting on the same qubit and ensure that we only flush the Pauli record for this qubit once.
// CHECK-LABEL: test_to_pauli_frame_multiple_namedobs
func.func @test_to_pauli_frame_multiple_namedobs(%arg0 : !quantum.bit) -> () {
    // CHECK: [[q0:%.+]] = quantum.custom "Hadamard"()
    %q0 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    // CHECK: pauli_frame.flush [[q0]]
    // CHECK: scf.if {{%.+}} -> (!quantum.bit) {
    // CHECK:   quantum.custom "X"()
    // CHECK: }
    // CHECK: [[q1:%.+]] = scf.if {{%.+}} -> (!quantum.bit) {
    // CHECK:   quantum.custom "Z"()
    // CHECK: }
    // CHECK: [[obs0:%.+]] = quantum.namedobs [[q1]][ PauliZ] : !quantum.obs
    // CHECK: quantum.expval [[obs0]]
    %obs0 = quantum.namedobs %q0 [PauliZ] : !quantum.obs
    %expval = quantum.expval %obs0 : f64
    // CHECK-NOT: pauli_frame.flush
    // CHECK: [[obs1:%.+]] = quantum.namedobs [[q1]][ PauliZ] : !quantum.obs
    // CHECK: quantum.var [[obs1]]
    %obs1 = quantum.namedobs %q0 [PauliZ] : !quantum.obs
    %var = quantum.var %obs1 : f64
    // CHECK: return
    func.return
}

// -----

// COM: This test checks that we correctly handle programs with multiple observable types (compbasis
// COM: and namedobs) acting on the same qubit and ensure that we only flush the Pauli record for
// COM: this qubit once.
// CHECK-LABEL: test_to_pauli_frame_multiple_obs_types
func.func @test_to_pauli_frame_multiple_obs_types(%arg0 : !quantum.bit) -> () {
    // CHECK: [[q0:%.+]] = quantum.custom "Hadamard"()
    %q0 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    // CHECK: pauli_frame.flush [[q0]]
    // CHECK: scf.if {{%.+}} -> (!quantum.bit) {
    // CHECK:   quantum.custom "X"()
    // CHECK: }
    // CHECK: [[q1:%.+]] = scf.if {{%.+}} -> (!quantum.bit) {
    // CHECK:   quantum.custom "Z"()
    // CHECK: }
    // CHECK: [[obs0:%.+]] = quantum.compbasis qubits [[q1]] : !quantum.obs
    // CHECK: quantum.sample [[obs0]] : tensor<1000x1xf64>
    %obs0 = quantum.compbasis qubits %q0 : !quantum.obs
    %samples = quantum.sample %obs0 : tensor<1000x1xf64>
    // CHECK-NOT: pauli_frame.flush
    // CHECK: [[obs1:%.+]] = quantum.namedobs [[q1]][ PauliZ] : !quantum.obs
    // CHECK: quantum.var [[obs1]]
    %obs1 = quantum.namedobs %q0 [PauliZ] : !quantum.obs
    %var = quantum.var %obs1 : f64
    // CHECK: return
    func.return
}
