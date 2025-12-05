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

func.func @test_to_pauli_frame_init_qubit() -> !quantum.bit {
    // CHECK:      [[q1:%.+]] = quantum.alloc_qb
    // CHECK-NEXT: [[q2:%.+]] = pauli_frame.init [[q1]]
    %q = quantum.alloc_qb : !quantum.bit
    // return [[q2]]
    func.return %q : !quantum.bit
}

// -----

func.func @test_to_pauli_frame_init_qreg() {
    // CHECK:      [[qreg1:%.+]] = quantum.alloc( 1)
    // CHECK-NEXT: [[qreg2:%.+]] = pauli_frame.init_qreg [[qreg1]]
    // CHECK-NEXT: quantum.extract [[qreg2]][ 0]
    %qreg = quantum.alloc( 1) : !quantum.reg
    %q = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    func.return
}

// -----

func.func @test_to_pauli_frame_correct_meas(%arg0 : !quantum.bit) -> (i1, !quantum.bit) {
    // CHECK:      [[mres1:%.+]], [[q1:%.+]] = quantum.measure %arg0
    // CHECK-NEXT: [[mres2:%.+]], [[q2:%.+]] = pauli_frame.correct_measurement [[mres1]], [[q1]]
    %mres, %q = quantum.measure %arg0 : i1, !quantum.bit
    // CHECK: return [[mres2]], [[q2]]
    func.return %mres, %q : i1, !quantum.bit
}