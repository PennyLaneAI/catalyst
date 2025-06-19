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

// RUN: quantum-opt --to-ppr --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_clifford_t_to_ppr(%q1 : !quantum.bit, %q2 : !quantum.bit){
    // pi / 4 = 0.78539816
    // pi / 8 = 826990
    // CHECK-NOT: quantum.custom
    // CHECK: ([[q1:%.+]]: !quantum.bit, [[q2:%.+]]: !quantum.bit)
    // CHECK: [[q1_0_0:%.+]] = qec.ppr ["Z"](4) [[q1]]
    // CHECK: [[q1_0_1:%.+]] = qec.ppr ["X"](4) [[q1_0_0]]
    // CHECK: [[q1_0_2:%.+]] = qec.ppr ["Z"](4) [[q1_0_1]]
    // CHECK: [[q1_1:%.+]] = qec.ppr ["Z"](4) [[q1_0_2]]
    // CHECK: [[q1_2:%.+]] = qec.ppr ["Z"](8) [[q1_1]]
    // CHECK: [[q1_3:%.+]]:2 = qec.ppr ["Z", "X"](4) [[q1_2]], [[q2]]
    %q1_0 = quantum.custom "H"() %q1 : !quantum.bit
    %q1_1 = quantum.custom "S"() %q1_0 : !quantum.bit
    %q1_2 = quantum.custom "T"() %q1_1 : !quantum.bit
    %q1_3:2 = quantum.custom "CNOT"() %q1_2, %q2 : !quantum.bit, !quantum.bit
    // CHECK: [[q1_4:%.+]] = qec.ppr ["Z"](-4) [[q1_3]]#0
    // CHECK: [[q1_5:%.+]] = qec.ppr ["X"](-4) [[q1_3]]#1
    // CHECK-NOT: quantum.custom
    // CHECK-NEXT: return
    func.return
}

// -----

func.func public @test_clifford_t_to_ppr_1() -> (tensor<i1>, tensor<i1>) {
    // CHECK: [[q0:%.+]] = quantum.alloc( 2) : !quantum.reg
    %0 = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[q1_0:%.+]] = quantum.extract [[q0]][ 1]
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1_1:%.+]] = qec.ppr ["Z"](4) [[q1_0]]
    %out_qubits = quantum.custom "S"() %1 : !quantum.bit
    // CHECK: [[q0_0:%.+]] = quantum.extract [[q0]][ 0]
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q0_1_0:%.+]] = qec.ppr ["Z"](4) [[q0_0]]
    // CHECK: [[q0_1_1:%.+]] = qec.ppr ["X"](4) [[q0_1_0]]
    // CHECK: [[q0_1_2:%.+]] = qec.ppr ["Z"](4) [[q0_1_1]]
    %out_qubits_0 = quantum.custom "Hadamard"() %2 : !quantum.bit
    // CHECK: [[q0_2:%.+]] = qec.ppr ["Z"](8) [[q0_1_2]]
    %out_qubits_1 = quantum.custom "T"() %out_qubits_0 : !quantum.bit
    // CHECK: [[q_3:%.+]]:2 = qec.ppr ["Z", "X"](4) [[q0_2]], [[q1_1]]
    // CHECK: [[q_4:%.+]] = qec.ppr ["Z"](-4) [[q_3]]#0
    // CHECK: [[q_5:%.+]] = qec.ppr ["X"](-4) [[q_3]]#1
    %out_qubits_2:2 = quantum.custom "CNOT"() %out_qubits_1, %out_qubits : !quantum.bit, !quantum.bit

    // CHECK: [[mres_0:%.+]], [[q0_4:%.+]] = qec.ppm ["Z"] [[q_4]]
    // CHECK: [[tensor_0:%.+]] = tensor.from_elements [[mres_0]] : tensor<i1>
    %mres_0, %out_qubit_0 = quantum.measure %out_qubits_2#0 : i1, !quantum.bit
    %from_elements_0 = tensor.from_elements %mres_0 : tensor<i1>

    // CHECK: [[mres_1:%.+]], [[q0_5:%.+]] = qec.ppm ["Z"] [[q_5]]
    // CHECK: [[tensor_1:%.+]] = tensor.from_elements [[mres_1]] : tensor<i1>
    %mres_1, %out_qubit_1 = quantum.measure %out_qubits_2#1 : i1, !quantum.bit
    %from_elements_1 = tensor.from_elements %mres_1 : tensor<i1>

    %3 = quantum.insert %0[ 0], %out_qubit_0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_qubit_1 : !quantum.reg, !quantum.bit
    quantum.dealloc %4 : !quantum.reg
    // CHECK: return [[tensor_0]], [[tensor_1]] : tensor<i1>, tensor<i1>
    return %from_elements_0, %from_elements_1 : tensor<i1>, tensor<i1>
}

// -----

func.func @test_clifford_t_to_ppr_2(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    // expected-error @+1 {{failed to legalize operation 'quantum.custom' that was explicitly marked illegal}}
    %0 = quantum.custom "SOME_UNKNOWN_GATE"() %q1 : !quantum.bit // expected-error @+0 {{Unsupported gate. Supported gates: H, S, T, X, Y, Z, S†, T†, and CNOT}}
    %1 = quantum.custom "S"() %0 : !quantum.bit
    %2 = quantum.custom "T"() %1 : !quantum.bit
    %3:2 = quantum.custom "CNOT"() %2, %q2 : !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func @test_clifford_t_to_ppr_3(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %0 = quantum.custom "PauliX"() %q1 : !quantum.bit
    %1 = quantum.custom "PauliY"() %0 : !quantum.bit
    %2 = quantum.custom "PauliZ"() %1 : !quantum.bit
    %3 = quantum.custom "S"() %2 {adjoint} : !quantum.bit
    %4 = quantum.custom "T"() %3 adj : !quantum.bit

    // CHECK: qec.ppr ["X"](2)
    // CHECK: qec.ppr ["Y"](2)
    // CHECK: qec.ppr ["Z"](2)
    // CHECK: qec.ppr ["Z"](-4)
    // CHECK: qec.ppr ["Z"](-8)
    func.return
}
