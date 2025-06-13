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

// RUN: quantum-opt quantum-opt --to-ppr --ppm-specs --commute-ppr --ppm-specs --merge-ppr-ppm --ppm-specs --decompose-non-clifford-ppr --ppm-specs --decompose-clifford-ppr --ppm-specs --split-input-file -verify-diagnostics %s > %t.ppm

func.func @test_clifford_t_to_ppm_1() -> (tensor<i1>, tensor<i1>) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "S"() %1 : !quantum.bit
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits_0 = quantum.custom "Hadamard"() %2 : !quantum.bit
    %out_qubits_1 = quantum.custom "T"() %out_qubits_0 : !quantum.bit
    %out_qubits_2:2 = quantum.custom "CNOT"() %out_qubits_1, %out_qubits : !quantum.bit, !quantum.bit
    %mres_0, %out_qubit_0 = quantum.measure %out_qubits_2#0 : i1, !quantum.bit
    %from_elements_0 = tensor.from_elements %mres_0 : tensor<i1>
    %mres_1, %out_qubit_1 = quantum.measure %out_qubits_2#1 : i1, !quantum.bit
    %from_elements_1 = tensor.from_elements %mres_1 : tensor<i1>
    %3 = quantum.insert %0[ 0], %out_qubit_0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_qubit_1 : !quantum.reg, !quantum.bit
    quantum.dealloc %4 : !quantum.reg
    return %from_elements_0, %from_elements_1 : tensor<i1>, tensor<i1>
    // CHECK: {
    // CHECK:      "test_clifford_t_to_ppm_1": {
    // CHECK:         "max_weight_pi4": 2,
    // CHECK:         "max_weight_pi8": 1,
    // CHECK:         "num_logical_qubits": 2,
    // CHECK:         "num_of_ppm": 2,
    // CHECK:         "num_pi4_gates": 7,
    // CHECK:         "num_pi8_gates": 1
    // CHECK:     }
    // CHECK: }
    // CHECK: {
    // CHECK:     "test_clifford_t_to_ppm_1": {
    // CHECK:         "max_weight_pi4": 2,
    // CHECK:         "max_weight_pi8": 1,
    // CHECK:         "num_logical_qubits": 2,
    // CHECK:         "num_of_ppm": 2,
    // CHECK:         "num_pi4_gates": 7,
    // CHECK:         "num_pi8_gates": 1
    // CHECK:     }
    // CHECK: }
    // CHECK: {
    // CHECK:     "test_clifford_t_to_ppm_1": {
    // CHECK:         "max_weight_pi8": 1,
    // CHECK:         "num_logical_qubits": 2,
    // CHECK:         "num_of_ppm": 2,
    // CHECK:         "num_pi8_gates": 1
    // CHECK:     }
    // CHECK: }
    // CHECK: {
    // CHECK:     "test_clifford_t_to_ppm_1": {
    // CHECK:         "max_weight_pi2": 1,
    // CHECK:         "num_logical_qubits": 2,
    // CHECK:         "num_of_ppm": 5,
    // CHECK:        "num_pi2_gates": 1
    // CHECK:     }
    // CHECK: }
    // CHECK: {
    // CHECK:     "test_clifford_t_to_ppm_1": {
    // CHECK:         "max_weight_pi2": 1,
    // CHECK:         "num_logical_qubits": 2,
    // CHECK:         "num_of_ppm": 5,
    // CHECK:         "num_pi2_gates": 1
    // CHECK:     }
    // CHECK: }
}
