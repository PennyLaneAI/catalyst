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

// RUN: quantum-opt --ppm-compilation --split-input-file -verify-diagnostics %s > %t.ppm
// RUN: quantum-opt --to-ppr --commute-ppr --merge-ppr-ppm --decompose-non-clifford-ppr --decompose-clifford-ppr --split-input-file -verify-diagnostics %s > %t.ppr
// RUN: test -s %t.ppm
// RUN: test -s %t.ppr
// RUN: diff %t.ppm %t.ppr

// With decompose-method=clifford-corrected and avoid-y-measure=false and max-pauli-size=3
// RUN: quantum-opt --ppm-compilation="decompose-method=clifford-corrected avoid-y-measure=false max-pauli-size=3" --split-input-file -verify-diagnostics %s > %t.ppm.params
// RUN: quantum-opt --to-ppr --commute-ppr="max-pauli-size=3" --merge-ppr-ppm="max-pauli-size=3" --decompose-non-clifford-ppr="decompose-method=clifford-corrected" --decompose-clifford-ppr="avoid-y-measure=false" --split-input-file -verify-diagnostics %s > %t.ppr.params
// RUN: test -s %t.ppm.params
// RUN: test -s %t.ppr.params
// RUN: diff %t.ppm.params %t.ppr.params

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
}

// -----

func.func @game_of_surface_code() -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>) {
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      %c_0 = stablehlo.constant dense<4> : tensor<i64>
      %0 = quantum.alloc( 4) : !quantum.reg
      %c_1 = stablehlo.constant dense<3> : tensor<i64>
      %extracted_2 = tensor.extract %c_1[] : tensor<i64>
      %1 = quantum.extract %0[%extracted_2] : !quantum.reg -> !quantum.bit
      // PPR X(-pi/4) = H Tdag H
      %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
      %out_qubits_3 = quantum.custom "T"() %out_qubits adj : !quantum.bit
      %out_qubits_4 = quantum.custom "Hadamard"() %out_qubits_3 : !quantum.bit
      %extracted_5 = tensor.extract %c[] : tensor<i64>
      %2 = quantum.extract %0[%extracted_5] : !quantum.reg -> !quantum.bit
      %out_qubits_6 = quantum.custom "T"() %2 : !quantum.bit
      %c_7 = stablehlo.constant dense<2> : tensor<i64>
      %extracted_8 = tensor.extract %c_7[] : tensor<i64>
      %3 = quantum.extract %0[%extracted_8] : !quantum.reg -> !quantum.bit
      %c_9 = stablehlo.constant dense<1> : tensor<i64>
      %extracted_10 = tensor.extract %c_9[] : tensor<i64>
      %4 = quantum.extract %0[%extracted_10] : !quantum.reg -> !quantum.bit
      %out_qubits_11:2 = quantum.custom "CNOT"() %3, %4 : !quantum.bit, !quantum.bit
      %out_qubits_12:2 = quantum.custom "CNOT"() %out_qubits_11#1, %out_qubits_6 : !quantum.bit, !quantum.bit
      %out_qubits_13:2 = quantum.custom "CNOT"() %out_qubits_4, %out_qubits_12#1 : !quantum.bit, !quantum.bit
      // PPR X(pi/4) = H T H
      %out_qubits_14 = quantum.custom "T"() %out_qubits_13#1 : !quantum.bit
      %out_qubits_15 = quantum.custom "Hadamard"() %out_qubits_14 : !quantum.bit
      %out_qubits_16 = quantum.custom "T"() %out_qubits_15 adj : !quantum.bit
      %out_qubits_17 = quantum.custom "Hadamard"() %out_qubits_16 : !quantum.bit
      %mres, %out_qubit = quantum.measure %out_qubits_17 : i1, !quantum.bit
      %from_elements = tensor.from_elements %mres : tensor<i1>
      %out_qubits_18 = quantum.custom "S"() %out_qubits_12#0 : !quantum.bit
      %out_qubits_19 = quantum.custom "Hadamard"() %out_qubits_18 : !quantum.bit
      %out_qubits_20 = quantum.custom "T"() %out_qubits_19 : !quantum.bit
      %out_qubits_21 = quantum.custom "Hadamard"() %out_qubits_20 : !quantum.bit
      %mres_22, %out_qubit_23 = quantum.measure %out_qubits_21 : i1, !quantum.bit
      %from_elements_24 = tensor.from_elements %mres_22 : tensor<i1>
      %out_qubits_25 = quantum.custom "Hadamard"() %out_qubits_11#0 : !quantum.bit
      %out_qubits_26 = quantum.custom "T"() %out_qubits_25 : !quantum.bit
      %out_qubits_27 = quantum.custom "Hadamard"() %out_qubits_26 : !quantum.bit
      %out_qubits_28 = quantum.custom "T"() %out_qubits_27 : !quantum.bit
      %out_qubits_29 = quantum.custom "Hadamard"() %out_qubits_28 : !quantum.bit
      %out_qubits_30 = quantum.custom "T"() %out_qubits_29 : !quantum.bit
      %out_qubits_31 = quantum.custom "Hadamard"() %out_qubits_30 : !quantum.bit
      %mres_32, %out_qubit_33 = quantum.measure %out_qubits_31 : i1, !quantum.bit
      %from_elements_34 = tensor.from_elements %mres_32 : tensor<i1>
      %out_qubits_35 = quantum.custom "S"() %out_qubits_13#0 : !quantum.bit
      %out_qubits_36 = quantum.custom "Hadamard"() %out_qubits_35 : !quantum.bit
      %out_qubits_37 = quantum.custom "T"() %out_qubits_36 : !quantum.bit
      %out_qubits_38 = quantum.custom "Hadamard"() %out_qubits_37 : !quantum.bit
      %mres_39, %out_qubit_40 = quantum.measure %out_qubits_38 : i1, !quantum.bit
      %from_elements_41 = tensor.from_elements %mres_39 : tensor<i1>
      %extracted_42 = tensor.extract %c[] : tensor<i64>
      %5 = quantum.insert %0[%extracted_42], %out_qubit : !quantum.reg, !quantum.bit
      %extracted_43 = tensor.extract %c_7[] : tensor<i64>
      %6 = quantum.insert %5[%extracted_43], %out_qubit_33 : !quantum.reg, !quantum.bit
      %extracted_44 = tensor.extract %c_9[] : tensor<i64>
      %7 = quantum.insert %6[%extracted_44], %out_qubit_23 : !quantum.reg, !quantum.bit
      %extracted_45 = tensor.extract %c_1[] : tensor<i64>
      %8 = quantum.insert %7[%extracted_45], %out_qubit_40 : !quantum.reg, !quantum.bit
      quantum.dealloc %8 : !quantum.reg
      quantum.device_release
      return %from_elements, %from_elements_24, %from_elements_34, %from_elements_41 : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>
}


// -----
