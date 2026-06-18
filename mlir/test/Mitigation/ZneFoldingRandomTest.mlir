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

// RUN: quantum-opt %s --lower-mitigation --split-input-file --verify-diagnostics | FileCheck %s

// Random local folding wraps every gate in a `size`-iteration loop, and on each
// iteration flips a runtime coin (__catalyst__rt__random_double() < 0.5) that
// decides whether to insert a $G G^\dagger$ folding pair (scf.if) or pass the qubits
// through unchanged (else branch).

// CHECK: func.func private @__catalyst__rt__random_double() -> f64

// CHECK-LABEL:   func.func private @circuit.folded(%arg0: index) -> tensor<f64> {
    // CHECK-DAG:   [[half:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK-DAG:   [[c0:%.+]] = index.constant 0
    // CHECK-DAG:   [[c1:%.+]] = index.constant 1
    // CHECK:   [[qReg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK:   [[q0:%.+]] = quantum.extract [[qReg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK:   [[q0_out:%.+]] = scf.for %arg1 = [[c0]] to %arg0 step [[c1]] iter_args([[q0_in:%.+]] = [[q0]]) -> (!quantum.bit) {
    // CHECK:     [[rnd:%.+]] = func.call @__catalyst__rt__random_double() : () -> f64
    // CHECK:     [[coin:%.+]] = arith.cmpf olt, [[rnd]], [[half]] : f64
    // CHECK:     [[folded:%.+]] = scf.if [[coin]] -> (!quantum.bit) {
    // CHECK:       [[h0:%.+]] = quantum.custom "Hadamard"() [[q0_in]] : !quantum.bit
    // CHECK:       [[h0adj:%.+]] = quantum.custom "Hadamard"() [[h0]] adj : !quantum.bit
    // CHECK:       scf.yield [[h0adj]] : !quantum.bit
    // CHECK:     } else {
    // CHECK:       scf.yield [[q0_in]] : !quantum.bit
    // CHECK:     }
    // CHECK:     scf.yield [[folded]] : !quantum.bit
    // CHECK:   [[q0_out2:%.+]] = quantum.custom "Hadamard"() [[q0_out]] : !quantum.bit
    // CHECK:   [[q1:%.+]] = quantum.extract [[qReg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK:   [[q01_out:%.+]]:2 = scf.for %arg1 = [[c0]] to %arg0 step [[c1]] iter_args([[q01_in1:%.+]] = [[q0_out2]], [[q01_in2:%.+]] = [[q1]]) -> (!quantum.bit, !quantum.bit) {
    // CHECK:     [[rnd2:%.+]] = func.call @__catalyst__rt__random_double() : () -> f64
    // CHECK:     [[coin2:%.+]] = arith.cmpf olt, [[rnd2]], [[half]] : f64
    // CHECK:     [[folded2:%.+]]:2 = scf.if [[coin2]] -> (!quantum.bit, !quantum.bit) {
    // CHECK:       [[cnot:%.+]]:2 = quantum.custom "CNOT"() [[q01_in1]], [[q01_in2]] : !quantum.bit, !quantum.bit
    // CHECK:       [[cnotadj:%.+]]:2 = quantum.custom "CNOT"() [[cnot]]#0, [[cnot]]#1 adj : !quantum.bit, !quantum.bit
    // CHECK:       scf.yield [[cnotadj]]#0, [[cnotadj]]#1 : !quantum.bit, !quantum.bit
    // CHECK:     } else {
    // CHECK:       scf.yield [[q01_in1]], [[q01_in2]] : !quantum.bit, !quantum.bit
    // CHECK:     }
    // CHECK:     scf.yield [[folded2]]#0, [[folded2]]#1 : !quantum.bit, !quantum.bit
    // CHECK:   [[q01_out2:%.+]]:2 = quantum.custom "CNOT"() [[q01_out]]#0, [[q01_out]]#1 : !quantum.bit, !quantum.bit
    // CHECK:   [[obs:%.+]] = quantum.namedobs [[q01_out2]]#0[ PauliY] : !quantum.obs
    // CHECK:   [[result:%.+]] = quantum.expval [[obs]] : f64

//CHECK-LABEL: func.func @circuit() -> tensor<f64> attributes {qnode} {
func.func @circuit() -> tensor<f64> attributes {qnode} {
    %shots = arith.constant 0 : i64
    quantum.device shots(%shots) ["rtd_lightning.so", "LightningQubit", "{}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits, %2 : !quantum.bit, !quantum.bit
    %3 = quantum.namedobs %out_qubits_0#0[ PauliY] : !quantum.obs
    %4 = quantum.expval %3 : f64
    %from_elements = tensor.from_elements %4 : tensor<f64>
    %5 = quantum.insert %0[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %6 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<f64>
}

func.func @mitigated_circuit() -> tensor<3xf64> {
    %numFolds = arith.constant dense<[1, 2, 3]> : tensor<3xindex>
    %0 = mitigation.zne @circuit() folding (random) numFolds (%numFolds : tensor<3xindex>) : () -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}
