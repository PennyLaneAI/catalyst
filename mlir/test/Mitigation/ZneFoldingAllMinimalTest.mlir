// Copyright 2023 Xanadu Quantum Technologies Inc.

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

// CHECK-LABEL:   func.func private @circuit.folded(%arg0: index) -> tensor<f64> {
    // CHECK:   [[c0:%.+]] = index.constant 0
    // CHECK:   [[c1:%.+]] = index.constant 1
    // CHECK:   [[shots:%.+]] = arith.constant 0
    // CHECK:   quantum.device shots([[shots]]) ["rtd_lightning.so", "LightningQubit", "{}"]
    // CHECK:   [[qReg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK:   [[q0:%.+]] = quantum.extract [[qReg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK:   [[q0_out:%.+]] = scf.for %arg1 = [[c0]] to %arg0 step [[c1]] iter_args([[q0_in:%.+]] = [[q0]]) -> (!quantum.bit) {
    // CHECK:     [[q0_loop:%.+]] = quantum.custom "Hadamard"() [[q0_in]] : !quantum.bit
    // CHECK:     [[q0_loop2:%.+]] = quantum.custom "Hadamard"() [[q0_loop]] adj : !quantum.bit
    // CHECK:     scf.yield [[q0_loop2]] : !quantum.bit
    // CHECK:   [[q0_out2:%.+]] = quantum.custom "Hadamard"() [[q0_out]] : !quantum.bit
    // CHECK:   [[q1:%.+]] = quantum.extract [[qReg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK:   [[q01_out:%.+]]:2 = scf.for %arg1 = [[c0]] to %arg0 step [[c1]] iter_args([[q01_in1:%.+]] = [[q0_out2]], [[q01_in2:%.+]] = [[q1]]) -> (!quantum.bit, !quantum.bit) {
    // CHECK:     [[q01_loop:%.+]]:2 = quantum.custom "CNOT"() [[q01_in1]], [[q01_in2]] : !quantum.bit, !quantum.bit
    // CHECK:     [[q01_loop2:%.+]]:2 = quantum.custom "CNOT"() [[q01_loop]]#0, [[q01_loop]]#1 adj : !quantum.bit, !quantum.bit
    // CHECK:     scf.yield [[q01_loop2]]#0, [[q01_loop2]]#1 : !quantum.bit, !quantum.bit
    // CHECK:   [[q01_out2:%.+]]:2 = quantum.custom "CNOT"() [[q01_out]]#0, [[q01_out]]#1 : !quantum.bit, !quantum.bit
    // CHECK:   [[q2:%.+]] = quantum.namedobs [[q01_out2]]#0[ PauliY] : !quantum.obs
    // CHECK:   [[result:%.+]] = quantum.expval [[q2]] : f64
    // CHECK:   [[tensorRes:%.+]] = tensor.from_elements [[result]] : tensor<f64>
    // CHECK:   [[q2:%.+]] = quantum.insert %0[ 0], [[q01_out2]]#0 : !quantum.reg, !quantum.bit
    // CHECK:   [[q3:%.+]] = quantum.insert %7[ 1], [[q01_out2]]#1 : !quantum.reg, !quantum.bit
    // CHECK:   quantum.dealloc [[q3]] : !quantum.reg
    // CHECK:   quantum.device_release
    // CHECK:   return [[tensorRes]]

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

//CHECK-LABEL: func.func @mitigated_circuit()
    //CHECK-DAG:    [[c0:%.+]] = index.constant 0
    //CHECK-DAG:    [[c1:%.+]] = index.constant 1
    //CHECK-DAG:    [[c3:%.+]] = index.constant 3
    //CHECK-DAG:    [[emptyRes:%.+]] = tensor.empty() : tensor<3xf64> 
    //CHECK-DAG:    [[dense3:%.+]] = arith.constant dense<[1, 2, 3]>
    // CHECK:        [[results:%.+]] = scf.for [[idx:%.+]] = [[c0]] to [[c3]] step [[c1]] iter_args([[emptyArg:%.+]] = [[emptyRes]]) -> (tensor<3xf64>) {
        // CHECK:    [[scalarFactor:%.+]] = tensor.extract [[dense3]][[[idx]]] : tensor<3xindex>
        // CHECK:    [[intermediateRes:%.+]] = func.call @circuit.folded([[scalarFactor]]) : (index) -> tensor<f64>
        // CHECK:    [[extracted:%.+]] = tensor.extract [[intermediateRes]][] : tensor<f64>
        // CHECK:    [[from_elements:%.+]] = tensor.from_elements [[extracted]] : tensor<1xf64>
            // CHECK:    [[resultsFor:%.+]] = scf.for [[idxJ:%.+]] = [[c0]] to [[c1]] step [[c1]] iter_args([[scalarArg:%.+]] = [[emptyArg]]) -> (tensor<3xf64>) {
                // CHECK:    [[extracted:%.+]] = tensor.extract %from_elements[%arg2] : tensor<1xf64>
                // CHECK:    [[insertedRes:%.+]] = tensor.insert [[extracted]] into %arg3[%arg0] : tensor<3xf64>
                // CHECK:    scf.yield [[insertedRes]]
        // CHECK:    scf.yield [[resultsFor]]
    // CHECK:    return [[results]]
func.func @mitigated_circuit() -> tensor<3xf64> {
    %numFolds = arith.constant dense<[1, 2, 3]> : tensor<3xindex>
    %0 = mitigation.zne @circuit() folding (all) numFolds (%numFolds : tensor<3xindex>) : () -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}
