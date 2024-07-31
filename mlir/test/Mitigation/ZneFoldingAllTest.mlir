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

func.func @circuit() -> tensor<f64> attributes {qnode} {
    quantum.device ["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
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

  // CHECK:   func.func private @circuit.folded(%arg0: index) -> tensor<f64> {
      // CHECK:   %c2_i64 = arith.constant 2 : i64
      // CHECK:   %idx0 = index.constant 0
      // CHECK:   %idx1 = index.constant 1
      // CHECK:   quantum.device["rtd_lightning.so", "LightningQubit", "{shots: 0}"]
      // CHECK:   %0 = call @circuit.quantumAlloc(%c2_i64) : (i64) -> !quantum.reg
      // CHECK:   %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
      // CHECK:   %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
      // CHECK:   %2 = scf.for %arg1 = %idx0 to %arg0 step %idx1 -> (!quantum.bit) {
      // CHECK:     %out_qubits = quantum.custom "Hadamard"() %out_qubits {adjoint} : !quantum.bit
      // CHECK:     %out_qubits = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
      // CHECK:     scf.yield %out_qubits: !quantum.bit
      // CHECK:   }
      // CHECK:   %3 = quantum.extract %arg0[ 1] : !quantum.reg -> !quantum.bit
      // CHECK:   %out_qubits_0:2 = quantum.custom "CNOT"() %2, %3 : !quantum.bit, !quantum.bit
      // CHECK:   %4 = scf.for %arg1 = %idx0 to %arg0 step %idx1 -> (!quantum.bit, !quantum.bit) {
      // CHECK:     %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits_0#0, %out_qubits_0#1 {adjoint} : !quantum.bit, !quantum.bit
      // CHECK:     %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
      // CHECK:     scf.yield %out_qubits_0 : (!quantum.bit, !quantum.bit)
      // CHECK:   }
      // CHECK:   %5 = quantum.namedobs %out_qubits_0#0[ PauliY] : !quantum.obs
      // CHECK:   %6 = quantum.expval %5 : f64
      // CHECK:   %from_elements = tensor.from_elements %6 : tensor<f64>
      // CHECK:   %7 = quantum.insert %0[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit
      // CHECK:   %8 = quantum.insert %7[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit
      // CHECK:   quantum.dealloc %8 : !quantum.reg
      // CHECK:   quantum.device_release
      // CHECK:   return %from_elements : tensor<f64>
  // CHECK:   }

  // CHECK: func.func private @circuit.quantumAlloc(%arg0: i64) -> !quantum.reg {
    // CHECK:   %0 = quantum.alloc(%arg0) : !quantum.reg
    // CHECK:   return %0 : !quantum.reg
  // CHECK: }


func.func @mitigated_circuit() -> tensor<3xf64> {
    %scaleFactors = arith.constant dense<[1, 2, 3]> : tensor<3xindex>
    %0 = mitigation.zne @circuit() folding (all) scaleFactors (%scaleFactors : tensor<3xindex>) : () -> tensor<3xf64>
    func.return %0 : tensor<3xf64>
}
//CHECK:    func.func @mitigated_circuit() -> tensor<3xf64> {
    //CHECK:    %idx0 = index.constant 0
    //CHECK:    %idx1 = index.constant 1
    //CHECK:    %idx3 = index.constant 3
    //CHECK:    %cst = arith.constant dense<[1, 2, 3]> : tensor<3xindex>
    //CHECK:    %0 = tensor.empty() : tensor<3xf64>
    //CHECK:    %1 = scf.for %arg0 = %idx0 to %idx3 step %idx1 iter_args(%arg1 = %0) -> (tensor<3xf64>) {
        //CHECK:    %extracted = tensor.extract %cst[%arg0] : tensor<3xindex>
        //CHECK:    %2 = func.call @circuit.folded(%extracted) : (index) -> tensor<f64>
        //CHECK:    %extracted_0 = tensor.extract %2[] : tensor<f64>
        //CHECK:    %from_elements = tensor.from_elements %extracted_0 : tensor<1xf64>
        //CHECK:    %3 = scf.for %arg2 = %idx0 to %idx1 step %idx1 iter_args(%arg3 = %arg1) -> (tensor<3xf64>) {
            //CHECK:    %extracted_1 = tensor.extract %from_elements[%arg2] : tensor<1xf64>
            //CHECK:    %inserted = tensor.insert %extracted_1 into %arg3[%arg0] : tensor<3xf64>
            //CHECK:    scf.yield %inserted : tensor<3xf64>
        //CHECK:    }
        //CHECK:    scf.yield %3 : tensor<3xf64>
    //CHECK:    }
    //CHECK:    return %1 : tensor<3xf64>
//CHECK:    }
