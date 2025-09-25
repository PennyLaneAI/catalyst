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

// RUN: quantum-opt --quantum-specs-info --split-input-file -verify-diagnostics %s | FileCheck %s

//CHECK: {
//CHECK:    "circuit_0": {
//CHECK:        "PPMeasurement_Count": 1,
//CHECK:        "PPRotation_Count": 6,
//CHECK:        "QuantumGate_Count": 1,
//CHECK:        "Total_Count": 8
//CHECK:    }
//CHECK: }
func.func public @circuit_0() -> tensor<i1> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %cst = arith.constant 1.5707963267948966 : f64
    %c0_i64 = arith.constant 0 : i64
    quantum.device shots(%c0_i64) ["/home/zorawar.bassi/pyenv/versions/pennylaneB/lib/python3.13/site-packages/pennylane_lightning/liblightning_qubit_catalyst.so", "LightningSimulator", "{'mcmc': False, 'num_burnin': 0, 'kernel_name': None}"]
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %2 = qec.ppr ["Z"](4) %1 : !quantum.bit
    %3 = qec.ppr ["X"](4) %2 : !quantum.bit
    %4 = qec.ppr ["Z"](4) %3 : !quantum.bit
    %5 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.multirz(%cst) %5, %4 : !quantum.bit, !quantum.bit
    %6 = qec.ppr ["Z"](4) %out_qubits#1 : !quantum.bit
    %7 = qec.ppr ["X"](4) %6 : !quantum.bit
    %8 = qec.ppr ["Z"](4) %7 : !quantum.bit
    %mres, %out_qubits_0 = qec.ppm ["Z"] %8 : !quantum.bit
    %from_elements = tensor.from_elements %mres : tensor<i1>
    %9 = quantum.insert %0[ 1], %out_qubits_0 : !quantum.reg, !quantum.bit
    %10 = quantum.insert %9[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
    quantum.dealloc %10 : !quantum.reg
    quantum.device_release
    return %from_elements : tensor<i1>
}
