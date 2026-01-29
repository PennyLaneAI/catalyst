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

// RUN: quantum-opt --one-shot-mcm --split-input-file -verify-diagnostics %s | FileCheck %s

func.func public @test_expval(%arg0: f64) -> tensor<f64> {
  %1000 = arith.constant 1000 : i64
  quantum.device shots(%1000) ["", "", ""]
  %0 = quantum.alloc( 2) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
  %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
  %2 = quantum.namedobs %out_qubit[ PauliZ] : !quantum.obs
  %3 = quantum.expval %2 : f64
  %from_elements = tensor.from_elements %3 : tensor<f64>
  %4 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
  quantum.dealloc %4 : !quantum.reg
  quantum.device_release
  return %from_elements : tensor<f64>
}

// CHECK: scf.for
// CHECK: func.call @test_expval.quantum_kernel
// module {
//   func.func public @test_expval.quantum_kernel(%arg0: f64) -> tensor<f64> {
//     %c1_i64 = arith.constant 1 : i64
//     quantum.device shots(%c1_i64) ["", "", ""]
//     %0 = quantum.alloc( 2) : !quantum.reg
//     %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
//     %out_qubits = quantum.custom "RX"(%arg0) %1 : !quantum.bit
//     %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
//     %2 = quantum.namedobs %out_qubit[ PauliZ] : !quantum.obs
//     %3 = quantum.expval %2 : f64
//     %from_elements = tensor.from_elements %3 : tensor<f64>
//     %4 = quantum.insert %0[ 0], %out_qubit : !quantum.reg, !quantum.bit
//     quantum.dealloc %4 : !quantum.reg
//     quantum.device_release
//     return %from_elements : tensor<f64>
//   }
//   func.func public @test_expval(%arg0: f64) -> tensor<f64> {
//     %c1000_i64 = arith.constant 1000 : i64
//     %cst = arith.constant 0.000000e+00 : f64
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %0 = index.casts %c1000_i64 : i64 to index
//     %1 = scf.for %arg1 = %c0 to %0 step %c1 iter_args(%arg2 = %cst) -> (f64) {
//       %4 = func.call @test_expval.quantum_kernel(%arg0) : (f64) -> tensor<f64>
//       %extracted = tensor.extract %4[] : tensor<f64>
//       %5 = arith.addf %extracted, %arg2 : f64
//       scf.yield %5 : f64
//     }
//     %2 = arith.sitofp %c1000_i64 : i64 to f64
//     %3 = arith.divf %1, %2 : f64
//     %from_elements = tensor.from_elements %3 : tensor<f64>
//     return %from_elements : tensor<f64>
//   }
// }
