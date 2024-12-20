// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --static-custom-lowering --split-input-file -verify-diagnostics %s | FileCheck %s

func.func public @circuit() -> !quantum.bit {
  // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
  // CHECK: [[qubit:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
  // CHECK: [[sum1:%.+]] = arith.constant 2.000000e-01 : f64
  // CHECK: [[ret1:%.+]] = quantum.custom "RX"([[sum1]]) [[qubit]] : !quantum.bit
  // CHECK: [[sum2:%.+]] = arith.constant 1.000000e-01 : f64
  // CHECK: [[ret2:%.+]] = quantum.custom "RY"([[sum2]]) [[ret1]] : !quantum.bit
  %0 = quantum.alloc( 1) : !quantum.reg
  %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
  %out_qubits1 = quantum.static_custom "RX" [2.000000e-01] %1 : !quantum.bit
  %out_qubits2 = quantum.static_custom "RY" [1.000000e-01] %out_qubits1 : !quantum.bit
  return %out_qubits2 : !quantum.bit
}
