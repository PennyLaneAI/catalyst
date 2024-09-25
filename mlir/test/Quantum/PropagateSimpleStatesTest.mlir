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

// RUN: quantum-opt %s --pass-pipeline="builtin.module(func.func(propagate-simple-states{func-name=circuit}))" --split-input-file --verify-diagnostics | FileCheck %s


// Basic test with an actual circuit that has every state in the standard FSM.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> tensor<8xcomplex<f64>> {
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
// expected-remark@above {{PLUS}}
    %out_qubits_0 = quantum.custom "PauliZ"() %out_qubits : !quantum.bit
// expected-remark@above {{MINUS}}
    %identity = quantum.custom "Identity"() %out_qubits_0 : !quantum.bit
// expected-remark@above {{MINUS}}
    %out_qubits_1 = quantum.custom "S"() %identity : !quantum.bit
// expected-remark@above {{RIGHT}}
    %out_qubits_2 = quantum.custom "PauliY"() %out_qubits_1 : !quantum.bit
// expected-remark@above {{RIGHT}}
    %out_qubits_3 = quantum.custom "PauliX"() %out_qubits_2 : !quantum.bit
// expected-remark@above {{LEFT}}
    %out_qubits_4 = quantum.custom "S"() %out_qubits_3 {adjoint} : !quantum.bit
// expected-remark@above {{PLUS}}
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %out_qubits_5 = quantum.custom "Hadamard"() %2 : !quantum.bit
// expected-remark@above {{PLUS}}
    %out_qubits_6 = quantum.custom "PauliY"() %out_qubits_5 : !quantum.bit
// expected-remark@above {{MINUS}}
    %out_qubits_7:2 = quantum.custom "CNOT"() %out_qubits_4, %out_qubits_6 : !quantum.bit, !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %out_qubits_8 = quantum.custom "PauliX"() %3 : !quantum.bit
// expected-remark@above {{ONE}}
    %out_qubits_10:2 = quantum.custom "CNOT"() %out_qubits_7#1, %out_qubits_8 : !quantum.bit, !quantum.bit
    %4 = quantum.compbasis %out_qubits_7#0, %out_qubits_10#0, %out_qubits_10#1 : !quantum.obs
    %5 = quantum.state %4 : tensor<8xcomplex<f64>>
    return %5 : tensor<8xcomplex<f64>>
  }


// -----


// Test for gates outside the FSM: T

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %cst = arith.constant 3.140000e+00 : f64
    %0 = quantum.alloc( 1) : !quantum.reg
    %T = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %T_out = quantum.custom "T"() %T : !quantum.bit
// expected-remark@above {{NOT_A_BASIS}}
    return %cst : f64
  }



// -----


// Test for gates outside the FSM: RX, RY, RZ

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %cst = arith.constant 3.140000e+00 : f64
    %0 = quantum.alloc( 3) : !quantum.reg
    %RX = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %RX_out = quantum.custom "RX"(%cst) %RX : !quantum.bit
// expected-remark@above {{NOT_A_BASIS}}
    %RY = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %RY_out = quantum.custom "RY"(%cst) %RY : !quantum.bit
// expected-remark@above {{NOT_A_BASIS}}
    %RZ = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %RZ_out = quantum.custom "RZ"(%cst) %RZ : !quantum.bit
// expected-remark@above {{NOT_A_BASIS}}
    %failure_ahead = quantum.custom "Hadamard"() %RZ_out : !quantum.bit
// expected-remark@above {{NOT_A_BASIS}}
    return %cst : f64
  }


// -----


// Test when known states enter unknown gates

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %cst = arith.constant 3.140000e+00 : f64
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %2 = quantum.custom "PauliX"() %1 : !quantum.bit
// expected-remark@above {{ONE}}
    %T_out = quantum.custom "RY"(%cst) %2 : !quantum.bit
// expected-remark@above {{NOT_A_BASIS}}
    return %cst : f64
  }


// -----


// Explicit unit tests for all the FSM transition edges from |0>
// Note that these explicit edge tests reuse qubits for conciseness.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}

    %2 = quantum.custom "PauliX"() %1 : !quantum.bit
// expected-remark@above {{ONE}}
    %3 = quantum.custom "PauliY"() %1 : !quantum.bit
// expected-remark@above {{ONE}}
    %4 = quantum.custom "PauliZ"() %1 : !quantum.bit
// expected-remark@above {{ZERO}}
    %5 = quantum.custom "Hadamard"() %1 : !quantum.bit
// expected-remark@above {{PLUS}}
    %cst = arith.constant 3.140000e+00 : f64
    return %cst : f64
  }


// -----


// Explicit unit tests for all the FSM transition edges from |1>
// Note that these explicit edge tests reuse qubits for conciseness.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %2 = quantum.custom "PauliX"() %1 : !quantum.bit
// expected-remark@above {{ONE}}

    %3 = quantum.custom "PauliX"() %2 : !quantum.bit
// expected-remark@above {{ZERO}}
    %4 = quantum.custom "PauliY"() %2 : !quantum.bit
// expected-remark@above {{ZERO}}
    %5 = quantum.custom "PauliZ"() %2 : !quantum.bit
// expected-remark@above {{ONE}}
    %6 = quantum.custom "Hadamard"() %2 : !quantum.bit
// expected-remark@above {{MINUS}}
    %cst = arith.constant 3.140000e+00 : f64
    return %cst : f64
  }


// -----


// Explicit unit tests for all the FSM transition edges from |+>
// Note that these explicit edge tests reuse qubits for conciseness.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
// expected-remark@above {{PLUS}}

    %3 = quantum.custom "PauliX"() %2 : !quantum.bit
// expected-remark@above {{PLUS}}
    %4 = quantum.custom "PauliY"() %2 : !quantum.bit
// expected-remark@above {{MINUS}}
    %5 = quantum.custom "PauliZ"() %2 : !quantum.bit
// expected-remark@above {{MINUS}}
    %6 = quantum.custom "Hadamard"() %2 : !quantum.bit
// expected-remark@above {{ZERO}}
    %7 = quantum.custom "S"() %2 : !quantum.bit
// expected-remark@above {{LEFT}}
    %cst = arith.constant 3.140000e+00 : f64
    return %cst : f64
  }


// -----


// Explicit unit tests for all the FSM transition edges from |->
// Note that these explicit edge tests reuse qubits for conciseness.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
// expected-remark@above {{PLUS}}
    %3 = quantum.custom "PauliZ"() %2 : !quantum.bit
// expected-remark@above {{MINUS}}

    %4 = quantum.custom "PauliX"() %3 : !quantum.bit
// expected-remark@above {{MINUS}}
    %5 = quantum.custom "PauliY"() %3 : !quantum.bit
// expected-remark@above {{PLUS}}
    %6 = quantum.custom "PauliZ"() %3 : !quantum.bit
// expected-remark@above {{PLUS}}
    %7 = quantum.custom "Hadamard"() %3 : !quantum.bit
// expected-remark@above {{ONE}}
    %8 = quantum.custom "S"() %3 : !quantum.bit
// expected-remark@above {{RIGHT}}
    %cst = arith.constant 3.140000e+00 : f64
    return %cst : f64
  }


// -----


// Explicit unit tests for all the FSM transition edges from |L>
// Note that these explicit edge tests reuse qubits for conciseness.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
// expected-remark@above {{PLUS}}
    %3 = quantum.custom "S"() %2 : !quantum.bit
// expected-remark@above {{LEFT}}

    %4 = quantum.custom "PauliX"() %3 : !quantum.bit
// expected-remark@above {{RIGHT}}
    %5 = quantum.custom "PauliY"() %3 : !quantum.bit
// expected-remark@above {{LEFT}}
    %6 = quantum.custom "PauliZ"() %3 : !quantum.bit
// expected-remark@above {{RIGHT}}
    %7 = quantum.custom "Hadamard"() %3 : !quantum.bit
// expected-remark@above {{RIGHT}}
    %8 = quantum.custom "S"() %3 {adjoint} : !quantum.bit
// expected-remark@above {{PLUS}}
    %cst = arith.constant 3.140000e+00 : f64
    return %cst : f64
  }


// -----


// Explicit unit tests for all the FSM transition edges from |R>
// Note that these explicit edge tests reuse qubits for conciseness.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64  {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
// expected-remark@above {{ZERO}}
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
// expected-remark@above {{PLUS}}
    %3 = quantum.custom "S"() %2 : !quantum.bit
// expected-remark@above {{LEFT}}
    %4 = quantum.custom "PauliX"() %3 : !quantum.bit
// expected-remark@above {{RIGHT}}

    %5 = quantum.custom "PauliX"() %4 : !quantum.bit
// expected-remark@above {{LEFT}}
    %6 = quantum.custom "PauliY"() %4 : !quantum.bit
// expected-remark@above {{RIGHT}}
    %7 = quantum.custom "PauliZ"() %4 : !quantum.bit
// expected-remark@above {{LEFT}}
    %8 = quantum.custom "Hadamard"() %4 : !quantum.bit
// expected-remark@above {{LEFT}}
    %9 = quantum.custom "S"() %4 {adjoint} : !quantum.bit
// expected-remark@above {{MINUS}}
    %cst = arith.constant 3.140000e+00 : f64
    return %cst : f64
  }
