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

// RUN: quantum-opt %s --pass-pipeline="builtin.module(func.func(disentangle-CNOT{func-name=circuit}))" --split-input-file --verify-diagnostics | FileCheck %s


// Explicit unit tests for all CNOT disentangling table entries
// Note that these tests reuse qubits for conciseness.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64 {
  	%cst = arith.constant 3.140000e+00 : f64
    %_ = quantum.alloc( 2) : !quantum.reg
    %ZERO_0 = quantum.extract %_[ 0] : !quantum.reg -> !quantum.bit
    %ZERO_1 = quantum.extract %_[ 1] : !quantum.reg -> !quantum.bit
    %ONE_0 = quantum.custom "PauliX"() %ZERO_0 : !quantum.bit
    %ONE_1 = quantum.custom "PauliX"() %ZERO_1 : !quantum.bit
    %PLUS_0 = quantum.custom "Hadamard"() %ZERO_0 : !quantum.bit
    %PLUS_1 = quantum.custom "Hadamard"() %ZERO_1 : !quantum.bit
    %MINUS_0 = quantum.custom "Hadamard"() %ONE_0 : !quantum.bit
    %MINUS_1 = quantum.custom "Hadamard"() %ONE_1 : !quantum.bit
    %OTHERS_0 = quantum.custom "RX"(%cst) %ZERO_0 : !quantum.bit
    %OTHERS_1 = quantum.custom "RX"(%cst) %ZERO_1 : !quantum.bit
    // CHECK: [[ZERO_0:%.+]] = quantum.extract {{%.+}}[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[ZERO_1:%.+]] = quantum.extract {{%.+}}[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[ONE_0:%.+]] = quantum.custom "PauliX"() [[ZERO_0]] : !quantum.bit
    // CHECK: [[ONE_1:%.+]] = quantum.custom "PauliX"() [[ZERO_1]] : !quantum.bit
    // CHECK: [[PLUS_0:%.+]] = quantum.custom "Hadamard"() [[ZERO_0]] : !quantum.bit
    // CHECK: [[PLUS_1:%.+]] = quantum.custom "Hadamard"() [[ZERO_1]] : !quantum.bit
    // CHECK: [[MINUS_0:%.+]] = quantum.custom "Hadamard"() [[ONE_0]] : !quantum.bit
    // CHECK: [[MINUS_1:%.+]] = quantum.custom "Hadamard"() [[ONE_1]] : !quantum.bit
    // CHECK: [[OTHERS_0:%.+]] = quantum.custom "RX"({{%.+}}) [[ZERO_0]] : !quantum.bit
    // CHECK: [[OTHERS_1:%.+]] = quantum.custom "RX"({{%.+}}) [[ZERO_1]] : !quantum.bit


    %0:2 = quantum.custom "CNOT"() %ZERO_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_0:2 = quantum.custom "CRZ"(%cst) %0#0, %0#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ZERO_0]], [[OTHERS_1]] : !quantum.bit, !quantum.bit


    %1:2 = quantum.custom "CNOT"() %ZERO_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_1:2 = quantum.custom "CRZ"(%cst) %1#0, %1#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ZERO_0]], [[PLUS_1]] : !quantum.bit, !quantum.bit


    %2:2 = quantum.custom "CNOT"() %ZERO_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_2:2 = quantum.custom "CRZ"(%cst) %2#0, %2#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ZERO_0]], [[MINUS_1]] : !quantum.bit, !quantum.bit


    %3:2 = quantum.custom "CNOT"() %ZERO_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_3:2 = quantum.custom "CRZ"(%cst) %3#0, %3#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ZERO_0]], [[ZERO_1]] : !quantum.bit, !quantum.bit


    %4:2 = quantum.custom "CNOT"() %ZERO_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_4:2 = quantum.custom "CRZ"(%cst) %4#0, %4#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ZERO_0]], [[ONE_1]] : !quantum.bit, !quantum.bit





    return %cst : f64
  }
