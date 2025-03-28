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

// RUN: quantum-opt %s --pass-pipeline="builtin.module(disentangle-CNOT)" --split-input-file --verify-diagnostics | FileCheck %s


// Explicit unit tests for all CNOT disentangling table entries
// Note that these tests reuse qubits for conciseness.

// CHECK: func.func private @circuit()
  func.func private @circuit() -> f64 {
    %cst = arith.constant 1.230000e+00 : f64
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


    %5:2 = quantum.custom "CNOT"() %ONE_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_5:2 = quantum.custom "CRZ"(%cst) %5#0, %5#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: [[mid_5:%.+]] = quantum.custom "PauliX"() [[OTHERS_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ONE_0]], [[mid_5]] : !quantum.bit, !quantum.bit


    %6:2 = quantum.custom "CNOT"() %ONE_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_6:2 = quantum.custom "CRZ"(%cst) %6#0, %6#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ONE_0]], [[PLUS_1]] : !quantum.bit, !quantum.bit


    %7:2 = quantum.custom "CNOT"() %ONE_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_7:2 = quantum.custom "CRZ"(%cst) %7#0, %7#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ONE_0]], [[MINUS_1]] : !quantum.bit, !quantum.bit


    %8:2 = quantum.custom "CNOT"() %ONE_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_8:2 = quantum.custom "CRZ"(%cst) %8#0, %8#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: [[mid_8:%.+]] = quantum.custom "PauliX"() [[ZERO_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ONE_0]], [[mid_8]] : !quantum.bit, !quantum.bit


    %9:2 = quantum.custom "CNOT"() %ONE_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_9:2 = quantum.custom "CRZ"(%cst) %9#0, %9#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: [[mid_9:%.+]] = quantum.custom "PauliX"() [[ONE_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ONE_0]], [[mid_9]] : !quantum.bit, !quantum.bit


    %10:2 = quantum.custom "CNOT"() %PLUS_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_10:2 = quantum.custom "CRZ"(%cst) %10#0, %10#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_10:%.+]]:2 = quantum.custom "CNOT"() [[PLUS_0]], [[OTHERS_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_10]]#0, [[_10]]#1 : !quantum.bit, !quantum.bit


    %11:2 = quantum.custom "CNOT"() %PLUS_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_11:2 = quantum.custom "CRZ"(%cst) %11#0, %11#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[PLUS_0]], [[PLUS_1]] : !quantum.bit, !quantum.bit


    %12:2 = quantum.custom "CNOT"() %PLUS_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_12:2 = quantum.custom "CRZ"(%cst) %12#0, %12#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: [[mid_12:%.+]] = quantum.custom "PauliZ"() [[PLUS_0]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[mid_12]], [[MINUS_1]] : !quantum.bit, !quantum.bit


    %13:2 = quantum.custom "CNOT"() %PLUS_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_13:2 = quantum.custom "CRZ"(%cst) %13#0, %13#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_13:%.+]]:2 = quantum.custom "CNOT"() [[PLUS_0]], [[ZERO_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_13]]#0, [[_13]]#1 : !quantum.bit, !quantum.bit


    %14:2 = quantum.custom "CNOT"() %PLUS_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_14:2 = quantum.custom "CRZ"(%cst) %14#0, %14#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_14:%.+]]:2 = quantum.custom "CNOT"() [[PLUS_0]], [[ONE_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_14]]#0, [[_14]]#1 : !quantum.bit, !quantum.bit


    %15:2 = quantum.custom "CNOT"() %MINUS_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_15:2 = quantum.custom "CRZ"(%cst) %15#0, %15#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_15:%.+]]:2 = quantum.custom "CNOT"() [[MINUS_0]], [[OTHERS_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_15]]#0, [[_15]]#1 : !quantum.bit, !quantum.bit


    %16:2 = quantum.custom "CNOT"() %MINUS_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_16:2 = quantum.custom "CRZ"(%cst) %16#0, %16#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[MINUS_0]], [[PLUS_1]] : !quantum.bit, !quantum.bit


    %17:2 = quantum.custom "CNOT"() %MINUS_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_17:2 = quantum.custom "CRZ"(%cst) %17#0, %17#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: [[mid_17:%.+]] = quantum.custom "PauliZ"() [[MINUS_0]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[mid_17]], [[MINUS_1]] : !quantum.bit, !quantum.bit


    %18:2 = quantum.custom "CNOT"() %MINUS_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_18:2 = quantum.custom "CRZ"(%cst) %18#0, %18#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_18:%.+]]:2 = quantum.custom "CNOT"() [[MINUS_0]], [[ZERO_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_18]]#0, [[_18]]#1 : !quantum.bit, !quantum.bit


    %19:2 = quantum.custom "CNOT"() %MINUS_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_19:2 = quantum.custom "CRZ"(%cst) %19#0, %19#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_19:%.+]]:2 = quantum.custom "CNOT"() [[MINUS_0]], [[ONE_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_19]]#0, [[_19]]#1 : !quantum.bit, !quantum.bit


    %20:2 = quantum.custom "CNOT"() %OTHERS_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_20:2 = quantum.custom "CRZ"(%cst) %20#0, %20#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_20:%.+]]:2 = quantum.custom "CNOT"() [[OTHERS_0]], [[OTHERS_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_20]]#0, [[_20]]#1 : !quantum.bit, !quantum.bit


    %21:2 = quantum.custom "CNOT"() %OTHERS_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_21:2 = quantum.custom "CRZ"(%cst) %21#0, %21#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[OTHERS_0]], [[PLUS_1]] : !quantum.bit, !quantum.bit


    %22:2 = quantum.custom "CNOT"() %OTHERS_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_22:2 = quantum.custom "CRZ"(%cst) %22#0, %22#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "CNOT"
    // CHECK: [[mid_22:%.+]] = quantum.custom "PauliZ"() [[OTHERS_0]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[mid_22]], [[MINUS_1]] : !quantum.bit, !quantum.bit


    %23:2 = quantum.custom "CNOT"() %OTHERS_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_23:2 = quantum.custom "CRZ"(%cst) %23#0, %23#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_23:%.+]]:2 = quantum.custom "CNOT"() [[OTHERS_0]], [[ZERO_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_23]]#0, [[_23]]#1 : !quantum.bit, !quantum.bit


    %24:2 = quantum.custom "CNOT"() %OTHERS_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_24:2 = quantum.custom "CRZ"(%cst) %24#0, %24#1 : !quantum.bit, !quantum.bit
    // CHECK: [[_24:%.+]]:2 = quantum.custom "CNOT"() [[OTHERS_0]], [[ONE_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[_24]]#0, [[_24]]#1 : !quantum.bit, !quantum.bit


    %true = arith.constant true
    // CHECK: scf.if
    %scf_res = scf.if %true -> !quantum.bit {
        %ZERO_0_in_if = quantum.extract %_[ 0] : !quantum.reg -> !quantum.bit

        // CHECK: quantum.custom "CNOT"
        %25:2 = quantum.custom "CNOT"() %ZERO_0_in_if, %OTHERS_1 : !quantum.bit, !quantum.bit
        %user_25:2 = quantum.custom "CRZ"(%cst) %25#0, %25#1 : !quantum.bit, !quantum.bit
        scf.yield %user_25#0 : !quantum.bit
    } else {
        scf.yield %ZERO_0 : !quantum.bit
    }

    return %cst : f64
  }
