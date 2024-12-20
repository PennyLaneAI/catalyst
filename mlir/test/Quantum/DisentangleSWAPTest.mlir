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

// RUN: quantum-opt %s --pass-pipeline="builtin.module(func.func(disentangle-SWAP))" --split-input-file --verify-diagnostics | FileCheck %s


// Explicit unit tests for all SWAP disentangling table entries
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


    %0:2 = quantum.custom "SWAP"() %ZERO_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_0:2 = quantum.custom "CRZ"(%cst) %0#0, %0#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]]:2 = quantum.custom "CNOT"() [[OTHERS_1]], [[ZERO_0]] : !quantum.bit, !quantum.bit
    // CHECK: [[b:%.+]]:2 = quantum.custom "CNOT"() [[a]]#1, [[a]]#0 : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[b]]#0, [[b]]#1 : !quantum.bit, !quantum.bit

    %1:2 = quantum.custom "SWAP"() %ZERO_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_1:2 = quantum.custom "CRZ"(%cst) %1#0, %1#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ZERO_0]], [[ZERO_1]] : !quantum.bit, !quantum.bit

    %2:2 = quantum.custom "SWAP"() %ZERO_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_2:2 = quantum.custom "CRZ"(%cst) %2#0, %2#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "PauliX"() [[ZERO_0]] : !quantum.bit
    // CHECK: [[b:%.+]] = quantum.custom "PauliX"() [[ONE_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]], [[b]] : !quantum.bit, !quantum.bit

    %3:2 = quantum.custom "SWAP"() %ZERO_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_3:2 = quantum.custom "CRZ"(%cst) %3#0, %3#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "Hadamard"() [[ZERO_0]] : !quantum.bit
    // CHECK: [[b:%.+]] = quantum.custom "Hadamard"() [[PLUS_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]], [[b]] : !quantum.bit, !quantum.bit

    %4:2 = quantum.custom "SWAP"() %ZERO_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_4:2 = quantum.custom "CRZ"(%cst) %4#0, %4#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a0:%.+]] = quantum.custom "PauliX"() [[ZERO_0]] : !quantum.bit
    // CHECK: [[a1:%.+]] = quantum.custom "Hadamard"() [[a0]] : !quantum.bit
    // CHECK: [[b0:%.+]] = quantum.custom "Hadamard"() [[MINUS_1]] : !quantum.bit
    // CHECK: [[b1:%.+]] = quantum.custom "PauliX"() [[b0]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a1]], [[b1]] : !quantum.bit, !quantum.bit

    %5:2 = quantum.custom "SWAP"() %ONE_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_5:2 = quantum.custom "CRZ"(%cst) %5#0, %5#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "PauliX"() [[OTHERS_1]] : !quantum.bit
    // CHECK: [[b:%.+]]:2 = quantum.custom "CNOT"() [[a]], [[ONE_0]] : !quantum.bit, !quantum.bit
    // CHECK: [[c:%.+]]:2 = quantum.custom "CNOT"() [[b]]#1, [[b]]#0 : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[c]]#0, [[c]]#1 : !quantum.bit, !quantum.bit

    %6:2 = quantum.custom "SWAP"() %ONE_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_6:2 = quantum.custom "CRZ"(%cst) %6#0, %6#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "PauliX"() [[ONE_0]] : !quantum.bit
    // CHECK: [[b:%.+]] = quantum.custom "PauliX"() [[ZERO_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]], [[b]] : !quantum.bit, !quantum.bit

    %7:2 = quantum.custom "SWAP"() %ONE_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_7:2 = quantum.custom "CRZ"(%cst) %7#0, %7#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[ONE_0]], [[ONE_1]] : !quantum.bit, !quantum.bit

    %8:2 = quantum.custom "SWAP"() %ONE_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_8:2 = quantum.custom "CRZ"(%cst) %8#0, %8#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a0:%.+]] = quantum.custom "PauliX"() [[ONE_0]] : !quantum.bit
    // CHECK: [[a1:%.+]] = quantum.custom "Hadamard"() [[a0]] : !quantum.bit
    // CHECK: [[b0:%.+]] = quantum.custom "Hadamard"() [[PLUS_1]] : !quantum.bit
    // CHECK: [[b1:%.+]] = quantum.custom "PauliX"() [[b0]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a1]], [[b1]] : !quantum.bit, !quantum.bit

        
    %9:2 = quantum.custom "SWAP"() %ONE_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_9:2 = quantum.custom "CRZ"(%cst) %9#0, %9#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "Hadamard"() [[ONE_0]] : !quantum.bit
    // CHECK: [[b:%.+]] = quantum.custom "Hadamard"() [[MINUS_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]], [[b]] : !quantum.bit, !quantum.bit

    %10:2 = quantum.custom "SWAP"() %PLUS_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_10:2 = quantum.custom "CRZ"(%cst) %10#0, %10#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]]:2 = quantum.custom "CNOT"() [[PLUS_0]], [[OTHERS_1]] : !quantum.bit, !quantum.bit
    // CHECK: [[b:%.+]]:2 = quantum.custom "CNOT"() [[a]]#1, [[a]]#0 : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[b]]#1, [[b]]#0 : !quantum.bit, !quantum.bit

    %11:2 = quantum.custom "SWAP"() %PLUS_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_11:2 = quantum.custom "CRZ"(%cst) %11#0, %11#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "Hadamard"() [[PLUS_0]] : !quantum.bit
    // CHECK: [[b:%.+]] = quantum.custom "Hadamard"() [[ZERO_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]], [[b]] : !quantum.bit, !quantum.bit

    %12:2 = quantum.custom "SWAP"() %PLUS_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_12:2 = quantum.custom "CRZ"(%cst) %12#0, %12#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a0:%.+]] = quantum.custom "Hadamard"() [[PLUS_0]] : !quantum.bit
    // CHECK: [[a1:%.+]] = quantum.custom "PauliX"() [[a0]] : !quantum.bit
    // CHECK: [[b0:%.+]] = quantum.custom "PauliX"() [[ONE_1]] : !quantum.bit
    // CHECK: [[b1:%.+]] = quantum.custom "Hadamard"() [[b0]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a1]], [[b1]] : !quantum.bit, !quantum.bit

    %13:2 = quantum.custom "SWAP"() %PLUS_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_13:2 = quantum.custom "CRZ"(%cst) %13#0, %13#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[PLUS_0]], [[PLUS_1]] : !quantum.bit, !quantum.bit

    %14:2 = quantum.custom "SWAP"() %PLUS_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_14:2 = quantum.custom "CRZ"(%cst) %14#0, %14#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "PauliZ"() [[PLUS_0]] : !quantum.bit
    // CHECK: [[b:%.+]] = quantum.custom "PauliZ"() [[MINUS_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]], [[b]] : !quantum.bit, !quantum.bit

    %15:2 = quantum.custom "SWAP"() %MINUS_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_15:2 = quantum.custom "CRZ"(%cst) %15#0, %15#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "PauliZ"() [[OTHERS_1]] : !quantum.bit
    // CHECK: [[b:%.+]]:2 = quantum.custom "CNOT"() [[MINUS_0]], [[a]] : !quantum.bit, !quantum.bit
    // CHECK: [[c:%.+]]:2 = quantum.custom "CNOT"() [[b]]#1, [[b]]#0 : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[c]]#1, [[c]]#0 : !quantum.bit, !quantum.bit

    %16:2 = quantum.custom "SWAP"() %MINUS_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_16:2 = quantum.custom "CRZ"(%cst) %16#0, %16#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a0:%.+]] = quantum.custom "Hadamard"() [[MINUS_0]] : !quantum.bit
    // CHECK: [[a1:%.+]] = quantum.custom "PauliX"() [[a0]] : !quantum.bit
    // CHECK: [[b0:%.+]] = quantum.custom "PauliX"() [[ZERO_1]] : !quantum.bit
    // CHECK: [[b1:%.+]] = quantum.custom "Hadamard"() [[b0]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a1]], [[b1]] : !quantum.bit, !quantum.bit

    %17:2 = quantum.custom "SWAP"() %MINUS_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_17:2 = quantum.custom "CRZ"(%cst) %17#0, %17#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "Hadamard"() [[MINUS_0]] : !quantum.bit
    // CHECK: [[b:%.+]] = quantum.custom "Hadamard"() [[ONE_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]], [[b]] : !quantum.bit, !quantum.bit

    %18:2 = quantum.custom "SWAP"() %MINUS_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_18:2 = quantum.custom "CRZ"(%cst) %18#0, %18#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "PauliZ"() [[MINUS_0]] : !quantum.bit
    // CHECK: [[b:%.+]] = quantum.custom "PauliZ"() [[PLUS_1]] : !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]], [[b]] : !quantum.bit, !quantum.bit

    %19:2 = quantum.custom "SWAP"() %MINUS_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_19:2 = quantum.custom "CRZ"(%cst) %19#0, %19#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[MINUS_0]], [[MINUS_1]] : !quantum.bit, !quantum.bit

    %20:2 = quantum.custom "SWAP"() %OTHERS_0, %OTHERS_1 : !quantum.bit, !quantum.bit
    %user_20:2 = quantum.custom "CRZ"(%cst) %20#0, %20#1 : !quantum.bit, !quantum.bit
    // CHECK: [[a:%.+]]:2 = quantum.custom "SWAP"() [[OTHERS_0]], [[OTHERS_1]] : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[a]]#0, [[a]]#1 : !quantum.bit, !quantum.bit

    %21:2 = quantum.custom "SWAP"() %OTHERS_0, %ZERO_1 : !quantum.bit, !quantum.bit
    %user_21:2 = quantum.custom "CRZ"(%cst) %21#0, %21#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]]:2 = quantum.custom "CNOT"() [[OTHERS_0]], [[ZERO_1]] : !quantum.bit, !quantum.bit
    // CHECK: [[b:%.+]]:2 = quantum.custom "CNOT"() [[a]]#1, [[a]]#0 : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[b]]#1, [[b]]#0 : !quantum.bit, !quantum.bit

    %22:2 = quantum.custom "SWAP"() %OTHERS_0, %ONE_1 : !quantum.bit, !quantum.bit
    %user_22:2 = quantum.custom "CRZ"(%cst) %22#0, %22#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "PauliX"() [[OTHERS_0]] : !quantum.bit
    // CHECK: [[b:%.+]]:2 = quantum.custom "CNOT"() [[a]], [[ONE_1]] : !quantum.bit, !quantum.bit
    // CHECK: [[c:%.+]]:2 = quantum.custom "CNOT"() [[b]]#1, [[b]]#0 : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[c]]#1, [[c]]#0 : !quantum.bit, !quantum.bit

    %23:2 = quantum.custom "SWAP"() %OTHERS_0, %PLUS_1 : !quantum.bit, !quantum.bit
    %user_23:2 = quantum.custom "CRZ"(%cst) %23#0, %23#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]]:2 = quantum.custom "CNOT"() [[PLUS_1]], [[OTHERS_0]] : !quantum.bit, !quantum.bit
    // CHECK: [[b:%.+]]:2 = quantum.custom "CNOT"() [[a]]#1, [[a]]#0 : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[b]]#0, [[b]]#1 : !quantum.bit, !quantum.bit

    %24:2 = quantum.custom "SWAP"() %OTHERS_0, %MINUS_1 : !quantum.bit, !quantum.bit
    %user_24:2 = quantum.custom "CRZ"(%cst) %24#0, %24#1 : !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.custom "SWAP"
    // CHECK: [[a:%.+]] = quantum.custom "PauliZ"() [[OTHERS_0]] : !quantum.bit
    // CHECK: [[b:%.+]]:2 = quantum.custom "CNOT"() [[MINUS_1]], [[a]] : !quantum.bit, !quantum.bit
    // CHECK: [[c:%.+]]:2 = quantum.custom "CNOT"() [[b]]#1, [[b]]#0 : !quantum.bit, !quantum.bit
    // CHECK: {{%.+}} = quantum.custom "CRZ"({{%.+}}) [[c]]#0, [[c]]#1 : !quantum.bit, !quantum.bit

    return %cst : f64
  }
