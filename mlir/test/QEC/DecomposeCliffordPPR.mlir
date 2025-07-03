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

// RUN: quantum-opt --pass-pipeline="builtin.module(decompose-clifford-ppr)" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-Z
// RUN: quantum-opt --pass-pipeline="builtin.module(decompose-clifford-ppr{avoid-y-measure=true})" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-Y

func.func @test_ppr_to_ppm(%q1 : !quantum.bit) {
    %0 = qec.ppr ["Z"](4) %q1 : !quantum.bit
    return

    // // Decompose by preparing |0⟩

    // // Prepare |0⟩ on auxiliary qubit
    // // PPM -P⊗Y on input qubit and auxiliary qubit -> M1
    // // PPM X on auxiliary qubit -> M2
    // // XOR M1 and M2 -> M3
    // // PPR[Z](2) on input qubit if M3 is true

    // CHECK-Z: [[QREG:%.+]]: !quantum.bit) {
    // CHECK-Z: [[q_2:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK-Z: [[M1:%.+]], [[out_0:%.+]]:2 = qec.ppm ["Z", "Y"] [[QREG]], [[q_2]]
    // CHECK-Z: [[M2:%.+]], [[out_1:%.+]] = qec.ppm ["X"] [[out_0]]#1 : !quantum.bit
    // CHECK-Z: [[q_3:%.+]] = arith.xori [[M1]], [[M2]] : i1
    // CHECK-Z: [[q_4:%.+]] = qec.ppr ["Z"](2) [[out_0]]#0 cond([[q_3]]) : !quantum.bit

    // // Decompose by preparing |Y⟩

    // // Prepare |Y⟩ on auxiliary qubit
    // // PPM P⊗Z on input qubit and auxiliary qubit -> M1
    // // PPM X on auxiliary qubit -> M2
    // // XOR M1 and M2 -> M3
    // // PPR[Z](2) on input qubit if M3 is true

    // CHECK-Y: [[QREG:%.+]]: !quantum.bit) {
    // CHECK-Y: [[q_2:%.+]] = qec.fabricate  plus_i : !quantum.bit
    // CHECK-Y: [[M1:%.+]], [[out_0:%.+]]:2 = qec.ppm ["Z", "Z"](-1) [[QREG]], [[q_2]]
    // CHECK-Y: [[M2:%.+]], [[out_1:%.+]] = qec.ppm ["X"] [[out_0]]#1 : !quantum.bit
    // CHECK-Y: [[q_3:%.+]] = arith.xori [[M1]], [[M2]] : i1
    // CHECK-Y: [[q_4:%.+]] = qec.ppr ["Z"](2) [[out_0]]#0 cond([[q_3]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_1(%q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit) {
    %0:3 = qec.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3 : !quantum.bit, !quantum.bit, !quantum.bit
    return

    // CHECK-Z: [[arg0:%.+]]: !quantum.bit, [[arg1:%.+]]: !quantum.bit, [[arg2:%.+]]: !quantum.bit)
    // CHECK-Z: [[q_0:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK-Z: [[m1:%.+]], [[o1:%.+]]:4 = qec.ppm ["X", "Y", "Z", "Y"] [[arg0]], [[arg1]], [[arg2]], [[q_0]]
    // CHECK-Z: [[m2:%.+]], {{.*}} = qec.ppm ["X"] [[o1]]#3 : !quantum.bit
    // CHECK-Z: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK-Z: {{.*}} = qec.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit

    // CHECK-Y: [[arg0:%.+]]: !quantum.bit, [[arg1:%.+]]: !quantum.bit, [[arg2:%.+]]: !quantum.bit)
    // CHECK-Y: [[q_0:%.+]] = qec.fabricate  plus_i : !quantum.bit
    // CHECK-Y: [[m1:%.+]], [[o1:%.+]]:4 = qec.ppm ["X", "Y", "Z", "Z"](-1) [[arg0]], [[arg1]], [[arg2]], [[q_0]]
    // CHECK-Y: [[m2:%.+]], {{.*}} = qec.ppm ["X"] [[o1]]#3 : !quantum.bit
    // CHECK-Y: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK-Y: {{.*}} = qec.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_with_condition(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit) {
    %m, %0 = qec.ppm ["Z"] %q0 : !quantum.bit
    %1:3 = qec.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3 cond(%m) : !quantum.bit, !quantum.bit, !quantum.bit
    return

    // CHECK-Z: [[m0:%.+]], {{.*}} = qec.ppm ["Z"]
    // CHECK-Z: quantum.alloc_qb : !quantum.bit
    // CHECK-Z: [[m1:%.+]], [[q1:%.+]]:4 = qec.ppm ["X", "Y", "Z", "Y"] {{.*}} cond([[m0]])
    // CHECK-Z: [[m2:%.+]], {{.*}} = qec.ppm ["X"] [[q1]]#3 cond([[m0]])
    // CHECK-Z: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK-Z: {{.*}} = qec.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit

    // CHECK-Y: [[m0:%.+]], {{.*}} = qec.ppm ["Z"]
    // CHECK-Y: qec.fabricate  plus_i : !quantum.bit
    // CHECK-Y: [[m1:%.+]], [[q1:%.+]]:4 = qec.ppm ["X", "Y", "Z", "Z"](-1) {{.*}} cond([[m0]])
    // CHECK-Y: [[m2:%.+]], {{.*}} = qec.ppm ["X"] [[q1]]#3 cond([[m0]])
    // CHECK-Y: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK-Y: {{.*}} = qec.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit
}
