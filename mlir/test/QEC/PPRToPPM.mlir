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

// RUN: quantum-opt --pass-pipeline="builtin.module(ppr-to-ppm)" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK

func.func @test_ppr_to_ppm_clifford(%q1 : !quantum.bit) {
    %0 = qec.ppr ["Z"](4) %q1 : !quantum.bit
    return

    // Prepare |0⟩ on auxiliary qubit
    // PPM -P⊗Y on input qubit and auxiliary qubit -> M1
    // PPM X on auxiliary qubit -> M2
    // XOR M1 and M2 -> M3
    // PPR[Z](2) on input qubit if M3 is true

    // CHECK: [[QREG:%.+]]: !quantum.bit) {
    // CHECK: [[q_2:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[M1:%.+]], [[out_0:%.+]]:2 = qec.ppm ["Z", "Y"](-1) [[QREG]], [[q_2]]
    // CHECK: [[M2:%.+]], [[out_1:%.+]] = qec.ppm ["X"] [[out_0]]#1 : !quantum.bit
    // CHECK: [[q_3:%.+]] = arith.xori [[M1]], [[M2]] : i1
    // CHECK: [[q_4:%.+]] = qec.ppr ["Z"](2) [[out_0]]#0 cond([[q_3]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_multi_clifford(%q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit) {
    %0:3 = qec.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3 : !quantum.bit, !quantum.bit, !quantum.bit
    return

    // CHECK: [[arg0:%.+]]: !quantum.bit, [[arg1:%.+]]: !quantum.bit, [[arg2:%.+]]: !quantum.bit)
    // CHECK: [[q_0:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[m1:%.+]], [[o1:%.+]]:4 = qec.ppm ["X", "Y", "Z", "Y"](-1) [[arg0]], [[arg1]], [[arg2]], [[q_0]]
    // CHECK: [[m2:%.+]], {{.*}} = qec.ppm ["X"] [[o1]]#3 : !quantum.bit
    // CHECK: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK: {{.*}} = qec.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_non_clifford(%q1 : !quantum.bit) {
    %0 = qec.ppr ["Z"](8) %q1 : !quantum.bit
    %0 = qec.ppr ["Z"](4) %q1 : !quantum.bit
    return

    // CHECK: [[magic:%.+]] = qec.fabricate  magic
    // CHECK: [[m_0:%.+]], [[out_0:%.+]]:2 = qec.ppm ["Z", "Z"] %arg0, [[magic]] : !quantum.bit, !quantum.bit
    // CHECK: [[m_1:%.+]], [[out_1:%.+]] = qec.select.ppm([[m_0]], ["Y"], ["X"]) [[out_0]]#1 : !quantum.bit
    // CHECK: [[q0_1:%.+]]  = qec.ppr ["Z"](2) [[out_0]]#0 cond([[m_1]]) : !quantum.bit

    // CHECK: [[QREG:%.+]]: !quantum.bit) {
    // CHECK: [[q_2:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[M1:%.+]], [[out_0:%.+]]:2 = qec.ppm ["Z", "Y"](-1) [[QREG]], [[q_2]]
    // CHECK: [[M2:%.+]], [[out_1:%.+]] = qec.ppm ["X"] [[out_0]]#1 : !quantum.bit
    // CHECK: [[q_3:%.+]] = arith.xori [[M1]], [[M2]] : i1
    // CHECK: [[q_4:%.+]] = qec.ppr ["Z"](2) [[out_0]]#0 cond([[q_3]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_mixed_clifford_non_clifford(%q1 : !quantum.bit) {
    %0 = qec.ppr ["Z"](8) %q1 : !quantum.bit
    return

    // Decompose non-Clifford PPR
    // PPR[P](8) on Q
    // into
    // prepare |m⟩
    // PPM P⊗Z on Q and |m⟩ => m0
    // PPM X on |m⟩ if cond(m0) is true else PPM Y on |m⟩ if cond(m0) is false => m1
    // PPR[P](2) on Q if cond(m1) is true

    // // prepare |m⟩
    // CHECK: [[magic:%.+]] = qec.fabricate  magic

    // // PPM P⊗Z on Q and |m⟩ => m0
    // CHECK: [[m_0:%.+]], [[out_0:%.+]]:2 = qec.ppm ["Z", "Z"] %arg0, [[magic]] : !quantum.bit, !quantum.bit

    // // PPM X or Y on |m⟩ => m1
    // CHECK: [[m_1:%.+]], [[out_1:%.+]] = qec.select.ppm([[m_0]], ["Y"], ["X"]) [[out_0]]#1 : !quantum.bit

    // // PPR[Z](2) on Q if cond(m1) is true
    // CHECK: [[q0_1:%.+]]  = qec.ppr ["Z"](2) [[out_0]]#0 cond([[m_1]]) : !quantum.bit
}


// -----

func.func @test_ppr_to_ppm_with_condition(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit) {
    %m, %0 = qec.ppm ["Z"] %q0 : !quantum.bit
    %1:3 = qec.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3 cond(%m) : !quantum.bit, !quantum.bit, !quantum.bit
    return

    // CHECK: [[m0:%.+]], {{.*}} = qec.ppm ["Z"]
    // CHECK: quantum.alloc_qb : !quantum.bit
    // CHECK: [[m1:%.+]], [[q1:%.+]]:4 = qec.ppm ["X", "Y", "Z", "Y"](-1) {{.*}} cond([[m0]])
    // CHECK: [[m2:%.+]], {{.*}} = qec.ppm ["X"] [[q1]]#3 cond([[m0]])
    // CHECK: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK: {{.*}} = qec.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit

    // CHECK: [[m0:%.+]], {{.*}} = qec.ppm ["Z"]
    // CHECK: qec.fabricate  plus_i : !quantum.bit
    // CHECK: [[m1:%.+]], [[q1:%.+]]:4 = qec.ppm ["X", "Y", "Z", "Z"] {{.*}} cond([[m0]])
    // CHECK: [[m2:%.+]], {{.*}} = qec.ppm ["X"] [[q1]]#3 cond([[m0]])
    // CHECK: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK: {{.*}} = qec.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit
}
