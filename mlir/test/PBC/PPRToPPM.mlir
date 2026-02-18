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

// RUN: quantum-opt --ppr-to-ppm --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK

func.func @test_ppr_to_ppm_clifford(%q1 : !quantum.bit) {
    %0 = pbc.ppr ["Z"](4) %q1 : !quantum.bit
    return

    // Prepare |0⟩ on auxiliary qubit
    // PPM -P⊗Y on input qubit and auxiliary qubit -> M1
    // PPM X on auxiliary qubit -> M2
    // XOR M1 and M2 -> M3
    // PPR[Z](2) on input qubit if M3 is true

    // CHECK: [[QREG:%.+]]: !quantum.bit) {
    // CHECK: [[q_2:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[M1:%.+]], [[out_0:%.+]]:2 = pbc.ppm ["Z", "Y"](-1) [[QREG]], [[q_2]] : i1, !quantum.bit, !quantum.bit
    // CHECK: [[M2:%.+]], [[out_1:%.+]] = pbc.ppm ["X"] [[out_0]]#1 : i1, !quantum.bit
    // CHECK: [[q_3:%.+]] = arith.xori [[M1]], [[M2]] : i1
    // CHECK: [[q_4:%.+]] = pbc.ppr ["Z"](2) [[out_0]]#0 cond([[q_3]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_multi_clifford(%q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit) {
    %0:3 = pbc.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3 : !quantum.bit, !quantum.bit, !quantum.bit
    return

    // CHECK: [[arg0:%.+]]: !quantum.bit, [[arg1:%.+]]: !quantum.bit, [[arg2:%.+]]: !quantum.bit)
    // CHECK: [[q_0:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[m1:%.+]], [[o1:%.+]]:4 = pbc.ppm ["X", "Y", "Z", "Y"](-1) [[arg0]], [[arg1]], [[arg2]], [[q_0]] : i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: [[m2:%.+]], {{.*}} = pbc.ppm ["X"] [[o1]]#3 : i1, !quantum.bit
    // CHECK: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK: {{.*}} = pbc.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_non_clifford(%q1 : !quantum.bit) {
    %0 = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    return

    // Decompose non-Clifford PPR
    // PPR[P](8) on Q
    // into
    // prepare |m⟩
    // PPM P⊗Z on Q and |m⟩ => m0
    // PPM X on |m⟩ if cond(m0) is true else PPM Y on |m⟩ if cond(m0) is false => m1
    // PPR[P](2) on Q if cond(m1) is true

    // CHECK: [[magic:%.+]] = pbc.fabricate  magic
    // CHECK: [[m_0:%.+]], [[out_0:%.+]]:2 = pbc.ppm ["Z", "Z"] %arg0, [[magic]] : i1, !quantum.bit, !quantum.bit
    // CHECK: [[m_1:%.+]], [[out_1:%.+]] = pbc.select.ppm([[m_0]], ["Y"], ["X"]) [[out_0]]#1 : i1, !quantum.bit
    // CHECK: [[q0_1:%.+]]  = pbc.ppr ["Z"](2) [[out_0]]#0 cond([[m_1]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_mixed_clifford_non_clifford(%q1 : !quantum.bit) {
    %0 = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    %1 = pbc.ppr ["Z"](4) %0 : !quantum.bit
    return

    // CHECK: [[magic:%.+]] = pbc.fabricate  magic
    // CHECK: [[m_0:%.+]], [[out_0:%.+]]:2 = pbc.ppm ["Z", "Z"] %arg0, [[magic]] : i1, !quantum.bit, !quantum.bit
    // CHECK: [[m_1:%.+]], [[out_1:%.+]] = pbc.select.ppm([[m_0]], ["Y"], ["X"]) [[out_0]]#1 : i1, !quantum.bit
    // CHECK: [[q_0:%.+]]  = pbc.ppr ["Z"](2) [[out_0]]#0 cond([[m_1]]) : !quantum.bit

    // CHECK: [[q_1:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[M1:%.+]], [[out_2:%.+]]:2 = pbc.ppm ["Z", "Y"](-1) [[q_0]], [[q_1]] : i1, !quantum.bit, !quantum.bit
    // CHECK: [[M2:%.+]], [[out_3:%.+]] = pbc.ppm ["X"] [[out_2]]#1 : i1, !quantum.bit
    // CHECK: [[q_2:%.+]] = arith.xori [[M1]], [[M2]] : i1
    // CHECK: [[q_3:%.+]] = pbc.ppr ["Z"](2) [[out_2]]#0 cond([[q_2]]) : !quantum.bit
}

// -----

func.func @test_ppr_to_ppm_with_condition(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit) {
    %m, %0 = pbc.ppm ["Z"] %q0 : i1, !quantum.bit
    %1:3 = pbc.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3 cond(%m) : !quantum.bit, !quantum.bit, !quantum.bit
    return

    // CHECK: [[m0:%.+]], {{.*}} = pbc.ppm ["Z"]
    // CHECK: quantum.alloc_qb : !quantum.bit
    // CHECK: [[m1:%.+]], [[q1:%.+]]:4 = pbc.ppm ["X", "Y", "Z", "Y"](-1) {{.*}} cond([[m0]]) : i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: [[m2:%.+]], {{.*}} = pbc.ppm ["X"] [[q1]]#3 cond([[m0]]) : i1, !quantum.bit
    // CHECK: [[q_1:%.+]] = arith.xori [[m1]], [[m2]] : i1
    // CHECK: {{.*}} = pbc.ppr ["X", "Y", "Z"](2) {{.*}} cond([[q_1]]) : !quantum.bit
}
