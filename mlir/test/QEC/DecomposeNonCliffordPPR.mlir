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

// RUN: quantum-opt --pass-pipeline="builtin.module(decompose-non-clifford-ppr{decompose-method=clifford-corrected})" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-INJECT
// RUN: quantum-opt --pass-pipeline="builtin.module(decompose-non-clifford-ppr{decompose-method=auto-corrected})" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-AUTO

func.func @test_ppr_to_ppm(%q1 : !quantum.bit) {
    %0 = qec.ppr ["Z"](8) %q1 : !quantum.bit
    return
}

// Decompose via inject magic state method
// PPR[P](8) on Q
// into
// prepare |m⟩
// PPM P⊗Z on Q and |m⟩ => m0
// PPR[Z](4) on Q if cond(m0) is true
// PPM X on |m⟩ => m1
// PPR[Z](2) on Q if cond(m1) is true

// // prepare |m⟩
// CHECK-INJECT: [[magic:%.+]] = qec.fabricate  magic

// // PPM P⊗Z on Q and |m⟩ => m0
// CHECK-INJECT: [[m_0:%.+]], [[out_0:%.+]]:2 = qec.ppm ["Z", "Z"] %arg0, [[magic]] : !quantum.bit, !quantum.bit

// // PPR[Z](4) on Q if cond(m0) is true
// CHECK-INJECT: [[q0:%.+]] = qec.ppr ["Z"](4) [[out_0]]#0 cond([[m_0]]) : !quantum.bit

// // PPM X on |m⟩ => m1
// CHECK-INJECT: [[m_1:%.+]], [[out_1:%.+]] = qec.ppm ["X"] [[out_0]]#1 : !quantum.bit

// // PPR[Z](2) on Q if cond(m1) is true
// CHECK-INJECT: [[q0_1:%.+]]  = qec.ppr ["Z"](2) [[q0]] cond([[m_1]]) : !quantum.bit


// Decompose via auto corrected method
// PPR[P](8) on Q
// into
// prepare |m⟩ and |0⟩
// PPM P⊗Z on Q and |m⟩   => m0
// PPM Z⊗Y on |m⟩ and |0⟩ => m1
// PPM X on |m⟩ => m2
// PPM X or Z on |0⟩ if cond(m0) is true
// Compute XOR to check if m2 and m1 are different and m3 is true
// PPR P(π/2) on Q if cond(m3) is true


// // prepare |m⟩ and |0⟩
// CHECK-AUTO: [[zero_1:%.+]] = quantum.alloc_qb : !quantum.bit
// CHECK-AUTO: [[magic_1:%.+]] = qec.fabricate  magic

// // PPM P⊗Z on Q and |m⟩   => m0
// CHECK-AUTO: [[m_0:%.+]], [[out_0:%.+]]:2 = qec.ppm ["Z", "Z"] %arg0, [[magic_1]] : !quantum.bit, !quantum.bit

// // PPM Z⊗Y on |m⟩ and |0⟩ => m1
// CHECK-AUTO: [[m_1:%.+]], [[out_1:%.+]]:2 = qec.ppm ["Z", "Y"] [[out_0]]#1, [[zero_1]] : !quantum.bit, !quantum.bit

// // PPM X on |m⟩ => m2
// CHECK-AUTO: [[m_2:%.+]], [[out_2:%.+]] = qec.ppm ["X"] [[out_1]]#0 : !quantum.bit

// // PPM X or Z on |0⟩ if cond(m0) is true
// CHECK-AUTO: [[m_3:%.+]], [[out_3:%.+]] = qec.select.ppm([[m_0]], ["X"], ["Z"]) [[out_1]]#1 : !quantum.bit

// // Compute XOR to check if m2 and m1 are different and m3 is true
// CHECK-AUTO: [[cond:%.+]] = arith.xori [[m_1]], [[m_2]] : i1

// // PPR P(π/2) on Q if cond(m3) is true
// CHECK-AUTO: [[qubit:%.+]] = qec.ppr ["Z"](2) [[out_0]]#0 cond([[cond]]) : !quantum.bit

// -----

// Check if the PPR is negative. If so, it will prepare conjugate magic state
func.func @test_ppr_to_ppm_1(%q1 : !quantum.bit) {
    %0 = qec.ppr ["Z"](-8) %q1 : !quantum.bit
    return

    // CHECK-INJECT: qec.fabricate  magic_conj
    // CHECK-AUTO: qec.fabricate  magic_conj
}

// -----

func.func @test_ppr_to_ppm_2(%q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit, %q4 : !quantum.bit) {
    %0:4 = qec.ppr ["X", "Y", "Z", "Y"](8) %q1, %q2, %q3, %q4 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    return

    // // P = ["X", "Y", "Z", "Y"]
    // // PPM P⊗Z on Q and |m⟩   => m0
    // CHECK-INJECT: qec.ppm ["X", "Y", "Z", "Y", "Z"]
    // CHECK-INJECT: qec.ppr ["X", "Y", "Z", "Y"](4) {{.*}} cond({{.*}})
    // CHECK-INJECT: qec.ppm ["X"]
    // CHECK-INJECT: qec.ppr ["X", "Y", "Z", "Y"](2) {{.*}} cond({{.*}})


    // // P = ["X", "Y", "Z", "Y"]
    // // PPM P⊗Z on Q and |m⟩   => m0
    // CHECK-AUTO: qec.ppm ["X", "Y", "Z", "Y", "Z"]
    // CHECK-AUTO: qec.ppm ["Z", "Y"]
    // CHECK-AUTO: qec.ppm ["X"]
    // CHECK-AUTO: qec.select.ppm({{.*}}, ["X"], ["Z"])
    // CHECK-AUTO: arith.xori

    // // PPR P(π/2) on Q if cond(m3) is true
    // CHECK-AUTO: qec.ppr ["X", "Y", "Z", "Y"](2) {{.*}} cond({{.*}})
}

// -----

// Check if the PPR has a condition, it will not be decomposed
func.func @test_ppr_to_ppm_3(%q1 : !quantum.bit, %i1 : i1) {
    %0 = qec.ppr ["Z"](-8) %q1 cond(%i1) : !quantum.bit
    return

    // CHECK-INJECT: qec.ppr ["Z"](-8) {{.*}} cond({{.*}})
    // CHECK-INJECT: return
    // CHECK-AUTO: qec.ppr ["Z"](-8) {{.*}} cond({{.*}})
    // CHECK-AUTO: return
}

// -----

// Q0 -> arg0, Q1 -> arg1, Q2 -> arg2, Q3 -> arg3
func.func public @game_of_surface_code(%arg0: !quantum.bit, %arg1: !quantum.bit, %arg2: !quantum.bit, %arg3: !quantum.bit) {
    // PPR Z -> Q0
    %0 = qec.ppr ["Z"](8) %arg0 : !quantum.bit
    // PPR Y -> Q3
    %1 = qec.ppr ["Y"](-8) %arg3 : !quantum.bit
    // PPR Y, X -> Q2, Q1
    %2:2 = qec.ppr ["Y", "X"](8) %arg2, %arg1 : !quantum.bit, !quantum.bit
    // PPR Z, Z, Y, Z -> Q2, Q1, Q3, Q0
    %3:4 = qec.ppr ["Z", "Z", "Y", "Z"](-8) %2#0, %2#1, %1, %0 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    // PPM Z, Z, Y, Y -> Q2, Q1, Q0, Q3
    %mres, %out_qubits:4 = qec.ppm ["Z", "Z", "Y", "Y"] %3#0, %3#1, %3#3, %3#2 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    // PPM X, X -> Q1, Q0
    %mres_0, %out_qubits_1:2 = qec.ppm ["X", "X"] %out_qubits#1, %out_qubits#2 : !quantum.bit, !quantum.bit
    // PPM Z(-1) -> Q2
    %mres_2, %out_qubits_3 = qec.ppm ["Z"](-1) %out_qubits#0 : !quantum.bit
    // PPM X, X -> Q0, Q3
    %mres_4, %out_qubits_5:2 = qec.ppm ["X", "X"] %out_qubits_1#1, %out_qubits#3 : !quantum.bit, !quantum.bit
    return
    
    /////// Inject magic state Method ///////
    
    // // PPR ["Z"](8) Q0 
    
    // CHECK-INJECT: [[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit, [[Q2:%.+]]: !quantum.bit, [[Q3:%.+]]: !quantum.bit) {
    // CHECK-INJECT: [[q2:%.+]]     = qec.fabricate  magic
    // CHECK-INJECT: [[M:%.+]], [[out:%.+]]:2 = qec.ppm ["Z", "Z"] [[Q0]], [[q2]] : !quantum.bit, !quantum.bit

    // CHECK-INJECT: [[q3:%.+]]  = qec.ppr ["Z"](4) [[out]]#0 cond([[M]]) : !quantum.bit

    // CHECK-INJECT: [[M1:%.+]], [[out_0:%.+]] = qec.ppm ["X"] [[out]]#1 : !quantum.bit
    // CHECK-INJECT: [[q4:%.+]]     = qec.ppr ["Z"](2) [[q3]] cond([[M1]]) : !quantum.bit

    // // Q0 -> %q4

    // // PPR ["Y"](-8) %Q3 
    // CHECK-INJECT: [[q7:%.+]]     = qec.fabricate  magic_conj
    // CHECK-INJECT: [[mres_0:%.+]], [[out:%.+]]:2 = qec.ppm ["Y", "Z"] [[Q3]], [[q7]] : !quantum.bit, !quantum.bit
    // CHECK-INJECT: [[q8:%.+]]     = qec.ppr ["Y"](4) [[out]]#0 cond([[mres_0]]) : !quantum.bit
    // CHECK-INJECT: [[mres_1:%.+]], [[out_1:%.+]] = qec.ppm ["X"] [[out]]#1 : !quantum.bit
    // CHECK-INJECT: [[q9:%.+]]     = qec.ppr ["Y"](2) [[q8]] cond([[mres_1]]) : !quantum.bit

    // // Q3 -> %q9

    // // PPR ["Y", "X"](8) Q2, Q1
    // CHECK-INJECT: [[q12:%.+]] = qec.fabricate  magic
    // CHECK-INJECT: [[mres_2:%.+]], [[out_2:%.+]]:3 = qec.ppm ["Y", "X", "Z"] [[Q2]], [[Q1]], [[q12]]
    // CHECK-INJECT: [[q13:%.+]]:2 = qec.ppr ["Y", "X"](4) [[out_2]]#0, [[out_2]]#1 cond([[mres_2]])
    // CHECK-INJECT: [[mres_3:%.+]], [[out_3:%.+]] = qec.ppm ["X"] [[out_2]]#2 : !quantum.bit
    // CHECK-INJECT: [[q14:%.+]]:2 = qec.ppr ["Y", "X"](2) [[q13]]#0, [[q13]]#1 cond([[mres_3]]) : !quantum.bit, !quantum.bit

    // // Q2 -> %q14#0
    // // Q1 -> %q14#1

    // // PPR ["Z", "Z", "Y", "Z"](-8) Q2, Q1, Q3, Q0
    // CHECK-INJECT: [[q17:%.+]] = qec.fabricate  magic_conj
    ////// PPM ["Z", "Z", "Y", "Z", "Z"] Q2, Q1, Q3, Q0, conj |m⟩
    // CHECK-INJECT: [[mres_4:%.+]], [[out_4:%.+]]:5 = qec.ppm ["Z", "Z", "Y", "Z", "Z"] [[q14]]#0, [[q14]]#1, [[q9]], [[q4]], [[q17]]
    // CHECK-INJECT: [[q18:%.+]]:4 = qec.ppr ["Z", "Z", "Y", "Z"](4) [[out_4]]#0, [[out_4]]#1, [[out_4]]#2, [[out_4]]#3 cond([[mres_4]])
    // CHECK-INJECT: [[mres_5:%.+]], [[out_5:%.+]] = qec.ppm ["X"] [[out_4]]#4 : !quantum.bit
    // CHECK-INJECT: [[q19:%.+]]:4 = qec.ppr ["Z", "Z", "Y", "Z"](2) [[q18]]#0, [[q18]]#1, [[q18]]#2, [[q18]]#3 cond([[mres_5]]) 

    // // Q2 -> %q19#0
    // // Q1 -> %q19#1
    // // Q3 -> %q19#2
    // // Q0 -> %q19#3

    // // PPM Z, Z, Y, Y -> Q2, Q1, Q0, Q3
    // CHECK-INJECT: [[mres_6:%.+]], [[out_6:%.+]]:4 = qec.ppm ["Z", "Z", "Y", "Y"] [[q19]]#0, [[q19]]#1, [[q19]]#3, [[q19]]#2 

    // // Q2 -> %out_6#0
    // // Q1 -> %out_6#1
    // // Q0 -> %out_6#2
    // // Q3 -> %out_6#3

    // // PPM X, X -> Q1, Q0
    // CHECK-INJECT: [[mres_7:%.+]], [[out_7:%.+]]:2 = qec.ppm ["X", "X"] [[out_6]]#1, [[out_6]]#2 : !quantum.bit, !quantum.bit

    // // Q0 -> %out_7#1

    // // PPM Z(-1) -> Q2
    // CHECK-INJECT: [[mres_8:%.+]], [[out_8:%.+]] = qec.ppm ["Z"](-1) [[out_6]]#0 : !quantum.bit

    // // PPM X, X -> Q0, Q3
    // CHECK-INJECT: [[mres_9:%.+]], [[out_9:%.+]]:2 = qec.ppm ["X", "X"] [[out_7]]#1, [[out_6]]#3 : !quantum.bit, !quantum.bit


    /////// Auto-Corrected Method ///////


    // // Q0 -> %arg0, Q1 -> %arg1, Q2 -> %arg2, Q3 -> %arg3

    // // PPR ["Z"](8) Q0 

    // CHECK-AUTO: [[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit, [[Q2:%.+]]: !quantum.bit, [[Q3:%.+]]: !quantum.bit) {
    // CHECK-AUTO: [[Q_3:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK-AUTO: [[Q_4:%.+]] = qec.fabricate  magic
    // CHECK-AUTO: [[M:%.+]], [[OUT:%.+]]:2 = qec.ppm ["Z", "Z"] [[Q0]], [[Q_4]] : !quantum.bit, !quantum.bit
    // CHECK-AUTO: [[M_0:%.+]], [[OUT_1:%.+]]:2 = qec.ppm ["Z", "Y"] [[OUT]]#1, [[Q_3]] : !quantum.bit, !quantum.bit
    // CHECK-AUTO: [[M_2:%.+]], [[OUT_3:%.+]] = qec.ppm ["X"] [[OUT_1]]#0 : !quantum.bit
    // CHECK-AUTO: [[M_4:%.+]], [[OUT_5:%.+]] = qec.select.ppm([[M]], ["X"], ["Z"]) [[OUT_1]]#1 : !quantum.bit
    // CHECK-AUTO: [[Q_5:%.+]] = arith.xori [[M_0]], [[M_2]] : i1
    // CHECK-AUTO: [[Q_6:%.+]] = qec.ppr ["Z"](2) [[OUT]]#0 cond([[Q_5]]) : !quantum.bit

    // // Q0 -> Q_6

    // // PPR ["Y"](-8) Q3

    // CHECK-AUTO: [[Q_10:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK-AUTO: [[Q_11:%.+]] = qec.fabricate  magic_conj
    // CHECK-AUTO: [[M_6:%.+]], [[OUT_7:%.+]]:2 = qec.ppm ["Y", "Z"] [[Q3]], [[Q_11]] : !quantum.bit, !quantum.bit
    // CHECK-AUTO: [[M_8:%.+]], [[OUT_9:%.+]]:2 = qec.ppm ["Z", "Y"] [[OUT_7]]#1, [[Q_10]] : !quantum.bit, !quantum.bit
    // CHECK-AUTO: [[M_10:%.+]], [[OUT_11:%.+]] = qec.ppm ["X"] [[OUT_9]]#0 : !quantum.bit
    // CHECK-AUTO: [[M_12:%.+]], [[OUT_13:%.+]] = qec.select.ppm([[M_6]], ["X"], ["Z"]) [[OUT_9]]#1 : !quantum.bit
    // CHECK-AUTO: [[Q_12:%.+]] = arith.xori [[M_8]], [[M_10]] : i1
    // CHECK-AUTO: [[Q_13:%.+]] = qec.ppr ["Y"](2) [[OUT_7]]#0 cond([[Q_12]]) : !quantum.bit

    // // Q3 -> Q_13

    // // PPR ["Y", "X"](8) Q2, Q1

    // CHECK-AUTO: [[Q_17:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK-AUTO: [[Q_18:%.+]] = qec.fabricate  magic
    ///// PPM ["Y", "X", "Z"] Q2, Q1, conj |m⟩
    // CHECK-AUTO: [[M_14:%.+]], [[OUT_15:%.+]]:3 = qec.ppm ["Y", "X", "Z"] [[Q2]], [[Q1]], [[Q_18]]
    // CHECK-AUTO: [[M_16:%.+]], [[OUT_17:%.+]]:2 = qec.ppm ["Z", "Y"] [[OUT_15]]#2, [[Q_17]] : !quantum.bit, !quantum.bit
    // CHECK-AUTO: [[M_18:%.+]], [[OUT_19:%.+]] = qec.ppm ["X"] [[OUT_17]]#0 : !quantum.bit
    // CHECK-AUTO: [[M_20:%.+]], [[OUT_21:%.+]] = qec.select.ppm([[M_14]], ["X"], ["Z"]) [[OUT_17]]#1 : !quantum.bit
    // CHECK-AUTO: [[Q_19:%.+]] = arith.xori [[M_16]], [[M_18]] : i1
    // CHECK-AUTO: [[Q_20:%.+]]:2 = qec.ppr ["Y", "X"](2) [[OUT_15]]#0, [[OUT_15]]#1 cond([[Q_19]]) : !quantum.bit, !quantum.bit

    // // Q2 -> Q_20#0
    // // Q1 -> Q_20#1

    // // PPR ["Z", "Z", "Y", "Z"](-8) Q2, Q1, Q3, Q0


    // CHECK-AUTO: [[Q_24:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK-AUTO: [[Q_25:%.+]] = qec.fabricate  magic_conj
    ///// PPM ["Z", "Z", "Y", "Z", "Z"] Q2, Q1, Q3, Q0, conj |m⟩
    // CHECK-AUTO: [[M_22:%.+]], [[OUT_23:%.+]]:5 = qec.ppm ["Z", "Z", "Y", "Z", "Z"] [[Q_20]]#0, [[Q_20]]#1, [[Q_13]], [[Q_6]], [[Q_25]]
    // CHECK-AUTO: [[M_24:%.+]], [[OUT_25:%.+]]:2 = qec.ppm ["Z", "Y"] [[OUT_23]]#4, [[Q_24]] : !quantum.bit, !quantum.bit
    // CHECK-AUTO: [[M_26:%.+]], [[OUT_27:%.+]] = qec.ppm ["X"] [[OUT_25]]#0 : !quantum.bit
    // CHECK-AUTO: [[M_28:%.+]], [[OUT_29:%.+]] = qec.select.ppm([[M_22]], ["X"], ["Z"]) [[OUT_25]]#1 : !quantum.bit
    // CHECK-AUTO: [[Q_26:%.+]] = arith.xori [[M_24]], [[M_26]] : i1
    // CHECK-AUTO: [[Q_27:%.+]]:4 = qec.ppr ["Z", "Z", "Y", "Z"](2) [[OUT_23]]#0, [[OUT_23]]#1, [[OUT_23]]#2, [[OUT_23]]#3 cond([[Q_26]]) 


    // // Q2 -> Q_27#0
    // // Q1 -> Q_27#1
    // // Q3 -> Q_27#2
    // // Q0 -> Q_27#3

    // // PPM Z, Z, Y, Y -> Q2, Q1, Q0, Q3

    // CHECK-AUTO: [[M_30:%.+]], [[OUT_31:%.+]]:4 = qec.ppm ["Z", "Z", "Y", "Y"] [[Q_27]]#0, [[Q_27]]#1, [[Q_27]]#3, [[Q_27]]#2
    // CHECK-AUTO: [[M_32:%.+]], [[OUT_33:%.+]]:2 = qec.ppm ["X", "X"] [[OUT_31]]#1, [[OUT_31]]#2
    // CHECK-AUTO: [[M_34:%.+]], [[OUT_35:%.+]] = qec.ppm ["Z"](-1) [[OUT_31]]#0 : !quantum.bit
    // CHECK-AUTO: [[M_36:%.+]], [[OUT_37:%.+]]:2 = qec.ppm ["X", "X"] [[OUT_33]]#1, [[OUT_31]]#3 : !quantum.bit, !quantum.bit

}
