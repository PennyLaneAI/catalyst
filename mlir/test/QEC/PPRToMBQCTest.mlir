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

// RUN: quantum-opt --ppr-to-mbqc --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_ppr_to_mbqc_0(%q1 : !quantum.bit) -> (i1, i1, i1) {

    // CHECK: func.func @test_ppr_to_mbqc_0([[q1:%.+]]: !quantum.bit)
    // CHECK: [[q1_0:%.+]] = quantum.custom "Hadamard"() [[q1]]
    // CHECK: [[pi:%.+]] = arith.constant 3.14
    // CHECK: [[q1_1:%.+]] = quantum.custom "RZ"([[pi]]) [[q1_0]]
    // CHECK: [[q1_2:%.+]] = quantum.custom "Hadamard"() [[q1_1]]
    %0 = qec.ppr ["X"] (2) %q1 : !quantum.bit

    // CHECK: [[q1_3:%.+]] = quantum.custom "RZ"([[pi]]) [[q1_2]]
    %1 = qec.ppr ["Z"] (2) %0 : !quantum.bit

    // CHECK: [[a1:%.+]] = arith.constant 1.13
    // CHECK: [[a2:%.+]] = arith.constant 0.00
    // CHECK: [[a3:%.+]] = arith.constant 2.80
    // CHECK: [[q1_4:%.+]] = quantum.custom "RotXZX"([[a1]], [[a2]], [[a3]]) [[q1_3]]
    // CHECK: [[q1_5:%.+]] = quantum.custom "RZ"([[pi]]) [[q1_4]]
    // CHECK: [[a4:%.+]] = arith.constant 1.23
    // CHECK: [[a5:%.+]] = arith.constant 9.75
    // CHECK: [[q1_6:%.+]] = quantum.custom "RotXZX"([[a4]], [[a2]], [[a5]]) [[q1_5]]
    %2 = qec.ppr ["Y"] (2) %1 : !quantum.bit

    // CHECK: [[q1_7:%.+]] = quantum.custom "Hadamard"() [[q1_6]]
    // CHECK: [[pi2:%.+]] = arith.constant 1.57
    // CHECK: [[q1_8:%.+]] = quantum.custom "RZ"([[pi2]]) [[q1_7]]
    // CHECK: [[q1_9:%.+]] = quantum.custom "Hadamard"() [[q1_8]]
    %3 = qec.ppr ["X"] (4) %2 : !quantum.bit

    // CHECK: [[q1_10:%.+]] = quantum.custom "RZ"([[pi2]]) [[q1_9]]
    %4 = qec.ppr ["Z"] (4) %3 : !quantum.bit

    // CHECK: [[q1_11:%.+]] = quantum.custom "RotXZX"([[a1]], [[a2]], [[a3]]) [[q1_10]]
    // CHECK: [[q1_12:%.+]] = quantum.custom "RZ"([[pi2]]) [[q1_11]]
    // CHECK: [[q1_13:%.+]] = quantum.custom "RotXZX"([[a4]], [[a2]], [[a5]]) [[q1_12]]
    %5 = qec.ppr ["Y"] (4) %4 : !quantum.bit

    // CHECK: [[q1_14:%.+]] = quantum.custom "Hadamard"() [[q1_13]]
    // CHECK: [[pi4:%.+]] = arith.constant 0.785
    // CHECK: [[q1_15:%.+]] = quantum.custom "RZ"([[pi4]]) [[q1_14]]
    // CHECK: [[q1_16:%.+]] = quantum.custom "Hadamard"() [[q1_15]]
    %6 = qec.ppr ["X"] (8) %5 : !quantum.bit

    // CHECK: [[q1_17:%.+]] = quantum.custom "RZ"([[pi4]]) [[q1_16]]
    %7 = qec.ppr ["Z"] (8) %6 : !quantum.bit

    // CHECK: [[q1_18:%.+]] = quantum.custom "RotXZX"([[a1]], [[a2]], [[a3]]) [[q1_17]]
    // CHECK: [[q1_19:%.+]] = quantum.custom "RZ"([[pi4]]) [[q1_18]]
    // CHECK: [[q1_20:%.+]] = quantum.custom "RotXZX"([[a4]], [[a2]], [[a5]]) [[q1_19]]
    %8 = qec.ppr ["Y"] (8) %7 : !quantum.bit

    // CHECK: [[q1_21:%.+]] = quantum.custom "Hadamard"() [[q1_20]]
    // CHECK: [[m1:%.+]], [[q1_22:%.+]] = quantum.measure [[q1_21]]
    // CHECK: [[q1_23:%.+]] = quantum.custom "Hadamard"() [[q1_22]]
    %m1, %9 = qec.ppm ["X"] %8 : i1, !quantum.bit

    // CHECK: [[m2:%.+]], [[q1_23:%.+]] = quantum.measure [[q1_22]]
    %m2, %10 = qec.ppm ["Z"] %9 : i1, !quantum.bit
    // CHECK: [[q1_24:%.+]] = quantum.custom "RotXZX"([[a1]], [[a2]], [[a3]]) [[q1_23]]
    // CHECK: [[m3:%.+]], [[q1_25:%.+]] = quantum.measure [[q1_24]]
    %m3, %11 = qec.ppm ["Y"] %10 : i1, !quantum.bit

    // CHECK: [[m1]], [[m2]], [[m3]] : i1, i1, i1
    func.return %m1, %m2, %m3 : i1, i1, i1
}

// -----

func.func @test_ppr_to_mbqc_2(%q0: !quantum.bit, %q1: !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {

    // CHECK: func.func @test_ppr_to_mbqc_2([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit)
    // CHECK: [[q0_0:%.+]] = quantum.custom "Hadamard"() [[q0]]

    // CNOT(1, 0)
    // CHECK: [[q0_1:%.+]]:2 = quantum.custom "CNOT"() [[q1]], [[q0_0]] : !quantum.bit, !quantum.bit
    
    // RZ(pi / 4) (qubit 0)
    // CHECK: [[a1:%.+]] = arith.constant 0.78 
    // CHECK: [[q0_2:%.+]] = quantum.custom "RZ"([[a1]]) [[q0_1]]#1

    // CNOT(1, 0)
    // CHECK: [[q0_3:%.+]]:2 = quantum.custom "CNOT"() [[q0_1]]#0, [[q0_2]] : !quantum.bit, !quantum.bit
    // H (qubit 0)
    // CHECK: [[q0_4:%.+]] = quantum.custom "Hadamard"() [[q0_3]]#1
    %0, %1 = qec.ppr ["X", "Z"](8) %q0, %q1 : !quantum.bit, !quantum.bit

    // H (qubit 0)
    // H (qubit 1)
    // CNOT(1, 0)
    // M(↗) (qubit 0)
    // CNOT(1, 0)
    // H (qubit 0)
    // H (qubit 1)
    // CHECK: [[q0_5:%.+]] = quantum.custom "Hadamard"() [[q0_4]]
    // CHECK: [[q0_6:%.+]] = quantum.custom "Hadamard"() [[q0_3]]#0
    // CHECK: [[q0_7:%.+]]:2 = quantum.custom "CNOT"() [[q0_6]], [[q0_5]] : !quantum.bit, !quantum.bit
    // CHECK: [[m:%.+]], [[q0_8:%.+]] = quantum.measure [[q0_7]]#1
    // CHECK: [[q0_9:%.+]]:2 = quantum.custom "CNOT"() [[q0_7]]#0, [[q0_8]] : !quantum.bit, !quantum.bit
    // CHECK: [[q0_10:%.+]] = quantum.custom "Hadamard"() [[q0_9]]#1
    // CHECK: [[q0_11:%.+]] = quantum.custom "Hadamard"() [[q0_9]]#0
    %m, %2, %3 = qec.ppm ["X", "X"](8) %0, %1 : i1, !quantum.bit, !quantum.bit

    func.return %m, %2, %3 : i1, !quantum.bit, !quantum.bit
}

// -----

func.func @test_ppr_to_mbqc_3(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {

    // CHECK: func.func @test_ppr_to_mbqc_3([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit, [[q2:%.+]]: !quantum.bit, [[q3:%.+]]: !quantum.bit)
    // H (qubit 0)
    // I (qubit 1) skip
    // RotXZX (qubit 2)
    // H (qubit 3)

    // CHECK: [[q0_0:%.+]] = quantum.custom "Hadamard"() [[q0]]
    // CHECK: [[a1:%.+]] = arith.constant 1.13
    // CHECK: [[a2:%.+]] = arith.constant 0.00
    // CHECK: [[a3:%.+]] = arith.constant 2.80
    // CHECK: [[q0_1:%.+]] = quantum.custom "RotXZX"([[a1]], [[a2]], [[a3]]) [[q2]]
    // CHECK: [[q0_2:%.+]] = quantum.custom "Hadamard"() [[q3]]

    // CNOT(3, 2)
    // CNOT(2, 1)
    // CNOT(1, 0)

    // CHECK: [[q0_3:%.+]]:2 = quantum.custom "CNOT"() [[q0_2]], [[q0_1]]
    // CHECK: [[q0_4:%.+]]:2 = quantum.custom "CNOT"() [[q0_3]]#1, [[q1]]
    // CHECK: [[q0_5:%.+]]:2 = quantum.custom "CNOT"() [[q0_4]]#1, [[q0_0]]

    // RZ(pi / 2) (qubit 0)
    // CHECK: [[a4:%.+]] = arith.constant 1.57
    // CHECK: [[q0_6:%.+]] = quantum.custom "RZ"([[a4]]) [[q0_5]]#1

    // CNOT(1, 0)
    // CNOT(2, 1)
    // CNOT(3, 2)

    // CHECK: [[q0_7:%.+]]:2 = quantum.custom "CNOT"() [[q0_5]]#0, [[q0_6]]
    // CHECK: [[q0_8:%.+]]:2 = quantum.custom "CNOT"() [[q0_4]]#0, [[q0_7]]#0
    // CHECK: [[q0_9:%.+]]:2 = quantum.custom "CNOT"() [[q0_3]]#0, [[q0_8]]#0

    // H (qubit 0)
    // I (qubit 1) skip
    // RotXZX (qubit 2)
    // H (qubit 3)

    // CHECK: [[q0_10:%.+]] = quantum.custom "Hadamard"() [[q0_7]]#1
    // CHECK: [[a5:%.+]] = arith.constant 1.23
    // CHECK: [[a6:%.+]] = arith.constant 9.75
    // CHECK: [[q0_11:%.+]] = quantum.custom "RotXZX"([[a5]], [[a2]], [[a6]]) [[q0_9]]#1
    // CHECK: [[q0_12:%.+]] = quantum.custom "Hadamard"() [[q0_9]]#0
    %0, %1, %2, %3 = qec.ppr ["X", "Z", "Y", "X"](4) %q0, %q1, %q2, %q3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    func.return %0, %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

func.func @test_ppr_to_mbqc_4(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) -> (i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {

    // H (qubit 0)
    // I (qubit 1) skip
    // RotXZX (qubit 2)
    // H (qubit 3)

    // CNOT(3, 2)
    // CNOT(2, 1)
    // CNOT(1, 0)

    // M(↗) (qubit 0)

    // CNOT(1, 0)
    // CNOT(2, 1)
    // CNOT(3, 2)

    // H (qubit 0)
    // I (qubit 1) skip
    // RotXZX (qubit 2)
    // H (qubit 3)

    // CHECK: "Hadamard"
    // CHECK: "Hadamard"
    // CHECK: "Hadamard"
    // CHECK: "Hadamard"
    // CHECK-NOT: "Hadamard"

    // CHECK: "CNOT"
    // CHECK: "CNOT"
    // CHECK: "CNOT"
    // CHECK-NOT: "CNOT"

    // CHECK: quantum.measure

    // CHECK: "CNOT"
    // CHECK: "CNOT"
    // CHECK: "CNOT"
    // CHECK-NOT: "CNOT"

    // CHECK: "Hadamard"
    // CHECK: "Hadamard"
    // CHECK: "Hadamard"
    // CHECK: "Hadamard"
    // CHECK-NOT: "Hadamard"

    %m, %0, %1, %2, %3 = qec.ppm ["X", "X", "X", "X"](16) %q0, %q1, %q2, %q3 : i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    func.return %m, %0, %1, %2, %3 : i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

// Test PPRotationArbitraryOp lowering to MBQC
func.func @test_ppr_arbitrary_to_mbqc(%q0: !quantum.bit, %angle: f64) -> !quantum.bit {
    // CHECK: func.func @test_ppr_arbitrary_to_mbqc([[q0:%.+]]: !quantum.bit, [[angle:%.+]]: f64)
    
    // For single-qubit Z rotation:
    // PPR(theta, Z) = exp(-i * theta * Z)
    // RZ(phi) = exp(-i * phi/2 * Z)
    // Therefore: phi = 2 * theta
    
    // CHECK: [[two:%.+]] = arith.constant 2.0
    // CHECK: [[rz_angle:%.+]] = arith.mulf [[angle]], [[two]]
    // CHECK: [[q0_out:%.+]] = quantum.custom "RZ"([[rz_angle]]) [[q0]]
    %0 = qec.ppr.arbitrary ["Z"](%angle) %q0 : !quantum.bit
    
    func.return %0 : !quantum.bit
}

// -----

// Test PPRotationArbitraryOp with X Pauli lowering to MBQC
func.func @test_ppr_arbitrary_x_to_mbqc(%q0: !quantum.bit, %angle: f64) -> !quantum.bit {
    // CHECK: func.func @test_ppr_arbitrary_x_to_mbqc([[q0:%.+]]: !quantum.bit, [[angle:%.+]]: f64)
    
    // For X Pauli: H • RZ • H
    // CHECK: [[q0_0:%.+]] = quantum.custom "Hadamard"() [[q0]]
    // CHECK: [[two:%.+]] = arith.constant 2.0
    // CHECK: [[rz_angle:%.+]] = arith.mulf [[angle]], [[two]]
    // CHECK: [[q0_1:%.+]] = quantum.custom "RZ"([[rz_angle]]) [[q0_0]]
    // CHECK: [[q0_2:%.+]] = quantum.custom "Hadamard"() [[q0_1]]
    %0 = qec.ppr.arbitrary ["X"](%angle) %q0 : !quantum.bit
    
    func.return %0 : !quantum.bit
}

// -----

// Test multi-qubit PPRotationArbitraryOp lowering to MBQC
func.func @test_ppr_arbitrary_multi_to_mbqc(%q0: !quantum.bit, %q1: !quantum.bit, %angle: f64) -> (!quantum.bit, !quantum.bit) {
    // CHECK: func.func @test_ppr_arbitrary_multi_to_mbqc([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit, [[angle:%.+]]: f64)
    
    // For X ⊗ Z Pauli:
    // H(q0) • CNOT(q1, q0) • RZ(2*angle) • CNOT(q1, q0) • H(q0)
    
    // CHECK: [[q0_0:%.+]] = quantum.custom "Hadamard"() [[q0]]
    // CHECK: [[cnot1:%.+]]:2 = quantum.custom "CNOT"() [[q1]], [[q0_0]]
    // CHECK: [[two:%.+]] = arith.constant 2.0
    // CHECK: [[rz_angle:%.+]] = arith.mulf [[angle]], [[two]]
    // CHECK: [[q0_1:%.+]] = quantum.custom "RZ"([[rz_angle]]) [[cnot1]]#1
    // CHECK: [[cnot2:%.+]]:2 = quantum.custom "CNOT"() [[cnot1]]#0, [[q0_1]]
    // CHECK: [[q0_2:%.+]] = quantum.custom "Hadamard"() [[cnot2]]#1
    %0, %1 = qec.ppr.arbitrary ["X", "Z"](%angle) %q0, %q1 : !quantum.bit, !quantum.bit
    
    func.return %0, %1 : !quantum.bit, !quantum.bit
}
