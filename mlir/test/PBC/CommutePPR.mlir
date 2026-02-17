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

// RUN: quantum-opt --commute-ppr --split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: quantum-opt --commute-ppr="max-pauli-size=3" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK-MPS

func.func @test_commute_1(%q1 : !quantum.bit){
    
    // Z(pi/4) * Z(pi/8) * Z(pi/4) * Z(pi/4) * Z(pi/8) * Z(pi/8)
    
    // CHECK: [[q1_0:%.+]] = pbc.ppr ["Z"](8) 
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["Z"](8) [[q1_0]]
    // CHECK: [[q1_2:%.+]] = pbc.ppr ["Z"](8) [[q1_1]]
    // CHECK: [[q1_3:%.+]] = pbc.ppr ["Z"](4) [[q1_2]]
    // CHECK: [[q1_4:%.+]] = pbc.ppr ["Z"](4) [[q1_3]]
    // CHECK: [[q1_5:%.+]] = pbc.ppr ["Z"](4) [[q1_4]]
    %0 = pbc.ppr ["Z"](4) %q1 : !quantum.bit
    %1 = pbc.ppr ["Z"](8) %0 : !quantum.bit
    %2 = pbc.ppr ["Z"](4) %1 : !quantum.bit
    %3 = pbc.ppr ["Z"](4) %2 : !quantum.bit
    %4 = pbc.ppr ["Z"](8) %3 : !quantum.bit
    %5 = pbc.ppr ["Z"](8) %4 : !quantum.bit
    func.return
}

// -----

func.func @test_commute_2(%q1 : !quantum.bit, %q2 : !quantum.bit){

    // XX(pi/4) * XX(pi/8) * XX(pi/4) * XX(pi/8) * XX(pi/8)

    // CHECK: [[q1_0:%.+]]:2 = pbc.ppr ["X", "X"](8)
    // CHECK: [[q1_1:%.+]]:2 = pbc.ppr ["X", "X"](8) [[q1_0]]#0, [[q1_0]]#1
    // CHECK: [[q1_2:%.+]]:2 = pbc.ppr ["X", "X"](8) [[q1_1]]#0, [[q1_1]]#1
    // CHECK: [[q1_3:%.+]]:2 = pbc.ppr ["X", "X"](4) [[q1_2]]#0, [[q1_2]]#1
    // CHECK: [[q1_4:%.+]]:2 = pbc.ppr ["X", "X"](4) [[q1_3]]#0, [[q1_3]]#1
    %0:2 = pbc.ppr ["X", "X"](4) %q1, %q2 : !quantum.bit, !quantum.bit
    %1:2 = pbc.ppr ["X", "X"](8) %0#0, %0#1 : !quantum.bit, !quantum.bit
    %2:2 = pbc.ppr ["X", "X"](4) %1#0, %1#1 : !quantum.bit, !quantum.bit
    %3:2 = pbc.ppr ["X", "X"](8) %2#0, %2#1 : !quantum.bit, !quantum.bit
    %4:2 = pbc.ppr ["X", "X"](8) %3#0, %3#1 : !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func @test_commute_3(%q1 : !quantum.bit, %q2 : !quantum.bit){

    // X0(pi/4) * X0X1(pi/4) * X0(pi/8) * X0X1(pi/4) * X0(pi/8) * X1(pi/8) 
    // -> X0(pi/8) * X0(pi/8) * X0(pi/8) * X0(pi/4) * X0X1(pi/4) * X0X1(pi/4)

    // CHECK: [[q1_0:%.+]] = pbc.ppr ["X"](8) %arg0
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["X"](8) [[q1_0]] 
    // CHECK: [[q1_2:%.+]] = pbc.ppr ["X"](8) [[q1_1]]
    // CHECK: [[q1_3:%.+]] = pbc.ppr ["X"](4) [[q1_2]]
    // CHECK: [[q1_4:%.+]]:2 = pbc.ppr ["X", "X"](4) [[q1_3]], %arg1
    // CHECK: [[q1_5:%.+]]:2 = pbc.ppr ["X", "X"](4) [[q1_4]]#0, [[q1_4]]#1
    %a = pbc.ppr ["X"](4) %q1 : !quantum.bit // q1
    %0:2 = pbc.ppr ["X", "X"](4) %a, %q2 : !quantum.bit, !quantum.bit // q1, q2
    %1 = pbc.ppr ["X"](8) %0#0 : !quantum.bit // q1
    %2:2 = pbc.ppr ["X", "X"](4) %1, %0#1 : !quantum.bit, !quantum.bit // q1, q2
    %3 = pbc.ppr ["X"](8) %2#0 : !quantum.bit // q1
    %4 = pbc.ppr ["X"](8) %3 : !quantum.bit // q1
    func.return
}

// -----

func.func @test_anticommute_4(%q1 : !quantum.bit){
    
    // Pauli rules:
    // iXY = -Z
    // iXZ = +Y

    // X(4) * Y(8) * X(4) * Y(8)
    // -> Z(-8) * X(4) * X(4) * Y(8)
    // -> Z(-8) * X(4) * Z(-8) * X(4)
    // -> Z(-8) * Y(-8) * X(4) * X(4)

    // CHECK: [[q1_0:%.+]] = pbc.ppr ["Z"](-8) %arg0
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["Y"](-8) [[q1_0]]
    // CHECK: [[q1_2:%.+]] = pbc.ppr ["X"](4) [[q1_1]]
    // CHECK: [[q1_3:%.+]] = pbc.ppr ["X"](4) [[q1_2]]
    %0 = pbc.ppr ["X"](4) %q1 : !quantum.bit
    %1 = pbc.ppr ["Y"](8) %0 : !quantum.bit
    %2 = pbc.ppr ["X"](4) %1 : !quantum.bit
    %3 = pbc.ppr ["Y"](8) %2 : !quantum.bit
    func.return
}

// -----

func.func @test_anticommute_5(%q1 : !quantum.bit, %q2 : !quantum.bit){

    // XY(4) * Z0(8) * Y0(8)
    // -> YY(8) * XY(4) * Y0(8)
    // -> YY(8) * ZY(-8) * XY(4)

    // CHECK: [[q1_0:%.+]]:2 = pbc.ppr ["Y", "Y"](8) %arg0, %arg1
    // CHECK: [[q1_1:%.+]]:2 = pbc.ppr ["Z", "Y"](-8) [[q1_0]]#0, [[q1_0]]#1
    // CHECK: [[q1_2:%.+]]:2 = pbc.ppr ["X", "Y"](4) [[q1_1]]#0, [[q1_1]]#1
    %0:2 = pbc.ppr ["X", "Y"](4) %q1, %q2 : !quantum.bit, !quantum.bit
    %1 = pbc.ppr ["Z"](8) %0#0 : !quantum.bit
    %2 = pbc.ppr ["Y"](8) %1 : !quantum.bit
    func.return
}

// -----

func.func @test_anticommute_6(%q1 : !quantum.bit, %q2 : !quantum.bit){
    
    // XY commutes with ZZ

    // XY(4) * ZZ(8) * ZY(8)
    // -> ZZ(8) * XY(4) * ZY(8)
    // -> ZZ(8) * Y_(8) * XY(4)

    // CHECK: [[q1_0:%.+]]:2 = pbc.ppr ["Z", "Z"](8) %arg0, %arg1
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["Y"](8) [[q1_0]]#0
    // CHECK: [[q1_2:%.+]]:2 = pbc.ppr ["X", "Y"](4) [[q1_1]], [[q1_0]]#1
    %0:2 = pbc.ppr ["X", "Y"](4) %q1, %q2 : !quantum.bit, !quantum.bit
    %1:2 = pbc.ppr ["Z", "Z"](8) %0#0, %0#1 : !quantum.bit, !quantum.bit
    %2:2 = pbc.ppr ["Z", "Y"](8) %1#0, %0#1 : !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func @test_anticommute_7(%q1 : !quantum.bit, %q2 : !quantum.bit){

    // XY(4) * Z0(8) * X1(8)
    // -> YY(8) * XY(4) * X1(8)
    // -> YY(8) * XZ(8) * XY(4)

    // CHECK: [[q1_0:%.+]]:2 = pbc.ppr ["Y", "Y"](8) %arg0, %arg1
    // CHECK: [[q1_1:%.+]]:2 = pbc.ppr ["X", "Z"](8) [[q1_0]]#0, [[q1_0]]#1
    // CHECK: [[q1_2:%.+]]:2 = pbc.ppr ["X", "Y"](4) [[q1_1]]#0, [[q1_1]]#1
    %0:2 = pbc.ppr ["X", "Y"](4) %q1, %q2 : !quantum.bit, !quantum.bit
    %1 = pbc.ppr ["Z"](8) %0#0 : !quantum.bit
    %2 = pbc.ppr ["X"](8) %0#1 : !quantum.bit
    func.return
}

// -----

func.func @test_anticommute_8(%q : !quantum.reg){

    // X0(4) * ZY(8) * X1(8)
    // -> YY(8) * X1(8) * X0(4)

    // CHECK: [[q1_0:%.+]]:2 = pbc.ppr ["Y", "Y"](8) %0, %1
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["X"](4) [[q1_0]]#0
    // CHECK: [[q1_2:%.+]] = pbc.ppr ["X"](8) [[q1_0]]#1
    %q1 = quantum.extract %q[0] : !quantum.reg -> !quantum.bit
    %0 = pbc.ppr ["X"](4) %q1 : !quantum.bit
    %q2 = quantum.extract %q[1] : !quantum.reg -> !quantum.bit
    %1:2 = pbc.ppr ["Z", "Y"](8) %0, %q2 : !quantum.bit, !quantum.bit
    %2 = pbc.ppr ["X"](8) %1#1 : !quantum.bit
    func.return
}


func.func @test_anticommute_9(%q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit){

    // XYZ(4) * Y1(4) * Z0X1(8) * X3(8)

    // CHECK: [[q1_0:%.+]]:2 = pbc.ppr ["Z", "Z"](8) %arg0, %arg1
    // CHECK: [[q1_1:%.+]]:3 = pbc.ppr ["X", "Y", "Y"](-8) [[q1_0]]#0, [[q1_0]]#1, %arg2
    // CHECK: [[q1_2:%.+]]:3 = pbc.ppr ["X", "Y", "Z"](4) [[q1_1]]#0, [[q1_1]]#1, [[q1_1]]#2
    // CHECK: [[q1_3:%.+]] = pbc.ppr ["Y"](4) [[q1_2]]#1
    %1:3 = pbc.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3 : !quantum.bit, !quantum.bit, !quantum.bit
    %2 = pbc.ppr ["Y"](4) %1#1 : !quantum.bit
    %3:2 = pbc.ppr ["Z", "X"](8) %1#0, %2 : !quantum.bit, !quantum.bit
    %4 = pbc.ppr ["X"](8) %1#2 : !quantum.bit
    func.return
}

// -----

func.func @test_anticommute_10(%q1 : !quantum.bit){

    // Pauli rule:
    // YZ = -ZY

    // Y(2) * Z(8) * Y(2) * Z(8)
    // -> Z(-8) * Y(2) * Y(2) * Z(8)
    // -> Z(-8) * Y(2) * Z(-8) * Y(2)
    // -> Z(-8) * Z(8) * Y(2) * Y(2)

    // CHECK: [[q1_0:%.+]] = pbc.ppr ["Z"](-8) %arg0
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["Z"](8) [[q1_0]]
    // CHECK: [[q1_2:%.+]] = pbc.ppr ["Y"](2) [[q1_1]]
    // CHECK: [[q1_3:%.+]] = pbc.ppr ["Y"](2) [[q1_2]]
    %0 = pbc.ppr ["Y"](2) %q1 : !quantum.bit
    %1 = pbc.ppr ["Z"](8) %0 : !quantum.bit
    %2 = pbc.ppr ["Y"](2) %1 : !quantum.bit
    %3 = pbc.ppr ["Z"](8) %2 : !quantum.bit
    func.return
}

// -----

func.func @test_anticommute_11(%q1 : !quantum.bit, %q2 : !quantum.bit, %q3 : !quantum.bit){

    // Pauli rules:
    //
    // XZ = -ZX
    //
    // P and P' commute if there is an even number of anti-commuting single-qubit pairs.
    // Otherwise, they anti-commute.


    // XYZ(2) * XZY(8) * ZYZ(8)
    // -> XZY(8) * XYZ(2) * ZYZ(8)   // XYZ(2) and XZY(8) commute (XX commute, YZ and ZY anti-commute)
    // -> XZY(8) * ZYZ(-8) * XYZ(2)  // XYZ(2) and ZYZ(8) anti-commute (XZ anti-commutes, YY and ZZ commute)

    // CHECK: [[q1_0:%.+]]:3 = pbc.ppr ["X", "Z", "Y"](8) %arg0, %arg1, %arg2
    // CHECK: [[q1_1:%.+]]:3 = pbc.ppr ["Z", "Y", "Z"](-8) [[q1_0]]#0, [[q1_0]]#1, [[q1_0]]#2
    // CHECK: [[q1_2:%.+]]:3 = pbc.ppr ["X", "Y", "Z"](2) [[q1_1]]#0, [[q1_1]]#1, [[q1_1]]#2
    %0:3 = pbc.ppr ["X", "Y", "Z"](2) %q1, %q2, %q3 : !quantum.bit, !quantum.bit, !quantum.bit
    %1:3 = pbc.ppr ["X", "Z", "Y"](8) %0#0, %0#1, %0#2 : !quantum.bit, !quantum.bit, !quantum.bit
    %2:3 = pbc.ppr ["Z", "Y", "Z"](8) %1#0, %1#1, %1#2 : !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func public @circuit_first_minus(%q0: !quantum.bit, %q1: !quantum.bit) {
    
    // CHECK: pbc.ppr ["Y"](-8)
    // CHECK: pbc.ppr ["X"](-4)
    %0 = pbc.ppr ["X"](-4) %q0 : !quantum.bit
    %1 = pbc.ppr ["Z"](8) %0 : !quantum.bit
    return
}

// -----

func.func public @circuit_second_minus(%q0: !quantum.bit, %q1: !quantum.bit) {

    // CHECK: pbc.ppr ["Y"](-8)
    // CHECK: pbc.ppr ["X"](4)
    %0 = pbc.ppr ["X"](4) %q0 : !quantum.bit
    %1 = pbc.ppr ["Z"](-8) %0 : !quantum.bit
    return
}

// -----

func.func public @circuit_all_minus(%q0: !quantum.bit, %q1: !quantum.bit) {

    // CHECK: pbc.ppr ["Y"](8)
    // CHECK: pbc.ppr ["X"](-4)
    %0 = pbc.ppr ["X"](-4) %q0 : !quantum.bit
    %1 = pbc.ppr ["Z"](-8) %0 : !quantum.bit
    return
}

// -----

func.func public @circuit_transformed() {

    // H(0) * CNOT(0, 1) * T(0)
    // XZ(8) * Z0(4) * X0(4) * Z0(4) * ZX(4) * Z0(-4) * X1(-4)

    // CHECK: [[q1_0:%.+]]:2 = pbc.ppr ["X", "Z"](8) %1, %2
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["Z"](4) [[q1_0]]#0
    // CHECK: [[q1_2:%.+]] = pbc.ppr ["X"](4) [[q1_1]]
    // CHECK: [[q1_3:%.+]] = pbc.ppr ["Z"](4) [[q1_2]]
    // CHECK: [[q1_4:%.+]]:2 = pbc.ppr ["Z", "X"](4) [[q1_3]], [[q1_0]]#1
    // CHECK: [[q1_5:%.+]] = pbc.ppr ["Z"](-4) [[q1_4]]#0
    // CHECK: [[q1_6:%.+]] = pbc.ppr ["X"](-4) [[q1_4]]#1
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = pbc.ppr ["Z"](4) %1 : !quantum.bit
    %3 = pbc.ppr ["X"](4) %2 : !quantum.bit
    %4 = pbc.ppr ["Z"](4) %3 : !quantum.bit
    %5 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %6:2 = pbc.ppr ["Z", "X"](4) %4, %5 : !quantum.bit, !quantum.bit
    %7 = pbc.ppr ["Z"](-4) %6#0 : !quantum.bit
    %8 = pbc.ppr ["X"](-4) %6#1 : !quantum.bit
    %9 = pbc.ppr ["Z"](8) %8 : !quantum.bit
    %10 = quantum.insert %0[ 0], %7 : !quantum.reg, !quantum.bit
    %11 = quantum.insert %10[ 1], %9 : !quantum.reg, !quantum.bit
    return
}

// -----

// The circuit that represents in the Game of Surface Code, Figure 4.
// The result is checked layer-by-layer based on the result in the paper.
func.func public @game_of_surface_code(%q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit, %q4: !quantum.bit) {
    
    // %arg0 = Q1, %arg1 = Q2, %arg2 = Q3, %arg3 = Q4

    // CHECK: quantum.device

    // Z(8) Q1
    // Y(-8) Q4
    // YX(8) Q3, Q2
    // CHECK:     [[Q0_0:%.+]] = pbc.ppr ["Z"](8) %arg0 : !quantum.bit
    // CHECK-DAG: [[Q3_0:%.+]] = pbc.ppr ["Y"](-8) %arg3 : !quantum.bit
    // CHECK-DAG: [[Q2_Q1:%.+]]:2 = pbc.ppr ["Y", "X"](8) %arg2, %arg1 : !quantum.bit, !quantum.bit

    // ZZZY(8) Q2, Q1, Q3, Q0
    // CHECK:     [[Q2_Q1_Q3_Q0:%.+]]:4 = pbc.ppr ["Z", "Z", "Y", "Z"](-8) [[Q2_Q1]]#0, [[Q2_Q1]]#1, [[Q3_0]], [[Q0_0]]

    // CNOT Q1, Q2 (decompose into 3 PPRs)
    // X(-4) Q3
    // CHECK:     [[Q2_Q1:%.+]]:2 = pbc.ppr ["Z", "X"](4) [[Q2_Q1_Q3_Q0]]#0, [[Q2_Q1_Q3_Q0]]#1 : !quantum.bit, !quantum.bit
    // CHECK:     [[Q1:%.+]] = pbc.ppr ["X"](-4) [[Q2_Q1]]#1 : !quantum.bit
    // CHECK-DAG: [[Q2:%.+]] = pbc.ppr ["Z"](-4) [[Q2_Q1]]#0 : !quantum.bit
    // CHECK-DAG: [[Q3:%.+]] = pbc.ppr ["X"](-4) [[Q2_Q1_Q3_Q0]]#2 : !quantum.bit

    // CNOT Q1, Q0 (decompose into 3 PPRs)
    // X(4) Q2
    // CHECK-DAG: [[Q1_Q0:%.+]]:2 = pbc.ppr ["Z", "X"](4) [[Q1]], [[Q2_Q1_Q3_Q0]]#3 : !quantum.bit, !quantum.bit
    // CHECK-DAG: [[Q1:%.+]] = pbc.ppr ["Z"](-4) [[Q1_Q0]]#0 : !quantum.bit
    // CHECK-DAG: [[Q0:%.+]] = pbc.ppr ["X"](-4) [[Q1_Q0]]#1 : !quantum.bit
    // CHECK-DAG: [[Q2_0:%.+]] = pbc.ppr ["X"](4) [[Q2]] : !quantum.bit

    // CNOT Q3, Q0 (decompose into 3 PPRs)
    // CHECK-DAG: [[Q3_Q0:%.+]]:2 = pbc.ppr ["Z", "X"](4) [[Q3]], [[Q0]] : !quantum.bit, !quantum.bit
    // CHECK-DAG: [[Q3:%.+]] = pbc.ppr ["Z"](-4) [[Q3_Q0]]#0 : !quantum.bit
    // CHECK-DAG: [[Q0:%.+]] = pbc.ppr ["X"](-4) [[Q3_Q0]]#1 : !quantum.bit

    // Z(4) Q1
    // Z(4) Q3
    // CHECK-DAG: [[Q1_0:%.+]] = pbc.ppr ["Z"](4) [[Q1]] : !quantum.bit
    // CHECK-DAG: [[Q3_0:%.+]] = pbc.ppr ["Z"](4) [[Q3]] : !quantum.bit

    // X(-4) Q0
    // X(4) Q1
    // X(4) Q2
    // X(4) Q3
    // CHECK: [[Q0_2:%.+]] = pbc.ppr ["X"](-4) [[Q0]] : !quantum.bit
    // CHECK: [[Q1_2:%.+]] = pbc.ppr ["X"](4) [[Q1_0]] : !quantum.bit
    // CHECK: [[Q2_2:%.+]] = pbc.ppr ["X"](4) [[Q2_0]] : !quantum.bit
    // CHECK: [[Q3_2:%.+]] = pbc.ppr ["X"](4) [[Q3_0]] : !quantum.bit

    // CHECK: quantum.device_release

    // Because the Pauli size is limited to 3, the two operations below are not commuted.
    // CHECK-MPS: pbc.ppr ["Z", "X"](4)
    // CHECK-MPS: pbc.ppr ["Y", "Y", "Z"](8)
    // CHECK-MPS-NOT: pbc.ppr ["Z", "Z", "Y", "Z"](-8)

    %c1_i64 = arith.constant 1 : i64
    quantum.device shots(%c1_i64) ["../runtime/build/lib/librtd_null_qubit.dylib", "NullQubit", ""]

    %0 = pbc.ppr ["Z"](8) %q1 : !quantum.bit // q1

    %1:2 = pbc.ppr["Z", "X"](4) %q3, %q2 : !quantum.bit, !quantum.bit // q3, q2
    %a1 = pbc.ppr["Z"](-4) %1#0 : !quantum.bit
    %b1 = pbc.ppr["X"](-4) %1#1 : !quantum.bit

    %2 = pbc.ppr ["X"](-4) %q4 : !quantum.bit // q4

    // CNOT(q2, q1) * X3(4) * Z4(8)
    %3:2 = pbc.ppr["Z", "X"](4) %b1, %0 : !quantum.bit, !quantum.bit  // q2, q1
    %a3 = pbc.ppr["Z"](-4) %3#0 : !quantum.bit
    %b3 = pbc.ppr["X"](-4) %3#1 : !quantum.bit

    %4 = pbc.ppr ["X"](4) %a1 : !quantum.bit // q3
    %5 = pbc.ppr ["Z"](8) %2 : !quantum.bit // q4

    // CNOT(q4, q1)
    %6:2 = pbc.ppr["Z", "X"](4) %5, %b3 : !quantum.bit, !quantum.bit // q4, q1
    %a6 = pbc.ppr["Z"](-4) %6#0 : !quantum.bit
    %b6 = pbc.ppr["X"](-4) %6#1 : !quantum.bit
    
    // Z1(8) * Z2(4) * Z3(8) * Z4(4)
    %7 = pbc.ppr ["Z"](8) %b6 : !quantum.bit   // q1
    %8 = pbc.ppr ["Z"](4) %a3 : !quantum.bit   // q2
    %9 = pbc.ppr ["Z"](8) %4 : !quantum.bit     // q3
    %10 = pbc.ppr ["Z"](4) %a6 : !quantum.bit  // q4

    // X1(4) * X2(4) * X3(4) * X4(4)
    %11 = pbc.ppr ["X"](-4) %7 : !quantum.bit   // q1
    %12 = pbc.ppr ["X"](4) %8 : !quantum.bit    // q2
    %13 = pbc.ppr ["X"](4) %9 : !quantum.bit    // q3
    %14 = pbc.ppr ["X"](4) %10 : !quantum.bit   // q4

    quantum.device_release

    func.return
}  
