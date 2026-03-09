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

// RUN: quantum-opt --partition-layers --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @test_partition_layers_0(%qr0 : !quantum.bit, %qr1 : !quantum.bit, %qr2 : !quantum.bit) -> i1 {
    
   // CHECK:func.func @test_partition_layers_0([[qr0:%.+]]: !quantum.bit, [[qr1:%.+]]: !quantum.bit, [[qr2:%.+]]: !quantum.bit) -> i1 {
   // CHECK:  [[QB:%.+]]:4 = pbc.layer([[A0:%.+]] = [[qr0]], [[A1:%.+]] = [[qr1]], [[A2:%.+]] = [[qr2]]) : !quantum.bit, !quantum.bit, !quantum.bit {
   // CHECK:    [[M:%.+]], [[O:%.+]]:3 = pbc.ppm ["X", "Y", "Z"](8) [[A0]], [[A1]], [[A2]] : i1, !quantum.bit, !quantum.bit, !quantum.bit
   // CHECK:    pbc.yield [[M]], [[O]]#0, [[O]]#1, [[O]]#2 : i1, !quantum.bit, !quantum.bit, !quantum.bit
   // CHECK:  }
   // CHECK:  return [[QB]]#0 : i1

    // X Y Z (0, 1, 2) pi/8
    %m1, %0:3 = pbc.ppm ["X", "Y", "Z"] (8) %qr0, %qr1, %qr2 : i1, !quantum.bit, !quantum.bit, !quantum.bit
    
    func.return %m1 : i1
}

// -----

func.func @test_partition_layers_1(%qr0 : !quantum.bit, %qr1 : !quantum.bit, %qr2 : !quantum.bit, %qr3 : !quantum.bit) -> i1 {

    // CHECK:test_partition_layers_1([[qr0:%.+]]: !quantum.bit, [[qr1:%.+]]: !quantum.bit, [[qr2:%.+]]: !quantum.bit, [[qr3:%.+]]: !quantum.bit)
    
    // Layer 1: Two ops are not commutes and they act on disjoint qubits, so they can be partitioned into two layers.
    // CHECK: [[Q0:%.+]]:4 = pbc.layer([[A0:%.+]] = [[qr0]], [[A1:%.+]] = [[qr1]], [[A2:%.+]] = [[qr2]], [[A3:%.+]] = [[qr3]])
    // CHECK:       [[QL0:%.+]]:2 = pbc.ppr ["X", "Z"](-8) [[A0]], [[A1]]
    // CHECK:       [[QL1:%.+]]:2 = pbc.ppr ["Y", "Z"](8) [[A2]], [[A3]]
    // CHECK:  pbc.yield [[QL0]]#0, [[QL0]]#1, [[QL1]]#0, [[QL1]]#1
    %0:2 = pbc.ppr ["X", "Z"] (-8) %qr0, %qr1 : !quantum.bit, !quantum.bit // X Z (0, 1) pi/8 
    %1:2 = pbc.ppr ["Y", "Z"] (8) %qr2, %qr3 : !quantum.bit, !quantum.bit // Y Z (2, 3) pi/8

    // CHECK: [[Q1:%.+]]:4 = pbc.layer([[A0:%.+]] = [[Q0]]#3, [[A1:%.+]] = [[Q0]]#0, [[A2:%.+]] = [[Q0]]#1, [[A3:%.+]] = [[Q0]]#2)
    // CHECK:   [[QL2:%.+]] = pbc.ppr ["Y"](8) [[A0]]
    // CHECK:   [[QL3:%.+]] = pbc.ppr ["Z"](8) [[A1]]
    // CHECK:   [[QL4:%.+]]:2 = pbc.ppr ["X", "X"](8) [[A2]], [[A3]]
    // CHECK:   pbc.yield [[QL2]], [[QL3]], [[QL4]]#0, [[QL4]]#1

    %2 = pbc.ppr ["Y"] (8) %1#1 : !quantum.bit // Y (3) pi/8
    %3 = pbc.ppr ["Z"] (8) %0#0 : !quantum.bit // Z (0) pi/8
    %4:2 = pbc.ppr ["X", "X"] (8) %0#1, %1#0 : !quantum.bit, !quantum.bit // X X (1, 2) pi/8

    // CHECK: [[Q2:%.+]]:6 = pbc.layer([[A0:%.+]] = [[Q1]]#0, [[A1:%.+]] = [[Q1]]#1, [[A2:%.+]] = [[Q1]]#2, [[A3:%.+]] = [[Q1]]#3)
    // CHECK:   [[QL5:%.+]] = pbc.ppr ["Z"](8) [[A0]] : !quantum.bit
    // CHECK:   [[M0:%.+]], [[QL7:%.+]]:3 = pbc.ppm ["X", "Y", "Z"](8) [[A1]], [[A2]], [[A3]]
    // CHECK:   [[M1:%.+]], [[QL9:%.+]] = pbc.ppm ["Z"](8) [[QL5]] : i1, !quantum.bit
    // CHECK:   pbc.yield [[M0]], [[M1]], [[QL9]], [[QL7]]#0, [[QL7]]#1, [[QL7]]#2 : i1, i1, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    
    %5 = pbc.ppr ["Z"] (8) %2 : !quantum.bit // Z (3) pi/8
    %6:4 = pbc.ppm ["X", "Y", "Z"] (8) %3, %4#0, %4#1 : i1, !quantum.bit, !quantum.bit, !quantum.bit // X Y Z (0, 1, 2) pi/8
    %m, %7 = pbc.ppm ["Z"](8) %5 : i1, !quantum.bit // Z (3) pi/8

    // CHECK: return [[Q2]]#1 : i1
    func.return %m : i1
}

// -----

func.func @test_partition_layers_2(%q0: !quantum.bit, %q1: !quantum.bit) -> (i1, !quantum.bit, !quantum.bit) {
    // CHECK: func.func @test_partition_layers_2([[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit)
    // CHECK:   [[C0:%.+]] = arith.constant 0 : index
    // CHECK:   [[C1:%.+]] = arith.constant 10 : index
    // CHECK:   [[C2:%.+]] = arith.constant 1 : index
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index

    // CHECK:   [[Q0:%.+]]:2 = pbc.layer([[A0:%.+]] = [[q0]], [[A1:%.+]] = [[q1]]) : !quantum.bit, !quantum.bit {
    // CHECK:     [[QL0:%.+]]:2 = pbc.ppr ["X", "Z"](-8) [[A0]], [[A1]] : !quantum.bit, !quantum.bit
    // CHECK:     pbc.yield [[QL0]]#0, [[QL0]]#1 : !quantum.bit, !quantum.bit
    %0:2 = pbc.ppr ["X", "Z"] (-8) %q0, %q1 : !quantum.bit, !quantum.bit

    // CHECK:   [[C3:%.+]] = arith.constant false
    %m = arith.constant 0 : i1

  // CHECK:   [[Q1:%.+]]:3 = scf.for %arg2 = [[C0]] to [[C1]] step [[C2]] iter_args([[A0:%.+]] = [[C3]], [[A1:%.+]] = [[Q0]]#0, [[A2:%.+]] = [[Q0]]#1)
    %qq:3 = scf.for %i = %start to %stop step %step iter_args(%m_0 = %m, %q_0 = %0#0, %q_1 = %0#1) -> (i1, !quantum.bit, !quantum.bit) {
        
        // CHECK: [[QL1:%.+]]:2 = pbc.layer([[A00:%.+]] = [[A1]], [[A11:%.+]] = [[A2]])
        // CHECK:   [[Q01:%.+]]:2 = pbc.ppr ["X", "Z"](-8) [[A00]], [[A11]] : !quantum.bit, !quantum.bit
        // CHECK:   [[Q02:%.+]] = pbc.ppr ["X"](8) [[Q01]]#0 : !quantum.bit
        // CHECK:   pbc.yield [[Q02]], [[Q01]]#1 : !quantum.bit, !quantum.bit
        %q_2:2 = pbc.ppr ["X", "Z"] (-8) %q_0, %q_1 : !quantum.bit, !quantum.bit
        %single_qubit = pbc.ppr ["X"] (8) %q_2#0 : !quantum.bit

        // CHECK: [[QL2:%.+]]:3 = pbc.layer([[A00:%.+]] = [[QL1]]#1, [[A01:%.+]] = [[QL1]]#0)
        // CHECK:   [[M:%.+]], [[O:%.+]]:2 = pbc.ppm ["X", "X"](8) [[A00]], [[A01]] : i1, !quantum.bit, !quantum.bit
        // CHECK:   pbc.yield [[M]], [[O]]#0, [[O]]#1 : i1, !quantum.bit, !quantum.bit
        %q_3:3 = pbc.ppm ["X", "X"] (8) %q_2#1, %single_qubit : i1, !quantum.bit, !quantum.bit
        
        // CHECK: [[M1:%.+]] = arith.addi [[A0]], [[QL2]]#0 : i1
        // CHECK: scf.yield [[M1]], [[QL2]]#1, [[QL2]]#2 : i1, !quantum.bit, !quantum.bit
        %m_1 = arith.addi %m_0, %q_3#0 : i1 // update the variable
        scf.yield %m_1, %q_3#1, %q_3#2 : i1, !quantum.bit, !quantum.bit
    }

    // CHECK:   return [[Q1]]#0, [[Q1]]#1, [[Q1]]#2 : i1, !quantum.bit, !quantum.bit
    func.return %qq#0, %qq#1, %qq#2 : i1, !quantum.bit, !quantum.bit
}

// -----

func.func @test_partition_layers_3(%qr0 : !quantum.bit, %qr1 : !quantum.bit, %qr2 : !quantum.bit, %qr3 : !quantum.bit) {
    // CHECK: func.func @test_partition_layers_3([[qr0:%.+]]: !quantum.bit, [[qr1:%.+]]: !quantum.bit, [[qr2:%.+]]: !quantum.bit, [[qr3:%.+]]: !quantum.bit)

    // Layer 1
    // CHECK: [[Q0:%.+]]:4 = pbc.layer([[A0:%.+]] = [[qr0]], [[A1:%.+]] = [[qr1]], [[A2:%.+]] = [[qr2]], [[A3:%.+]] = [[qr3]])
    // CHECK:   [[QL0:%.+]]:4 = pbc.ppr ["I", "Z", "I", "I"](-8) [[A0]], [[A1]], [[A2]], [[A3]]
    // CHECK:   pbc.yield [[QL0]]#0, [[QL0]]#1, [[QL0]]#2, [[QL0]]#3
    %0:4 = pbc.ppr ["I", "Z", "I", "I"] (-8) %qr0, %qr1, %qr2, %qr3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    // Layer 2
    // CHECK: [[Q1:%.+]]:4 = pbc.layer([[A0:%.+]] = [[Q0]]#0, [[A1:%.+]] = [[Q0]]#1, [[A2:%.+]] = [[Q0]]#2, [[A3:%.+]] = [[Q0]]#3)
    // CHECK:   [[QL1:%.+]]:4 = pbc.ppr ["X", "Y", "Z", "Y"](8) [[A0]], [[A1]], [[A2]], [[A3]]
    // CHECK:   [[QL2:%.+]]:4 = pbc.ppr ["X", "I", "Y", "Z"](8) [[QL1]]#0, [[QL1]]#1, [[QL1]]#2, [[QL1]]#3
    // CHECK:   pbc.yield [[QL2]]#0, [[QL2]]#1, [[QL2]]#2, [[QL2]]#3
    %1:4 = pbc.ppr ["X", "Y", "Z", "Y"] (8) %0#0, %0#1, %0#2, %0#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %2:4 = pbc.ppr ["X", "I", "Y", "Z"] (8) %1#0, %1#1, %1#2, %1#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    // Layer 3
    // CHECK: [[Q2:%.+]]:4 = pbc.layer([[A0:%.+]] = [[Q1]]#0, [[A1:%.+]] = [[Q1]]#1, [[A2:%.+]] = [[Q1]]#2, [[A3:%.+]] = [[Q1]]#3)
    // CHECK:   [[QL3:%.+]]:4 = pbc.ppr ["X", "Z", "I", "X"](-8) [[A0]], [[A1]], [[A2]], [[A3]]
    // CHECK:   [[QL4:%.+]]:4 = pbc.ppr ["X", "Z", "Y", "I"](8) [[QL3]]#0, [[QL3]]#1, [[QL3]]#2, [[QL3]]#3
    // CHECK:   pbc.yield [[QL4]]#0, [[QL4]]#1, [[QL4]]#2, [[QL4]]#3
    %3:4 = pbc.ppr ["X", "Z", "I", "X"] (-8) %2#0, %2#1, %2#2, %2#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %4:4 = pbc.ppr ["X", "Z", "Y", "I"] (8) %3#0, %3#1, %3#2, %3#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    // Layer 4
    // CHECK: [[Q3:%.+]]:4 = pbc.layer([[A0:%.+]] = [[Q2]]#0, [[A1:%.+]] = [[Q2]]#1, [[A2:%.+]] = [[Q2]]#2, [[A3:%.+]] = [[Q2]]#3)
    // CHECK:   [[QL5:%.+]]:4 = pbc.ppr ["Y", "I", "Y", "Y"](8) [[A0]], [[A1]], [[A2]], [[A3]]
    // CHECK:   pbc.yield [[QL5]]#0, [[QL5]]#1, [[QL5]]#2, [[QL5]]#3
    %5:4 = pbc.ppr ["Y", "I", "Y", "Y"] (8) %4#0, %4#1, %4#2, %4#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    func.return
}
