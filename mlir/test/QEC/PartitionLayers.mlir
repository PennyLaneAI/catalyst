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

// RUN: quantum-opt --partition-layers --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_partition_layers(%qr0 : !quantum.bit, %qr1 : !quantum.bit, %qr2 : !quantum.bit, %qr3 : !quantum.bit) -> i1 {

    // CHECK:test_partition_layers([[qr0:%.+]]: !quantum.bit, [[qr1:%.+]]: !quantum.bit, [[qr2:%.+]]: !quantum.bit, [[qr3:%.+]]: !quantum.bit)
    // CHECK: [[Q0:%.+]]:4 = qec.layer([[A0:%.+]] = [[qr0]], [[A1:%.+]] = [[qr1]], [[A2:%.+]] = [[qr2]], [[A3:%.+]] = [[qr3]])
    // CHECK:       [[QL0:%.+]]:2 = qec.ppr ["X", "Z"](-8) [[A0]], [[A1]]
    // CHECK:       [[QL1:%.+]]:2 = qec.ppr ["Y", "X"](8) [[A2]], [[A3]]
    // CHECK:  qec.yield [[QL0]]#0, [[QL0]]#1, [[QL1]]#0, [[QL1]]#1
    
    %0:2= qec.ppr ["X", "Z"] (-8) %qr0, %qr1 : !quantum.bit, !quantum.bit // X Z (0, 1) pi/8 
    %1:2 = qec.ppr ["Y", "X"] (8) %qr2, %qr3 : !quantum.bit, !quantum.bit // Y X (2, 3) pi/8

    // CHECK: [[Q1:%.+]]:4 = qec.layer([[A0:%.+]] = [[Q0]]#3, [[A1:%.+]] = [[Q0]]#0, [[A2:%.+]] = [[Q0]]#1, [[A3:%.+]] = [[Q0]]#2)
    // CHECK:   [[QL2:%.+]] = qec.ppr ["Y"](8) [[A0]]
    // CHECK:   [[QL3:%.+]] = qec.ppr ["Z"](8) [[A1]]
    // CHECK:   [[QL4:%.+]]:2 = qec.ppr ["X", "X"](8) [[A2]], [[A3]]
    // CHECK:   qec.yield [[QL2]], [[QL3]], [[QL4]]#0, [[QL4]]#1

    %2 = qec.ppr ["Y"] (8) %1#1 : !quantum.bit // Y (3) pi/8
    %3 = qec.ppr ["Z"] (8) %0#0 : !quantum.bit // Z (0) pi/8
    %4:2 = qec.ppr ["X", "X"] (8) %0#1, %1#0 : !quantum.bit, !quantum.bit // X X (1, 2) pi/8

    // CHECK: [[Q2:%.+]]:7 = qec.layer([[A0:%.+]] = [[Q1]]#0, [[A1:%.+]] = [[Q1]]#1, [[A2:%.+]] = [[Q1]]#2, [[A3:%.+]] = [[Q1]]#3)
    // CHECK:   [[QL5:%.+]] = qec.ppr ["Z"](8) [[A0]] : !quantum.bit
    // CHECK:   [[M0:%.+]], [[QL7:%.+]]:3 = qec.ppm ["X", "Y", "Z"](8) [[A1]], [[A2]], [[A3]]
    // CHECK:   [[M1:%.+]], [[QL9:%.+]] = qec.ppm ["Z"](8) [[QL5]] : !quantum.bit
    // CHECK:   qec.yield [[QL5]], [[M0]], [[QL7]]#0, [[QL7]]#1, [[QL7]]#2, [[M1]], [[QL9]] : !quantum.bit, i1, !quantum.bit, !quantum.bit, !quantum.bit, i1, !quantum.bit
    
    %5 = qec.ppr ["Z"] (8) %2 : !quantum.bit // Z (3) pi/8
    %6:4 = qec.ppm ["X", "Y", "Z"] (8) %3, %4#0, %4#1 : !quantum.bit, !quantum.bit, !quantum.bit // X Y Z (0, 1, 2) pi/8
    %m, %7 = qec.ppm ["Z"](8) %5 : !quantum.bit // Z (3) pi/8

    // CHECK: return [[Q2]]#5 : i1
    func.return %m : i1
}

// -----
