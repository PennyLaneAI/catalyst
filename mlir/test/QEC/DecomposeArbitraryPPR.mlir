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

// RUN: quantum-opt --decompose-arbitrary-ppr --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_arb_ppr_to_arb_z_0(%q1 : !quantum.bit, %q2 : !quantum.bit){

    // CHECK-LABEL: @test_arb_ppr_to_arb_z_0
    // CHECK: ([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit)

    // CHECK: [[C:%.+]] = arith.constant 1.230000e-01 : f64
    // CHECK: [[A0:%.+]] = quantum.alloc_qb
    // CHECK: [[A0_1:%.+]] = qec.prepare  plus [[A0]]
    // CHECK: [[M0:%.+]], [[Q_3:%.+]]:3 = qec.ppm ["X", "Z", "Z"] [[Q0]], [[Q1]], [[A0_1]]
    // CHECK: [[A0_2:%.+]] = qec.ppr ["X"](2) [[Q_3]]#2 cond([[M0]])
    // CHECK: [[A0_3:%.+]] = qec.ppr.arbitrary ["Z"]([[C]]) [[A0_2]]
    // CHECK: [[M1:%.+]], [[A0_4:%.+]] = qec.ppm ["X"] [[A0_3]]
    // CHECK: [[Q_2:%.+]] = qec.ppr ["X", "Z"](2) [[Q_3]]#0, [[Q_3]]#1 cond([[M1]])
    // CHECK: quantum.dealloc_qb [[A0_4]]

    %const = arith.constant 0.123 : f64
    %Q0, %Q1 = qec.ppr.arbitrary ["X", "Z"](%const) %q1 , %q2 : !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func @test_arb_ppr_to_arb_z_1(%q1 : !quantum.bit, %q2 : !quantum.bit, %angle : f64){

    // CHECK-LABEL: @test_arb_ppr_to_arb_z_1
    // CHECK: ({{.*}} [[ANGLE:%.+]]: f64)
    
    // CHECK: quantum.alloc_qb
    // CHECK: qec.prepare  plus
    // CHECK: [[M0:%.+]], {{.*}} = qec.ppm ["X", "Z"]
    // CHECK: qec.ppr ["X"](2) {{.*}} cond([[M0]])
    // CHECK: qec.ppr.arbitrary ["Z"]([[ANGLE]])
    // CHECK: [[M1:%.+]], {{.*}} = qec.ppm ["X"]
    // CHECK: qec.ppr ["X"](2) {{.*}} cond([[M1]])
    // CHECK: quantum.dealloc_qb
    %Q1 = qec.ppr.arbitrary ["X"](%angle) %q1 : !quantum.bit

    // CHECK: quantum.alloc_qb
    // CHECK: qec.prepare  plus
    // CHECK: [[M2:%.+]], {{.*}} = qec.ppm ["Y", "Z"]
    // CHECK: qec.ppr ["X"](2) {{.*}} cond([[M2]])
    // CHECK: qec.ppr.arbitrary ["Z"]([[ANGLE]])
    // CHECK: [[M3:%.+]], {{.*}} = qec.ppm ["X"]
    // CHECK: qec.ppr ["Y"](2) {{.*}} cond([[M3]])
    // CHECK: quantum.dealloc_qb
    %Q2 = qec.ppr.arbitrary ["Y"](%angle) %q2 : !quantum.bit

    func.return
}


// ----- 

// Single-qubit Pauli-Z PPR should not be decomposed.
func.func @test_arb_ppr_to_arb_z_2(%q1: !quantum.bit, %angle : f64){

    // CHECK-LABEL: @test_arb_ppr_to_arb_z_2
    // CHECK: ([[Q1:%.+]]: !quantum.bit, [[ANGLE:%.+]]: f64)
    // CHECK: qec.ppr.arbitrary ["Z"]([[ANGLE]]) [[Q1]]
    %Q1 = qec.ppr.arbitrary ["Z"](%angle) %q1 : !quantum.bit

    func.return
}
