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

// RUN: quantum-opt --commute-clifford-t-ppr --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_commute_1(%q1 : !quantum.bit){
    
    // Z(pi/4) * Z(pi/8) * Z(pi/4) * Z(pi/4) * Z(pi/8) * Z(pi/8)
    
    // CHECK: [[q1_0:%.+]] = qec.ppr ["Z"](8) 
    // CHECK: [[q1_1:%.+]] = qec.ppr ["Z"](8) [[q1_0]]
    // CHECK: [[q1_2:%.+]] = qec.ppr ["Z"](8) [[q1_1]]
    // CHECK: [[q1_3:%.+]] = qec.ppr ["Z"](4) [[q1_2]]
    // CHECK: [[q1_4:%.+]] = qec.ppr ["Z"](4) [[q1_3]]
    // CHECK: [[q1_5:%.+]] = qec.ppr ["Z"](4) [[q1_4]]
    %0 = qec.ppr ["Z"](4) %q1 : !quantum.bit
    %1 = qec.ppr ["Z"](8) %0 : !quantum.bit
    %2 = qec.ppr ["Z"](4) %1 : !quantum.bit
    %3 = qec.ppr ["Z"](4) %2 : !quantum.bit
    %4 = qec.ppr ["Z"](8) %3 : !quantum.bit
    %5 = qec.ppr ["Z"](8) %4 : !quantum.bit
    func.return
}

// -----

func.func @test_commute_2(%q1 : !quantum.bit, %q2 : !quantum.bit){

    // XX(pi/4) * XX(pi/8) * XX(pi/4) * XX(pi/8) * XX(pi/8)

    // CHECK: [[q1_0:%.+]]:2 = qec.ppr ["X", "X"](8)
    // CHECK: [[q1_1:%.+]]:2 = qec.ppr ["X", "X"](8) [[q1_0]]#0, [[q1_0]]#1
    // CHECK: [[q1_2:%.+]]:2 = qec.ppr ["X", "X"](8) [[q1_1]]#0, [[q1_1]]#1
    // CHECK: [[q1_3:%.+]]:2 = qec.ppr ["X", "X"](4) [[q1_2]]#0, [[q1_2]]#1
    // CHECK: [[q1_4:%.+]]:2 = qec.ppr ["X", "X"](4) [[q1_3]]#0, [[q1_3]]#1
    %0:2 = qec.ppr ["X", "X"](4) %q1, %q2 : !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["X", "X"](8) %0#0, %0#1 : !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["X", "X"](4) %1#0, %1#1 : !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["X", "X"](8) %2#0, %2#1 : !quantum.bit, !quantum.bit
    %4:2 = qec.ppr ["X", "X"](8) %3#0, %3#1 : !quantum.bit, !quantum.bit
    func.return
}

// -----

// TODO: Add check for this case
func.func @test_commute_3(%q1 : !quantum.bit, %q2 : !quantum.bit){

    // X0(pi/4) * X0X1(pi/4) * X0(pi/8) * X0X1(pi/4) * X0(pi/8) * X1(pi/8) 

    // CHECK: [[q1_0:%.+]] = qec.ppr
    %a = qec.ppr ["X"](4) %q1 : !quantum.bit
    %0:2 = qec.ppr ["X", "X"](4) %a, %q2 : !quantum.bit, !quantum.bit
    %1 = qec.ppr ["X"](8) %0#0 : !quantum.bit
    %2:2 = qec.ppr ["X", "X"](4) %1, %0#1 : !quantum.bit, !quantum.bit
    %3 = qec.ppr ["X"](8) %2#0 : !quantum.bit
    %4 = qec.ppr ["X"](8) %3 : !quantum.bit
    func.return
}


