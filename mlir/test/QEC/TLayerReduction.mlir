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

// RUN: quantum-opt --reduce-t-depth --split-input-file --verify-diagnostics %s | FileCheck %s

// From example in GoSC paper
func.func @test_t_layer_opt_GoSC(%qr0 : !quantum.bit, %qr1 : !quantum.bit, %qr2 : !quantum.bit, %qr3 : !quantum.bit) {

    // CHECK: func.func @test_t_layer_opt_GoSC([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit, [[Q2:%.+]]: !quantum.bit, [[Q3:%.+]]: !quantum.bit) {
    
    // layer 1
    // CHECK: [[q0:%.+]]:4 = qec.ppr ["I", "Z", "I", "I"](-8) [[Q0]], [[Q1]], [[Q2]], [[Q3]]
    // CHECK: [[q1:%.+]]:4 = qec.ppr ["X", "I", "Y", "Z"](8) [[q0]]#0, [[q0]]#1, [[q0]]#2, [[q0]]#3
    // CHECK: [[q2:%.+]]:4 = qec.ppr ["X", "Z", "Y", "I"](8) [[q1]]#0, [[q1]]#1, [[q1]]#2, [[q1]]#3

    // layer 2
    // CHECK: [[q3:%.+]]:4 = qec.ppr ["X", "Y", "Z", "Y"](8) [[q2]]#0, [[q2]]#1, [[q2]]#2, [[q2]]#3
    // CHECK: [[q4:%.+]]:4 = qec.ppr ["X", "Z", "I", "X"](-8) [[q3]]#0, [[q3]]#1, [[q3]]#2, [[q3]]#3
    // CHECK: [[q5:%.+]]:4 = qec.ppr ["Y", "I", "Y", "Y"](8) [[q4]]#0, [[q4]]#1, [[q4]]#2, [[q4]]#3
    
    // layer 1 
    %0:4 = qec.ppr ["I", "Z", "I", "I"](-8) %qr0, %qr1, %qr2, %qr3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    // layer 2
    %1:4 = qec.ppr ["X", "Y", "Z", "Y"](8) %0#0, %0#1, %0#2, %0#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %2:4 = qec.ppr ["X", "I", "Y", "Z"](8) %1#0, %1#1, %1#2, %1#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    // layer 3
    %3:4 = qec.ppr ["X", "Z", "I", "X"](-8) %2#0, %2#1, %2#2, %2#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %4:4 = qec.ppr ["X", "Z", "Y", "I"](8) %3#0, %3#1, %3#2, %3#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    // layer 4
    %5:4 = qec.ppr ["Y", "I", "Y", "Y"](8) %4#0, %4#1, %4#2, %4#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    func.return
}

// -----

// Test from GoSC where the Identity removed from IR. 
// This ensure that the pass works correctly where two PPRs have different number of qubits.
func.func @test_t_layer_opt_GoSC_no_Identity(%qr0 : !quantum.bit, %qr1 : !quantum.bit, %qr2 : !quantum.bit, %qr3 : !quantum.bit) {

    // CHECK: func.func @test_t_layer_opt_GoSC_no_Identity([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit, [[Q2:%.+]]: !quantum.bit, [[Q3:%.+]]: !quantum.bit)
    
    // layer 1
    // Note: We track the Qubit that each PRR applies on.
    // q0 -> [__, Q1, __, __]
    // q1 -> [Q0, __, Q2, Q3]
    // q2 -> [Q0, Q1, Q2, __]
    // CHECK: [[q0:%.+]] = qec.ppr ["Z"](-8) [[Q1]]
    // CHECK: [[q1:%.+]]:3 = qec.ppr ["X", "Y", "Z"](8) [[Q0]], [[Q2]], [[Q3]] 
    // CHECK: [[q2:%.+]]:3 = qec.ppr ["X", "Z", "Y"](8) [[q1]]#0, [[q0]], [[q1]]#1 

    // layer 2
    // q3 -> [Q0, Q1, Q2, Q3]
    // q4 -> [Q0, Q1, __, Q3]
    // q5 -> [Q0, __, Q2, Q3]
    // CHECK: [[q3:%.+]]:4 = qec.ppr ["X", "Y", "Z", "Y"](8) [[q2]]#0, [[q2]]#1, [[q2]]#2, [[q1]]#2
    // CHECK: [[q4:%.+]]:3 = qec.ppr ["X", "Z", "X"](-8) [[q3]]#0, [[q3]]#1, [[q3]]#3 
    // CHECK: [[q5:%.+]]:3 = qec.ppr ["Y", "Y", "Y"](8) [[q4]]#0, [[q3]]#2, [[q4]]#2 
    
    // layer 1 
    %0 = qec.ppr ["Z"](-8) %qr1 : !quantum.bit

    // layer 2
    %1:4 = qec.ppr ["X", "Y", "Z", "Y"](8) %qr0, %0, %qr2, %qr3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %2:3 = qec.ppr ["X", "Y", "Z"](8) %1#0, %1#2, %1#3 : !quantum.bit, !quantum.bit, !quantum.bit

    // layer 3
    %3:3 = qec.ppr ["X", "Z", "X"](-8) %2#0, %1#1, %2#2 : !quantum.bit, !quantum.bit, !quantum.bit
    %4:3 = qec.ppr ["X", "Z", "Y"](8) %3#0, %3#1, %2#1 :  !quantum.bit, !quantum.bit, !quantum.bit

    // layer 4
    %5:3 = qec.ppr ["Y", "Y", "Y"](8) %4#0, %4#2, %3#2 : !quantum.bit, !quantum.bit, !quantum.bit

    func.return
}

// -----

func.func @test_t_layer_opt_0(%qr0 : !quantum.bit, %qr1 : !quantum.bit){

    // CHECK: func.func @test_t_layer_opt_0([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit) {

    // layer 1
    // CHECK: [[q0:%.+]]:2 = qec.ppr ["X", "Z"](8) [[Q0]], [[Q1]]
    // CHECK: [[q1:%.+]]:2 = qec.ppr ["Y", "X"](-8) [[q0]]#0, [[q0]]#1

    // layer 2
    // CHECK: [[q2:%.+]]:2 = qec.ppr ["X", "Y"](8) [[q1]]#0, [[q1]]#1

    // layer 1
    %0:2 = qec.ppr ["X", "Z"](8) %qr0, %qr1 : !quantum.bit, !quantum.bit
    
    // layer 2
    %1:2 = qec.ppr ["X", "Y"](8) %0#0, %0#1 : !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["Y", "X"](-8) %1#0, %1#1 : !quantum.bit, !quantum.bit
    
    func.return
}

// -----

func.func @test_t_layer_opt_1(%qr0 : !quantum.bit, %qr1 : !quantum.bit, %qr2 : !quantum.bit){

    // CHECK: func.func @test_t_layer_opt_1([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit, [[Q2:%.+]]: !quantum.bit) {

    // CHECK: [[q0:%.+]]:3 = qec.ppr ["X", "Z", "Y"](8) [[Q0]], [[Q1]], [[Q2]]
    // CHECK: [[q1:%.+]]:3 = qec.ppr ["I", "I", "Y"](-8) [[q0]]#0, [[q0]]#1, [[q0]]#2
    // CHECK: [[q2:%.+]]:3 = qec.ppr ["Z", "Y", "Y"](-8) [[q1]]#0, [[q1]]#1, [[q1]]#2

    // CHECK: [[q3:%.+]]:3 = qec.ppr ["X", "X", "Y"](8) [[q2]]#0, [[q2]]#1, [[q2]]#2
    // CHECK: [[q4:%.+]]:3 = qec.ppr ["Y", "Z", "Y"](8) [[q3]]#0, [[q3]]#1, [[q3]]#2

    // CHECK: [[q5:%.+]]:3 = qec.ppr ["X", "X", "Y"](8) [[q4]]#0, [[q4]]#1, [[q4]]#2

    // layer 1
    %0:3 = qec.ppr ["X", "Z", "Y"](8) %qr0, %qr1, %qr2 : !quantum.bit, !quantum.bit, !quantum.bit

    // layer 2
    %1:3 = qec.ppr ["X", "X", "Y"](8) %0#0, %0#1, %0#2 : !quantum.bit, !quantum.bit, !quantum.bit
    %2:3 = qec.ppr ["Y", "Z", "Y"](8) %1#0, %1#1, %1#2 : !quantum.bit, !quantum.bit, !quantum.bit
    %3:3 = qec.ppr ["I", "I", "Y"](-8) %2#0, %2#1, %2#2 : !quantum.bit, !quantum.bit, !quantum.bit

    // layer 3
    %4:3 = qec.ppr ["X", "X", "Y"](8) %3#0, %3#1, %3#2 : !quantum.bit, !quantum.bit, !quantum.bit
    %5:3 = qec.ppr ["Z", "Y", "Y"](-8) %4#0, %4#1, %4#2 : !quantum.bit, !quantum.bit, !quantum.bit

    func.return
}

// -----

func.func @test_t_layer_opt_3(%arg0 : !quantum.bit, %arg1 : !quantum.bit, %arg2 : !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit) {

    // CHECK: func.func @test_t_layer_opt_3([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit, [[Q2:%.+]]: !quantum.bit)

    // layer 1
    // CHECK: [[q0:%.+]]:2 = qec.ppr ["X", "Z"](8) [[Q0]], [[Q1]]
    // CHECK: [[q1:%.+]]:2 = qec.ppr ["X", "X"](8) [[q0]]#0, [[Q2]]

    // layer 2
    // CHECK: [[q2:%.+]]:3 = qec.ppr ["X", "X", "X"](8) [[q1]]#0, [[q0]]#1, [[q1]]#1
    // CHECK: return [[q2]]#0, [[q2]]#1, [[q2]]#2

    // layer 1
    %0:2 = qec.ppr ["X", "Z"     ](8) %arg0, %arg1       : !quantum.bit, !quantum.bit // 0, 1
    
    // layer 2
    %1:3 = qec.ppr ["X", "X", "X"](8) %0#0, %0#1, %arg2 : !quantum.bit, !quantum.bit, !quantum.bit // 0, 1, 2
    %2:2 = qec.ppr ["X",      "X"](8) %1#0,        %1#2 : !quantum.bit, !quantum.bit // 0, 2

    func.return %2#0, %1#1, %2#1 : !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

func.func @test_t_layer_opt_4(%arg0: !quantum.bit, %arg1: !quantum.bit, %arg2: !quantum.bit, %arg3: !quantum.bit, %arg4: !quantum.bit, %arg5: !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
  
    //CHECK: func.func @test_t_layer_opt_4([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit, [[Q2:%.+]]: !quantum.bit, [[Q3:%.+]]: !quantum.bit, [[Q4:%.+]]: !quantum.bit, [[Q5:%.+]]: !quantum.bit)
    
    // layer 1
    //CHECK: [[q0:%.+]]:5 = qec.ppr ["I", "Z", "I", "I", "X"](-8) [[Q0]], [[Q1]], [[Q2]], [[Q3]], [[Q5]]
    //CHECK: [[q1:%.+]]:4 = qec.ppr ["X", "I", "Y", "Z"](8) [[q0]]#0, [[q0]]#1, [[q0]]#2, [[q0]]#3
    //CHECK: [[q2:%.+]]:4 = qec.ppr ["X", "Z", "Y", "I"](8) [[q1]]#0, [[q1]]#1, [[q1]]#2, [[q1]]#3

    // layer 2
    //CHECK: [[q3:%.+]]:5 = qec.ppr ["X", "Y", "Z", "Y", "X"](8) [[q2]]#0, [[q2]]#1, [[q2]]#2, [[q2]]#3, [[Q4]]
    //CHECK: [[q4:%.+]]:4 = qec.ppr ["X", "Z", "I", "X"](-8) [[q3]]#0, [[q3]]#1, [[q3]]#2, [[q3]]#3
    //CHECK: [[q5:%.+]]:4 = qec.ppr ["Y", "I", "Y", "Y"](8) [[q4]]#0, [[q4]]#1, [[q4]]#2, [[q4]]#3
    //CHECK: return [[q5]]#0, [[q5]]#1, [[q5]]#2, [[q5]]#3, [[q0]]#4
  
    // layer 1: Q0, Q1, Q2, Q3, Q5
    %0:5 = qec.ppr ["I", "Z", "I", "I", "X"](-8) %arg0, %arg1, %arg2, %arg3, %arg5 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    // layer 2: Q0, Q1, Q2, Q3, Q4
    %1:5 = qec.ppr ["X", "Y", "Z", "Y", "X"](8) %0#0, %0#1, %0#2, %0#3, %arg4 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    // Q0, _, Q2, Q3
    %2:4 = qec.ppr ["X", "I", "Y", "Z"](8) %1#0, %1#1, %1#2, %1#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    // layer 3: Q0, Q1, _, Q3
    %3:4 = qec.ppr ["X", "Z", "I", "X"](-8) %2#0, %2#1, %2#2, %2#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    // Q0, Q1, Q2, _
    %4:4 = qec.ppr ["X", "Z", "Y", "I"](8) %3#0, %3#1, %3#2, %3#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
  
    // layer 4: Q0, _, Q2, Q3
    %5:4 = qec.ppr ["Y", "I", "Y", "Y"](8) %4#0, %4#1, %4#2, %4#3 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit

    func.return %5#0, %5#1, %5#2, %5#3, %0#4 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
}

// -----

func.func @test_t_layer_opt_5(%q0 : !quantum.bit, %q1 : !quantum.bit, %q2 : !quantum.bit) {
  // Test case where the ["X", "X", "X"] layer is applied to the same qubit twice, 
  // so its second PPR from second layer is moved merged to after the that first PPR from first layer.
  // And because of these can be merged, those two PPRs are merged into a single pi/4 PPR.

  // CHECK: func.func @test_t_layer_opt_5([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit, [[Q2:%.+]]: !quantum.bit)
  // layer 1
  // CHECK: [[q1:%.+]]:3 = qec.ppr ["X", "X", "X"](4) [[Q0]], [[Q1]], [[Q2]]
  // CHECK: [[q2:%.+]]:3 = qec.ppr ["X", "Z", "Y"](-8) [[q1]]#0, [[q1]]#1, [[q1]]#
  
  // layer 2
  // CHECK: [[q3:%.+]]:3 = qec.ppr ["Z", "Y", "X"](-8) [[q2]]#0, [[q2]]#1, [[q2]]#2

  %0:3 = qec.ppr ["X", "X", "X"](8) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
  %1:3 = qec.ppr ["X", "Z", "Y"](-8) %0#0, %0#1, %0#2 : !quantum.bit, !quantum.bit, !quantum.bit

  %2:3 = qec.ppr ["Z", "Y", "X"](-8) %1#0, %1#1, %1#2 : !quantum.bit, !quantum.bit, !quantum.bit
  %3:3 = qec.ppr ["X", "X", "X"](8) %2#0, %2#1, %2#2 : !quantum.bit, !quantum.bit, !quantum.bit

  return
}
