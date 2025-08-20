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

// RUN: quantum-opt --t-layer-reduction --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @test_t_layer_opt_0(%qr0 : !quantum.bit, %qr1 : !quantum.bit){

    // CHECK: func.func @test_t_layer_opt_0([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit) {

    // CHECK: [[q0:%.+]]:2 = qec.ppr ["X", "Z"](8) [[Q0]], [[Q1]]
    // CHECK: [[q1:%.+]]:2 = qec.ppr ["Y", "X"](-8) [[q0]]#0, [[q0]]#1

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
