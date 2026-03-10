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

// RUN: quantum-opt %s --split-input-file --verify-diagnostics | FileCheck %s

func.func @foo(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    pbc.ppr ["X", "Z"] (4) %q1, %q2 : !quantum.bit, !quantum.bit
    func.return
}

func.func @boo(%q1 : !quantum.bit) {
    %0 = pbc.prepare zero : !quantum.bit
    %1 = pbc.prepare one : !quantum.bit
    %2 = pbc.prepare plus : !quantum.bit
    %3 = pbc.prepare minus : !quantum.bit
    %4 = pbc.prepare plus_i : !quantum.bit
    %5 = pbc.prepare minus_i : !quantum.bit
    func.return
}

func.func @magic() {
    %0 = pbc.fabricate magic : !quantum.bit
    %1 = pbc.fabricate magic_conj : !quantum.bit
    %2 = pbc.fabricate plus_i : !quantum.bit
    %3 = pbc.fabricate minus_i : !quantum.bit
    func.return
}

func.func @bar(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %m_0, %0 = pbc.ppm ["Z"] %q1 : i1, !quantum.bit
    %m_1, %1 = pbc.select.ppm (%m_0, ["X"], ["Z"]) %q2 : i1, !quantum.bit
    func.return
}

func.func @baz(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %m_0, %0 = pbc.ppm ["Z"] %q1 : i1, !quantum.bit
    %1:2 = pbc.ppr ["Y", "Y"] (4) %0, %q2 cond(%m_0) : !quantum.bit, !quantum.bit
    func.return
}

func.func @layer(%arg0 : !quantum.bit, %arg1 : !quantum.bit) -> i1{

// CHECK:func.func @layer([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit) -> i1 {

    %0 = pbc.layer(%q0 = %arg0) : !quantum.bit {
        %q_1 = pbc.ppr ["Z"](4) %q0 : !quantum.bit
        pbc.yield %q_1 : !quantum.bit
    }

    // CHECK: [[q0:%.+]] = pbc.layer([[arg_0:%.+]] = [[Q0]]) : !quantum.bit {
    // CHECK:   [[q_0:%.+]] = pbc.ppr ["Z"](4) [[arg_0]] : !quantum.bit
    // CHECK:   pbc.yield [[q_0]] : !quantum.bit

    %1:2 = pbc.layer(%q0 = %0, %q1 = %arg1): !quantum.bit, !quantum.bit {
        %q_1:2 = pbc.ppr ["X", "Y"](4) %q0, %q1 : !quantum.bit, !quantum.bit
        pbc.yield %q_1#0, %q_1#1 : !quantum.bit, !quantum.bit
    }

    // CHECK:  [[q1:%.+]]:2 = pbc.layer([[arg_0:%.+]] = [[q0]], [[arg_1:%.+]] = [[Q1]]) : !quantum.bit, !quantum.bit {
    // CHECK:   [[q_1:%.+]]:2 = pbc.ppr ["X", "Y"](4) [[arg_0]], [[arg_1]] : !quantum.bit, !quantum.bit
    // CHECK:   pbc.yield [[q_1]]#0, [[q_1]]#1 : !quantum.bit, !quantum.bit

    %res, %2:2 = pbc.layer(%q0 = %1#0, %q1 = %1#1): !quantum.bit, !quantum.bit {
        %q_1:3 = pbc.ppm ["X", "Z"] %q0, %q1 : i1, !quantum.bit, !quantum.bit
        pbc.yield %q_1#0, %q_1#1, %q_1#2 : i1, !quantum.bit, !quantum.bit
    }

    // CHECK:  [[q2:%.+]]:3 = pbc.layer([[arg_0:%.+]] = [[q1]]#0, [[arg_1:%.+]] = [[q1]]#1) : !quantum.bit, !quantum.bit {
    // CHECK:  [[M:%.+]], [[O:%.+]]:2 = pbc.ppm ["X", "Z"] [[arg_0]], [[arg_1]] : i1, !quantum.bit, !quantum.bit
    // CHECK:  pbc.yield [[M]], [[O]]#0, [[O]]#1 : i1, !quantum.bit, !quantum.bit

    %res_1, %3:2 = pbc.layer(%q0 = %2#0, %q1 = %2#1): !quantum.bit, !quantum.bit {
        %q_res, %q_1:2 = pbc.ppm ["X", "Z"] %q0, %q1 : i1, !quantum.bit, !quantum.bit
        pbc.yield %q_res, %q_1#0, %q_1#1 : i1, !quantum.bit, !quantum.bit
    }

    // CHECK:  [[q3:%.+]]:3 = pbc.layer([[A0:%.+]] = [[q2]]#1, [[A1:%.+]] = [[q2]]#2) : !quantum.bit, !quantum.bit {
    // CHECK:  [[M:%.+]], [[O:%.+]]:2 = pbc.ppm ["X", "Z"] [[A0]], [[A1]] : i1, !quantum.bit, !quantum.bit
    // CHECK:  pbc.yield [[M]], [[O]]#0, [[O]]#1 : i1, !quantum.bit, !quantum.bit

    func.return %res_1 : i1
}

func.func @arbitrary(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %c0 = arith.constant 1 : i1
    %const = arith.constant 0.124 : f64
    %const_1 = arith.constant 0.14 : f64
    %0 = pbc.ppr.arbitrary ["X"](%const) %q1 : !quantum.bit
    %1:2 = pbc.ppr.arbitrary ["X", "Z"](%const_1) %0, %q2 : !quantum.bit, !quantum.bit
    %2:2 = pbc.ppr.arbitrary ["X", "Z"](%const_1) %1#0, %1#1 cond(%c0) : !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func @baz_error(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    // expected-error@below {{'pbc.ppr' op attribute 'rotation_kind' failed to satisfy constraint: 16-bit signless integer attribute whose value is ±1, ±2, ±4, or ±8}}
    %0, %1 = pbc.ppr ["X", "Z"] (16) %q1, %q2 : !quantum.bit, !quantum.bit
    func.return
}
