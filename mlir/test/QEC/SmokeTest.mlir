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

// RUN: quantum-opt %s | FileCheck %s

func.func @foo(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    qec.ppr ["X", "Z"] (4) %q1, %q2 : !quantum.bit, !quantum.bit
    func.return
}

func.func @boo(%q1 : !quantum.bit) {
    %0 = qec.prepare zero %q1 : !quantum.bit
    %1 = qec.prepare one %0 : !quantum.bit
    %2 = qec.prepare plus %1 : !quantum.bit
    %3 = qec.prepare minus %2 : !quantum.bit
    %4 = qec.prepare plus_i %3 : !quantum.bit
    %5 = qec.prepare minus_i %4 : !quantum.bit
    func.return
}

func.func @magic() {
    %0 = qec.fabricate magic : !quantum.bit
    %1 = qec.fabricate magic_conj : !quantum.bit
    %2 = qec.fabricate plus_i : !quantum.bit
    %3 = qec.fabricate minus_i : !quantum.bit
    func.return
}

func.func @bar(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %m_0, %0 = qec.ppm ["Z"] %q1 : i1, !quantum.bit
    %m_1, %1 = qec.select.ppm (%m_0, ["X"], ["Z"]) %q2 : i1, !quantum.bit
    func.return
}

func.func @baz(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %m_0, %0 = qec.ppm ["Z"] %q1 : i1, !quantum.bit
    %1:2 = qec.ppr ["Y", "Y"] (4) %0, %q2 cond(%m_0) : !quantum.bit, !quantum.bit
    func.return
}

func.func @layer(%arg0 : !quantum.bit, %arg1 : !quantum.bit) -> i1{

// CHECK:func.func @layer([[Q0:%.+]]: !quantum.bit, [[Q1:%.+]]: !quantum.bit) -> i1 {

    %0 = qec.layer(%q0 = %arg0) : !quantum.bit {
        %q_1 = qec.ppr ["Z"](4) %q0 : !quantum.bit
        qec.yield %q_1 : !quantum.bit
    }

    // CHECK: [[q0:%.+]] = qec.layer([[arg_0:%.+]] = [[Q0]]) : !quantum.bit {
    // CHECK:   [[q_0:%.+]] = qec.ppr ["Z"](4) [[arg_0]] : !quantum.bit
    // CHECK:   qec.yield [[q_0]] : !quantum.bit

    %1:2 = qec.layer(%q0 = %0, %q1 = %arg1): !quantum.bit, !quantum.bit {
        %q_1:2 = qec.ppr ["X", "Y"](4) %q0, %q1 : !quantum.bit, !quantum.bit
        qec.yield %q_1#0, %q_1#1 : !quantum.bit, !quantum.bit
    }

    // CHECK:  [[q1:%.+]]:2 = qec.layer([[arg_0:%.+]] = [[q0]], [[arg_1:%.+]] = [[Q1]]) : !quantum.bit, !quantum.bit {
    // CHECK:   [[q_1:%.+]]:2 = qec.ppr ["X", "Y"](4) [[arg_0]], [[arg_1]] : !quantum.bit, !quantum.bit
    // CHECK:   qec.yield [[q_1]]#0, [[q_1]]#1 : !quantum.bit, !quantum.bit

    %res, %2:2 = qec.layer(%q0 = %1#0, %q1 = %1#1): !quantum.bit, !quantum.bit {
        %q_1:3 = qec.ppm ["X", "Z"] %q0, %q1 : i1, !quantum.bit, !quantum.bit
        qec.yield %q_1#0, %q_1#1, %q_1#2 : i1, !quantum.bit, !quantum.bit
    }

    // CHECK:  [[q2:%.+]]:3 = qec.layer([[arg_0:%.+]] = [[q1]]#0, [[arg_1:%.+]] = [[q1]]#1) : !quantum.bit, !quantum.bit {
    // CHECK:  [[M:%.+]], [[O:%.+]]:2 = qec.ppm ["X", "Z"] [[arg_0]], [[arg_1]] : i1, !quantum.bit, !quantum.bit
    // CHECK:  qec.yield [[M]], [[O]]#0, [[O]]#1 : i1, !quantum.bit, !quantum.bit

    %res_1, %3:2 = qec.layer(%q0 = %2#0, %q1 = %2#1, %m = %res): !quantum.bit, !quantum.bit, i1 {
        %q_res, %q_1:2 = qec.ppm ["X", "Z"] %q0, %q1 cond(%m): i1, !quantum.bit, !quantum.bit
        qec.yield %q_res, %q_1#0, %q_1#1 : i1, !quantum.bit, !quantum.bit
    }

    // CHECK:  [[q3:%.+]]:3 = qec.layer([[A0:%.+]] = [[q2]]#1, [[A1:%.+]] = [[q2]]#2, [[A2:%.+]] = [[q2]]#0) : !quantum.bit, !quantum.bit, i1 {
    // CHECK:  [[M:%.+]], [[O:%.+]]:2 = qec.ppm ["X", "Z"] [[A0]], [[A1]] cond([[A2]]) : i1, !quantum.bit, !quantum.bit
    // CHECK:  qec.yield [[M]], [[O]]#0, [[O]]#1 : i1, !quantum.bit, !quantum.bit

    func.return %res_1 : i1
}

func.func @arbitrary(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %c0 = arith.constant 1 : i1
    %const = arith.constant 0.124 : f64
    %const_1 = arith.constant 0.14 : f64
    %0 = qec.ppr.arbitrary ["X"](%const) %q1 : !quantum.bit
    %1:2 = qec.ppr.arbitrary ["X", "Z"](%const_1) %0, %q2 : !quantum.bit, !quantum.bit
    %2:2 = qec.ppr.arbitrary ["X", "Z"](%const_1) %1#0, %1#1 cond(%c0) : !quantum.bit, !quantum.bit
    func.return
}
