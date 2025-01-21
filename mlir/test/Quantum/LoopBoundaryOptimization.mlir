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

// RUN: quantum-opt --loop-boundary --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_loop_boundary(%q: !quantum.bit) -> !quantum.bit {
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index
    %phi = arith.constant 0.1 : f64

    // CHECK-LABEL: func @test_loop_boundary(
    // CHECK-SAME:[[arg:%.+]]: !quantum.bit) -> !quantum.bit {
    // CHECK: [[qubit_0:%.+]] = quantum.custom "H"() [[arg]] : !quantum.bit 

    // CHECK: [[qubit_1:%.+]] = scf.for {{.*}} iter_args([[qubit_2:%.+]] = [[qubit_0]]) -> (!quantum.bit) {
    %qq = scf.for %i = %start to %stop step %step iter_args(%q_0 = %q) -> (!quantum.bit) {
        // CHECK-NOT: "H"
        %q_1 = quantum.custom "H"() %q_0 : !quantum.bit
        // CHECK: [[qubit_3:%.+]] = quantum.custom "RY"{{.*}} [[qubit_2]] : !quantum.bit
        %q_2 = quantum.custom "RY"(%phi) %q_1 : !quantum.bit
        // CHECK-NOT: "H"
        %q_3 = quantum.custom "H"() %q_2 : !quantum.bit
        // CHECK-NEXT: scf.yield [[qubit_3]] : !quantum.bit
        scf.yield %q_3 : !quantum.bit
    }

    // CHECK: [[qubit_4:%.+]] = quantum.custom "H"() [[qubit_1]] : !quantum.bit
    // CHECK: return [[qubit_4]]
    func.return %qq : !quantum.bit
}

func.func @test_loop_boundary_cnot(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index

    // Quantum circuit:
    // for _ in range(n):
        // CNOT Q0, Q1
        // H Q0
        // CNOT Q0, Q1

    // CHECK-LABEL: func @test_loop_boundary_cnot(
    // CHECK-SAME:[[arg0:%.+]]: !quantum.bit, [[arg1:%.+]]: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    // CHECK: [[qubit_0:%.+]]:2 = quantum.custom "CNOT"() [[arg0]], [[arg1]] : !quantum.bit, !quantum.bit
    // CHECK: [[qubit_1:%.+]]:2 = scf.for {{.*}} iter_args([[qubit_2:%.+]] = [[qubit_0]]#0, [[qubit_3:%.+]] = [[qubit_0]]#1) -> (!quantum.bit, !quantum.bit) {
    %qq:2 = scf.for %i = %start to %stop step %step iter_args(%q_0 = %q0, %q_1 = %q1) -> (!quantum.bit, !quantum.bit) {
        // CHECK-NOT: "CNOT"
        %q_2:2 = quantum.custom "CNOT"() %q_0, %q_1 : !quantum.bit, !quantum.bit
        // CHECK: [[qubit_4:%.+]] = quantum.custom "H"() [[qubit_2]] : !quantum.bit
        %single_qubit = quantum.custom "H"() %q_2#0 : !quantum.bit
        // CHECK-NOT: "CNOT"
        %q_3:2 = quantum.custom "CNOT"() %single_qubit, %q_2#1 : !quantum.bit, !quantum.bit
        // CHECK-NEXT: scf.yield [[qubit_4]], [[qubit_3]] : !quantum.bit, !quantum.bit
        scf.yield %q_3#0, %q_3#1 : !quantum.bit, !quantum.bit
    }
    // CHECK: [[qubit_5:%.+]]:2 = quantum.custom "CNOT"() [[qubit_1]]#0, [[qubit_1]]#1 : !quantum.bit, !quantum.bit
    // CHECK: return [[qubit_5]]#0, [[qubit_5]]#1
    func.return %qq#0, %qq#1 : !quantum.bit, !quantum.bit
}
