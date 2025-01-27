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

// -----

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

// -----

func.func @test_loop_boundary_cnot_2(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index

    // Quantum circuit:
    // for _ in range(n):
        // CNOT Q0, Q1
        // H Q0
        // CNOT Q1, Q0

    // CHECK-LABEL: func @test_loop_boundary_cnot_2(
    %qq:2 = scf.for %i = %start to %stop step %step iter_args(%q_0 = %q0, %q_1 = %q1) -> (!quantum.bit, !quantum.bit) {
        // CHECK: quantum.custom "CNOT"
        %q_2:2 = quantum.custom "CNOT"() %q_0, %q_1 : !quantum.bit, !quantum.bit
        // CHECK: quantum.custom "H"
        %single_qubit = quantum.custom "H"() %q_2#1 : !quantum.bit
        // CHECK: quantum.custom "CNOT"
        %q_3:2 = quantum.custom "CNOT"() %single_qubit, %q_2#0 : !quantum.bit, !quantum.bit
        // CHECK: yield
        scf.yield %q_3#0, %q_3#1 : !quantum.bit, !quantum.bit
    }
    // CHECK: return
    func.return %qq#0, %qq#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_loop_boundary_cnot_3(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index

    // Quantum circuit:
    // for _ in range(n):
        // X Q0
        // CNOT Q0, Q1
        // H Q0
        // CNOT Q0, Q1
        // X Q1

    // CHECK: scf.for
    %qq:2 = scf.for %i = %start to %stop step %step iter_args(%q_0 = %q0, %q_1 = %q1) -> (!quantum.bit, !quantum.bit) {
        // CHECK: quantum.custom "X"
        %q_x = quantum.custom "X"() %q_0 : !quantum.bit
        // CHECK: quantum.custom "CNOT"
        %q_2:2 = quantum.custom "CNOT"() %q_x, %q_1 : !quantum.bit, !quantum.bit
        %q_3 = quantum.custom "H"() %q_2#0 : !quantum.bit
        // CHECK: quantum.custom "CNOT"
        %q_4:2 = quantum.custom "CNOT"() %q_3, %q_2#1 : !quantum.bit, !quantum.bit
        // CHECK: quantum.custom "X"
        %q_5 = quantum.custom "X"() %q_4#1 : !quantum.bit
        scf.yield %q_4#0, %q_5 : !quantum.bit, !quantum.bit
    }

    func.return %qq#0, %qq#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_loop_boundary_rotation(%q0: !quantum.bit) -> !quantum.bit {
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index
    %phi = arith.constant 0.2 : f64
    %theta = arith.constant 0.5 : f64

    // CHECK-LABEL:func.func @test_loop_boundary_rotation(
    // CHECK-SAME:[[arg0:%.+]]: !quantum.bit) -> !quantum.bit {
    // CHECK: [[cst:%.+]] = arith.constant -2.000000e-01 : f64
    // CHECK: [[phi:%.+]] = arith.constant 2.000000e-01 : f64
    // CHECK: [[theta:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK: [[qubit_0:%.+]] = quantum.custom "RX"([[phi]]) [[arg0]] : !quantum.bit
    // CHECK: [[scf:%.+]] = scf.for {{.*}} iter_args([[q_arg:%.+]] = [[qubit_0]]) -> (!quantum.bit) {
    %scf = scf.for %i = %start to %stop step %step iter_args(%q_arg = %q0) -> (!quantum.bit) {
        // CHECK: [[qubit_1:%.+]] = quantum.custom "Z"() [[q_arg]] : !quantum.bit
        // CHECK: [[qubit_2:%.+]] = quantum.custom "RX"([[phi]]) [[qubit_1]] : !quantum.bit
        // CHECK: [[qubit_3:%.+]] = quantum.custom "RX"([[theta]]) [[qubit_2]] : !quantum.bit
        %q_0 = quantum.custom "RX"(%phi) %q_arg : !quantum.bit
        %q_1 = quantum.custom "Z"() %q_0 : !quantum.bit
        %q_2 = quantum.custom "RX"(%theta) %q_1 : !quantum.bit
        scf.yield %q_2 : !quantum.bit
    }

    // CHECK: [[qubit_4:%.+]] = quantum.custom "RX"([[cst]]) [[scf]] : !quantum.bit
    // CHECK: return [[qubit_4]]
    func.return %scf : !quantum.bit
}

// -----

func.func @test_loop_boundary_rotation_1(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {

    %stop = arith.constant 10 : index
    %one = arith.constant 1 : index
    %theta = arith.constant 0.2 : f64

    // Quantum circuit:
    // for _ in range(n):
        // RX(theta) Q0
        // X Q1
        // CNOT Q0, Q1
        // RX(theta) Q0
        // X Q1

    // CHECK: scf.for
    %scf:2 = scf.for %i = %one to %stop step %one iter_args(%q_arg0 = %q0, %q_arg1 = %q1) -> (!quantum.bit, !quantum.bit) {
        %q_0 = quantum.custom "RX"(%theta) %q_arg0 : !quantum.bit
        %q_1 = quantum.custom "X"() %q_arg1 : !quantum.bit
        %q_2:2 = quantum.custom "CNOT"() %q_0, %q_1 : !quantum.bit, !quantum.bit
        %q_3 = quantum.custom "RX"(%theta) %q_2#0 : !quantum.bit
        %q_4 = quantum.custom "X"() %q_2#1 : !quantum.bit
        scf.yield %q_3, %q_4 : !quantum.bit, !quantum.bit
    }

    func.return %scf#0, %scf#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_loop_boundary_register(%start: index, %stop: index) -> (!quantum.bit){
    
    %0 = quantum.alloc(1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

    // CHECK: "H"
    // CHECK: scf.for
    // CHECK: "X"
    // CHECK: "H"
    %scf = scf.for %i = %start to %stop step %start iter_args(%arg0 = %1) -> (!quantum.bit) {
        %2 = quantum.custom "H"() %arg0 : !quantum.bit
        %3 = quantum.custom "X"() %2 : !quantum.bit
        %4 = quantum.custom "H"() %3 : !quantum.bit
        scf.yield %4 : !quantum.bit
    }

    func.return %scf : !quantum.bit
}

// -----
