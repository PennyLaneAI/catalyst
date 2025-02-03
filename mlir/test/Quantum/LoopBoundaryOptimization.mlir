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
    // CHECK-SAME: [[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit) -> (!quantum.bit, !quantum.bit)
    // CHECK: [[scf:%.+]]:2 = scf.for {{.*}} iter_args([[arg0:%.+]] = [[q0]], [[arg1:%.+]] = [[q1]])
    %qq:2 = scf.for %i = %start to %stop step %step iter_args(%q_0 = %q0, %q_1 = %q1) -> (!quantum.bit, !quantum.bit) {
        // CHECK: [[qubit_0:%.+]]:2 = quantum.custom "CNOT"() [[arg0]], [[arg1]]
        %q_2:2 = quantum.custom "CNOT"() %q_0, %q_1 : !quantum.bit, !quantum.bit
        // CHECK: [[qubit_1:%.+]] = quantum.custom "H"() [[qubit_0]]#1 : !quantum.bit
        %single_qubit = quantum.custom "H"() %q_2#1 : !quantum.bit
        // CHECK: [[qubit_3:%.+]]:2 = quantum.custom "CNOT"() [[qubit_1]], [[qubit_0]]#0
        %q_3:2 = quantum.custom "CNOT"() %single_qubit, %q_2#0 : !quantum.bit, !quantum.bit
        // CHECK: scf.yield [[qubit_3]]#0, [[qubit_3]]#1
        scf.yield %q_3#0, %q_3#1 : !quantum.bit, !quantum.bit
    }
    // CHECK: return [[scf]]#0, [[scf]]#1
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

    // CHECK-LABEL: func.func @test_loop_boundary_cnot_3(
    // CHECK-SAME: [[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit) -> (!quantum.bit, !quantum.bit)
    // CHECK: [[scf:%.+]]:2 = scf.for {{.*}} iter_args([[arg0:%.+]] = [[q0]], [[arg1:%.+]] = [[q1]])
    %qq:2 = scf.for %i = %start to %stop step %step iter_args(%q_0 = %q0, %q_1 = %q1) -> (!quantum.bit, !quantum.bit) {
        // CHECK: [[qubit_0:%.+]] = quantum.custom "X"() [[arg0]] : !quantum.bit
        %q_x = quantum.custom "X"() %q_0 : !quantum.bit
        // CHECK: [[qubit_1:%.+]]:2 = quantum.custom "CNOT"() [[qubit_0]], [[arg1]]
        %q_2:2 = quantum.custom "CNOT"() %q_x, %q_1 : !quantum.bit, !quantum.bit
        // CHECK: [[qubit_2:%.+]] = quantum.custom "H"() [[qubit_1]]#0 : !quantum.bit
        %q_3 = quantum.custom "H"() %q_2#0 : !quantum.bit
        // CHECK: [[qubit_4:%.+]]:2 = quantum.custom "CNOT"() [[qubit_2]], [[qubit_1]]#1
        %q_4:2 = quantum.custom "CNOT"() %q_3, %q_2#1 : !quantum.bit, !quantum.bit
        // CHECK: [[qubit_5:%.+]] = quantum.custom "X"() [[qubit_4]]#1 : !quantum.bit
        %q_5 = quantum.custom "X"() %q_4#1 : !quantum.bit
        // CHECK: scf.yield [[qubit_4]]#0, [[qubit_5]] : !quantum.bit, !quantum.bit
        scf.yield %q_4#0, %q_5 : !quantum.bit, !quantum.bit
    }
    // CHECK: return [[scf]]#0, [[scf]]#1
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
    // CHECK-DAG: [[cst:%.+]] = arith.constant -2.000000e-01 : f64
    // CHECK-DAG: [[phi:%.+]] = arith.constant 2.000000e-01 : f64
    // CHECK-DAG: [[theta:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK: [[qubit_0:%.+]] = quantum.custom "RX"([[phi]]) [[arg0]] : !quantum.bit
    // CHECK: [[scf:%.+]] = scf.for {{.*}} iter_args([[q_arg:%.+]] = [[qubit_0]]) -> (!quantum.bit) {
    %scf = scf.for %i = %start to %stop step %step iter_args(%q_arg = %q0) -> (!quantum.bit) {
        // CHECK: [[qubit_1:%.+]] = quantum.custom "Z"() [[q_arg]] : !quantum.bit
        // CHECK-DAG: [[qubit_2:%.+]] = quantum.custom "RX"([[phi]]) [[qubit_1]] : !quantum.bit
        // CHECK-DAG: [[qubit_3:%.+]] = quantum.custom "RX"([[theta]]) [[qubit_2]] : !quantum.bit
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

    // CHECK-LABEL: func.func @test_loop_boundary_rotation_1(
    // CHECK-SAME: [[q0:%.+]]: !quantum.bit,
    // CHECK-SAME: [[q1:%.+]]: !quantum.bit
    // CHECK: [[thetaM:%.+]] = {{.*}} -2.000000e-01 : f64
    // CHECK: [[theta:%.+]] = {{.*}} 2.000000e-01 : f64
    // CHECK-DAG: [[qubit_0:%.+]] = {{.*}} "RX"([[theta]]) [[q0]]
    // CHECK-DAG: [[qubit_1:%.+]] = {{.*}} "X"() [[q1]] : !quantum.bit
    // CHECK: [[scf:%.+]]:2 = scf.for {{.*}} iter_args([[arg0:%.+]] = [[qubit_0]], [[arg1:%.+]] = [[qubit_1]])
    %scf:2 = scf.for %i = %one to %stop step %one iter_args(%q_arg0 = %q0, %q_arg1 = %q1) -> (!quantum.bit, !quantum.bit) {
        // CHECK-NOT: "RX"
        // CHECK-NOT: "X"
        %q_0 = quantum.custom "RX"(%theta) %q_arg0 : !quantum.bit
        %q_1 = quantum.custom "X"() %q_arg1 : !quantum.bit
        // CHECK: [[qubit_2:%.+]]:2 = {{.*}} "CNOT"() [[arg0]], [[arg1]]
        %q_2:2 = quantum.custom "CNOT"() %q_0, %q_1 : !quantum.bit, !quantum.bit
        // CHECK: [[qubit_3:%.+]] = {{.*}} "RX"([[theta]]) [[qubit_2]]#0
        // CHECK: [[qubit_4:%.+]] = {{.*}} "RX"([[theta]]) [[qubit_3]]
        %q_3 = quantum.custom "RX"(%theta) %q_2#0 : !quantum.bit
        // CHECK-NOT: "X"
        %q_4 = quantum.custom "X"() %q_2#1 : !quantum.bit
        scf.yield %q_3, %q_4 : !quantum.bit, !quantum.bit
    }

    // CHECK-DAG: [[qubit_5:%.+]] = {{.*}} "X"() [[scf]]#1
    // CHECK-DAG: [[qubit_6:%.+]] = {{.*}} "RX"([[thetaM]]) [[scf]]#0
    func.return %scf#0, %scf#1 : !quantum.bit, !quantum.bit
}

// -----

func.func @test_loop_boundary_register(%start: index, %stop: index) -> (!quantum.bit){
    
    %0 = quantum.alloc(1) : !quantum.reg
    %1 = quantum.extract %0[0] : !quantum.reg -> !quantum.bit

    // CHECK: [[reg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qubit_0:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit_1:%.+]] = {{.*}} "H"() [[qubit_0]] : !quantum.bit
    // CHECK: [[qubit_2:%.+]] = scf.for {{.*}} iter_args([[arg0:%.+]] = [[qubit_1]])
    %scf = scf.for %i = %start to %stop step %start iter_args(%arg0 = %1) -> (!quantum.bit) {
        // CHECK-NOT: "H"
        %2 = quantum.custom "H"() %arg0 : !quantum.bit
        // CHECK: [[qubit_3:%.+]] = {{.*}} "X"() [[arg0]]
        %3 = quantum.custom "X"() %2 : !quantum.bit
        // CHECK-NOT: "H"
        %4 = quantum.custom "H"() %3 : !quantum.bit
        // CHECK: scf.yield [[qubit_3]]
        scf.yield %4 : !quantum.bit
    }
    // CHECK: [[qubit_4:%.+]] = {{.*}} "H"() [[qubit_2]]
    // CHECK: return [[qubit_4]]
    func.return %scf : !quantum.bit
}


func.func @test_loop_boundary_register_2(%start: index, %stop: index, 
                                        %arg0: tensor<f64>, %arg1: tensor<f64>) -> (!quantum.reg){

    %0 = quantum.alloc(4) : !quantum.reg
    %idx = arith.constant 3 : i64
    // Quantum circuit:
    // for _ in range(n):
    //     CNOT Q0,Q1
    //     X Q1
    //     H Q0
    //     CNOT Q0,Q1

    // CHECK-LABEL: func.func @test_loop_boundary_register_2(
    // CHECK-SAME: [[arg0:%.+]]: index, [[arg1:%.+]]: index,
    // CHECK-SAME: [[arg2:%.+]]: tensor<f64>, [[arg3:%.+]]: tensor<f64>)
    // CHECK: [[reg:%.+]] = quantum.alloc( 4) : !quantum.reg
    // CHECK: [[qubit_0:%.+]] = quantum.extract [[reg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit_1:%.+]] = quantum.extract [[reg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit_2:%.+]]:2 = quantum.custom "CNOT"() [[qubit_0]], [[qubit_1]]
    // CHECK: [[insert_0:%.+]] = quantum.insert [[reg]][ 0], [[qubit_2]]#0 : !quantum.reg
    // CHECK: [[insert_1:%.+]] = quantum.insert [[insert_0]][ 1], [[qubit_2]]#1 : !quantum.reg
    // CHECK: [[scf:%.+]] = scf.for {{.*}} iter_args([[arg0:%.+]] = [[insert_1]]) -> (!quantum.reg)
    %scf = scf.for %i = %start to %stop step %start iter_args(%q_arr = %0) -> (!quantum.reg) {
        // CHECK: [[qubit_3:%.+]] = quantum.extract [[arg0]][ 0] : !quantum.reg -> !quantum.bit
        // CHECK: [[qubit_4:%.+]] = quantum.extract [[arg0]][ 1] : !quantum.reg -> !quantum.bit
        %1 = quantum.extract %q_arr[0] : !quantum.reg -> !quantum.bit
        %2 = quantum.extract %q_arr[1] : !quantum.reg -> !quantum.bit
        %3:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
        // CHECK: [[qubit_5:%.+]] = quantum.custom "X"() [[qubit_4]]
        // CHECK: [[qubit_6:%.+]] = quantum.custom "H"() [[qubit_3]]
        %4 = quantum.custom "X"() %3#1 : !quantum.bit
        %5 = quantum.custom "H"() %3#0 : !quantum.bit
        // CHECK-NOT: "CNOT"
        %6:2 = quantum.custom "CNOT"() %5, %4 : !quantum.bit, !quantum.bit
        %7 = quantum.insert %q_arr[0], %6#0 : !quantum.reg, !quantum.bit
        %8 = quantum.insert %7[1], %6#1 : !quantum.reg, !quantum.bit
        scf.yield %8 : !quantum.reg
    }

    // CHECK: [[qubit_7:%.+]] = quantum.extract [[scf]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit_8:%.+]] = quantum.extract [[scf]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[qubit_9:%.+]]:2 = quantum.custom "CNOT"() [[qubit_7]], [[qubit_8]]
    // CHECK: [[insert_2:%.+]] = quantum.insert [[scf]][ 1], [[qubit_9]]#1 : !quantum.reg
    // CHECK: [[insert_3:%.+]] = quantum.insert [[insert_2]][ 0], [[qubit_9]]#0 : !quantum.reg
    // CHECK: return [[insert_3]]
    func.return %scf : !quantum.reg
}

// -----
