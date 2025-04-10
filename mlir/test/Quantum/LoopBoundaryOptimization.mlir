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
        // CHECK-NOT: "RX"
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

func.func @test_loop_boundary_rotation_2(%q0: !quantum.bit) -> !quantum.bit {
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index

    // CHECK-LABEL:func.func @test_loop_boundary_rotation_2(
    // CHECK-SAME:[[arg0:%.+]]: !quantum.bit) -> !quantum.bit {
    // CHECK-DAG: [[cst:%.+]] = arith.constant -2.000000e-01 : f64
    // CHECK-DAG: [[theta:%.+]] = arith.constant 5.000000e-01 : f64
    // CHECK-DAG: [[phi:%.+]] = arith.constant 2.000000e-01 : f64
    // CHECK: [[qubit_0:%.+]] = quantum.custom "RX"([[phi]]) [[arg0]] : !quantum.bit
    // CHECK: [[scf:%.+]] = scf.for {{.*}} iter_args([[q_arg:%.+]] = [[qubit_0]]) -> (!quantum.bit) {
    %scf = scf.for %i = %start to %stop step %step iter_args(%q_arg = %q0) -> (!quantum.bit) {
        // CHECK-NOT: "RX"
        // CHECK: [[qubit_1:%.+]] = quantum.custom "Z"() [[q_arg]] : !quantum.bit
        // CHECK: [[qubit_2:%.+]] = quantum.custom "RX"([[phi]]) [[qubit_1]] : !quantum.bit
        // CHECK: [[qubit_3:%.+]] = quantum.custom "RX"([[theta]]) [[qubit_2]] : !quantum.bit
        %phi = arith.constant 0.2 : f64
        %theta = arith.constant 0.5 : f64
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

func.func @test_loop_boundary_register_1(%start: index, %stop: index) -> (!quantum.bit){
    
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

// -----

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
        // CHECK-NOT: "CNOT"
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

func.func @test_loop_boundary_register_3(%arg0: tensor<i64>, %arg1: tensor<f64>) -> !quantum.reg {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = quantum.alloc( 2) : !quantum.reg
    %extracted = tensor.extract %arg0[] : tensor<i64>
    %1 = arith.index_cast %extracted : i64 to index
    // CHECK-LABEL: func.func @test_loop_boundary_register_3(
    // CHECK-SAME: [[arg0:%.+]]: tensor<i64>,
    // CHECK-SAME: [[arg1:%.+]]: tensor<f64>
    // CHECK-DAG: [[alloc:%.+]] = quantum.alloc 
    // CHECK-DAG: [[extract_0:%.+]] = tensor.extract [[arg1]][] : tensor<f64>
    // CHECK-DAG: [[qubit_0:%.+]] = quantum.extract [[alloc]][ 0]
    // CHECK: [[qubit_1:%.+]] = {{.*}} "RX"([[extract_0]]) [[qubit_0]]
    // CHECK: [[insert_0:%.+]] = quantum.insert [[alloc]][ 0], [[qubit_1]]
    // CHECK: [[scf:%.+]] = scf.for {{.*}} iter_args([[arg3:%.+]] = [[insert_0]])
    %2 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %0) -> (!quantum.reg) {
        // CHECK-NOT: "RX"
        // CHECK-NOT: "Hadamard"
        // CHECK-DAG: [[extract_1:%.+]] = quantum.extract [[arg3]][ 0]
        // CHECK: [[extract_2:%.+]] = tensor.extract [[arg1]][]
        // CHECK: [[qubit_2:%.+]] = {{.*}} "Hadamard"() [[extract_1]]
        // CHECK-DAG: [[extract_3:%.+]] = quantum.extract [[arg3]][ 1]
        // CHECK: [[qubit_3:%.+]]:2 = {{.*}} "CNOT"() [[qubit_2]], [[extract_3]]
        // CHECK: [[add_1:%.+]] = arith.addf [[extract_2]], [[extract_2]]
        // CHECK: [[extract_4:%.+]] = tensor.extract [[arg1]][]
        // CHECK: [[qubit_4:%.+]] = {{.*}} "RX"([[extract_4]]) [[qubit_3]]#0
        // CHECK: [[qubit_5:%.+]] = {{.*}} "RX"([[add_1]]) [[qubit_4]]
        %5 = quantum.extract %arg3[ 0] : !quantum.reg -> !quantum.bit
        %extracted_1 = tensor.extract %arg1[] : tensor<f64>
        %out_qubits_2 = quantum.custom "RX"(%extracted_1) %5 : !quantum.bit
        %out_qubits_3 = quantum.custom "Hadamard"() %out_qubits_2 : !quantum.bit
        %6 = quantum.extract %arg3[ 1] : !quantum.reg -> !quantum.bit
        %out_qubits_4:2 = quantum.custom "CNOT"() %out_qubits_3, %6 : !quantum.bit, !quantum.bit
        %7 = arith.addf %extracted_1, %extracted_1 : f64
        %out_qubits_5 = quantum.custom "RX"(%7) %out_qubits_4#0 : !quantum.bit
        %8 = quantum.insert %arg3[ 0], %out_qubits_5 : !quantum.reg, !quantum.bit
        %9 = quantum.insert %8[ 1], %out_qubits_4#1 : !quantum.reg, !quantum.bit
        scf.yield %9 : !quantum.reg
    }

    // CHECK: [[negf:%.+]] = arith.negf [[extract_0]]
    // CHECK: [[extract_5:%.+]] = quantum.extract [[scf]][ 0]
    // CHECK: [[qubit_6:%.+]] = {{.*}} "RX"([[negf]]) [[extract_5]]
    // CHECK: [[insert_1:%.+]] = quantum.insert [[scf]][ 0], [[qubit_6]]
    
    // CHECK: [[extract_6:%.+]] = quantum.extract [[insert_1]][ 0]
    // CHECK: [[extract_7:%.+]] = tensor.extract [[arg1]][]
    // CHECK: [[qubit_5:%.+]] = {{.*}} "RX"([[extract_7]]) [[extract_6]]
    // CHECK: [[insert_2:%.+]] = quantum.insert [[insert_1]][ 0], [[qubit_5]]
    // CHECK: return [[insert_2]]
    %3 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
    %extracted_0 = tensor.extract %arg1[] : tensor<f64>
    %out_qubits = quantum.custom "RX"(%extracted_0) %3 : !quantum.bit
    %4 = quantum.insert %2[ 0], %out_qubits : !quantum.reg, !quantum.bit
    return %4 : !quantum.reg
}

// -----

 func.func public @test_loop_boundary_from_frontend(%arg0: tensor<i64>, %arg1: tensor<f64>) -> tensor<f64> {
      // CHECK-LABEL: func.func public @test_loop_boundary_from_frontend(
      // CHECK-SAME: [[arg0:%.+]]: tensor<i64>, 
      // CHECK-SAME: [[arg1:%.+]]: tensor<f64>) -> tensor<f64> {
      // CHECK-DAG: [[ex_0:%.+]] = quantum.extract %0[ 0]
      // CHECK-DAG: [[ex_1:%.+]] = tensor.extract [[arg1]][] : tensor<f64>
      // CHECK: [[q_0:%.+]] = quantum.custom "RX"([[ex_1]]) [[ex_0]]
      // CHECK: [[in_0:%.+]] = quantum.insert %0[ 0], [[q_0]]
      %c = stablehlo.constant dense<0> : tensor<i64>
      %extracted = tensor.extract %c[] : tensor<i64>
      %c_0 = stablehlo.constant dense<2> : tensor<i64>
      %0 = quantum.alloc( 2) : !quantum.reg
      %c_1 = stablehlo.constant dense<0> : tensor<i64>
      %c_2 = stablehlo.constant dense<1> : tensor<i64>
      %c_3 = stablehlo.constant dense<0> : tensor<i64>
      %extracted_4 = tensor.extract %c_1[] : tensor<i64>
      %1 = arith.index_cast %extracted_4 : i64 to index
      %extracted_5 = tensor.extract %arg0[] : tensor<i64>
      %2 = arith.index_cast %extracted_5 : i64 to index
      %extracted_6 = tensor.extract %c_2[] : tensor<i64>
      %3 = arith.index_cast %extracted_6 : i64 to index
      // CHECK: [[scf:%.+]]:2 = scf.for {{.*}} iter_args([[arg3:%.+]] = [[arg1]], [[arg4:%.+]] = [[in_0]])
      %4:2 = scf.for %arg2 = %1 to %2 step %3 iter_args(%arg3 = %arg1, %arg4 = %0) -> (tensor<f64>, !quantum.reg) {
        // CHECK-NOT: "RX"
        %8 = arith.index_cast %arg2 : index to i64
        %from_elements_9 = tensor.from_elements %8 : tensor<i64>
        %9 = stablehlo.sine %arg3 : tensor<f64>
        %c_10 = stablehlo.constant dense<0> : tensor<i64>
        %extracted_11 = tensor.extract %c_10[] : tensor<i64>
        %10 = quantum.extract %arg4[%extracted_11] : !quantum.reg -> !quantum.bit
        %extracted_12 = tensor.extract %arg3[] : tensor<f64>
        %out_qubits = quantum.custom "RX"(%extracted_12) %10 : !quantum.bit

        // CHECK: [[ex_2:%.+]] = quantum.extract [[arg4]][ 0]
        // CHECK: [[q_2:%.+]] = quantum.custom "Hadamard"() [[ex_2]]
        // CHECK: [[ex_3:%.+]] = tensor.extract [[arg1]][]
        // CHECK: [[q_3:%.+]] = quantum.custom "RX"([[ex_3]]) [[q_2]]
        // CHECK: [[ex_4:%.+]] = tensor.extract [[arg3]][]
        // CHECK: [[q_4:%.+]] = quantum.custom "RX"([[ex_4]]) [[q_3]]
        %out_qubits_13 = quantum.custom "Hadamard"() %out_qubits : !quantum.bit
        %extracted_14 = tensor.extract %arg3[] : tensor<f64>
        %out_qubits_15 = quantum.custom "RX"(%extracted_14) %out_qubits_13 : !quantum.bit
        %c_16 = stablehlo.constant dense<0> : tensor<i64>
        %extracted_17 = tensor.extract %c_16[] : tensor<i64>
        %11 = quantum.insert %arg4[%extracted_17], %out_qubits_15 : !quantum.reg, !quantum.bit
        scf.yield %9, %11 : tensor<f64>, !quantum.reg
      }

        // %5 = arith.negf %extracted_0 : f64
        // %6 = quantum.extract %4#1[ 0] : !quantum.reg -> !quantum.bit
        // %out_qubits_1 = quantum.custom "RX"(%5) %6 : !quantum.bit
        // %7 = quantum.insert %4#1[ 0], %out_qubits_1 : !quantum.reg, !quantum.bit
      
      // CHECK: [[negf:%.+]] = arith.negf [[extract_0]]
      // CHECK: [[ex_5:%.+]] = quantum.extract [[scf]]#1[ 0]
      // CHECK: [[q_5:%.+]] = quantum.custom "RX"([[negf]]) [[ex_5]]
      // CHECK: [[insert_0:%.+]] = quantum.insert [[scf]]#1[ 0], [[q_5]]
      %c_7 = stablehlo.constant dense<0> : tensor<i64>
      %extracted_8 = tensor.extract %c_7[] : tensor<i64>
      %5 = quantum.extract %4#1[%extracted_8] : !quantum.reg -> !quantum.bit
      %6 = quantum.namedobs %5[ PauliZ] : !quantum.obs
      %7 = quantum.expval %6 : f64
      %from_elements = tensor.from_elements %7 : tensor<f64>
      quantum.dealloc %4#1 : !quantum.reg
      return %from_elements : tensor<f64>
    }

// -----

func.func @test_loop_boundary_with_two_dominance_gate(%q0: !quantum.bit) -> !quantum.bit {
    %start = arith.constant 0 : index
    %stop = arith.constant 10 : index
    %step = arith.constant 1 : index
    %phi = arith.constant 0.2 : f64
    %theta = arith.constant 0.5 : f64

    // CHECK-NO: "RX"
    %scf = scf.for %i = %start to %stop step %step iter_args(%q_arg = %q0) -> (!quantum.bit) {
        // CHECK: "RX"
        // CHECK: "RX"
        %q_0 = quantum.custom "RX"(%phi) %q_arg : !quantum.bit
        %q_2 = quantum.custom "RX"(%theta) %q_0 : !quantum.bit
        scf.yield %q_2 : !quantum.bit
    }
    // CHECK-NO: "RX"
    func.return %scf : !quantum.bit
}

// -----

func.func public @test_loop_boundary_with_sequence_insert_bottom(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<f64> attributes {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e-01 : f64


    %0 = quantum.alloc( 2) : !quantum.reg
    %extracted = tensor.extract %arg0[] : tensor<i64>
    %1 = arith.index_cast %extracted : i64 to index

    // CHECK: [[q_0:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[ex_1:%.+]]  = tensor.extract %arg0[] : tensor<i64>
    // CHECK: [[q_1:%.+]]  = arith.index_cast [[ex_1]]  : i64 to index

    // CHECK: [[q_2:%.+]]  = quantum.extract [[q_0]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[o_0:%.+]]  = quantum.custom "Hadamard"() [[q_2]]  : !quantum.bit
    // CHECK: [[q_3:%.+]]  = quantum.insert [[q_0]][ 0], [[o_0]]  : !quantum.reg, !quantum.bit

    // CHECK: [[q_4:%.+]] = scf.for %arg2 = %c0 to [[q_1]] step %c1 iter_args(%arg3 = [[q_3]]) -> (!quantum.reg) {
    %2 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %0) -> (!quantum.reg) {
    // CHECK:   [[q_10:%.+]]  = quantum.extract %arg3[ 0] : !quantum.reg -> !quantum.bit
    // CHECK:   [[q_11:%.+]]  = quantum.insert %arg3[ 0], [[q_10]]  : !quantum.reg, !quantum.bit

    // CHECK:   [[q_12:%.+]]  = quantum.extract [[q_11]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK:   [[o_1:%.+]]  = quantum.custom "T"() [[q_12]]  : !quantum.bit
    // CHECK:   [[q_13:%.+]]  = quantum.insert %11[ 0], [[o_1]]  : !quantum.reg, !quantum.bit
    
    // CHECK:   [[q_14:%.+]]  = quantum.extract [[q_13]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK:   [[q_15:%.+]]  = quantum.insert [[q_13]][ 0], [[q_14]]  : !quantum.reg, !quantum.bit
    // CHECK:   scf.yield [[q_15]]  : !quantum.reg

      %6 = quantum.extract %arg3[ 0] : !quantum.reg -> !quantum.bit
      %out_qubits = quantum.custom "Hadamard"() %6 : !quantum.bit
      %7 = quantum.insert %arg3[ 0], %out_qubits : !quantum.reg, !quantum.bit

      %8 = quantum.extract %7[ 0] : !quantum.reg -> !quantum.bit
      %out_qubits_1 = quantum.custom "T"() %8 : !quantum.bit
      %9 = quantum.insert %7[ 0], %out_qubits_1 : !quantum.reg, !quantum.bit

      %10 = quantum.extract %9[ 0] : !quantum.reg -> !quantum.bit
      %out_qubits_2 = quantum.custom "Hadamard"() %10 : !quantum.bit
      %11 = quantum.insert %9[ 0], %out_qubits_2 : !quantum.reg, !quantum.bit

      scf.yield %11 : !quantum.reg
    }
    // CHECK: [[q_16:%.+]]  = quantum.extract [[q_4]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[o_2:%.+]]  = quantum.custom "Hadamard"() [[q_16]]  : !quantum.bit
    // CHECK: [[q_17:%.+]]  = quantum.insert %4[ 0], [[o_2]]  : !quantum.reg, !quantum.bit
    %extracted_0 = tensor.extract %arg1[] : tensor<i64>
    %3 = quantum.extract %2[%extracted_0] : !quantum.reg -> !quantum.bit
    %4 = quantum.namedobs %3[ PauliZ] : !quantum.obs
    %5 = quantum.expval %4 : f64
    %from_elements = tensor.from_elements %5 : tensor<f64>
    quantum.dealloc %2 : !quantum.reg
    return %from_elements : tensor<f64>
  }
