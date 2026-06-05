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

// RUN: quantum-opt --to-ppr --split-input-file -verify-diagnostics %s | FileCheck %s

func.func @test_clifford_t_to_ppr(%q1 : !quantum.bit, %q2 : !quantum.bit){
    // pi / 4 = 0.78539816
    // pi / 8 = 826990
    // CHECK-NOT: quantum.custom
    // CHECK: ([[q1:%.+]]: !quantum.bit, [[q2:%.+]]: !quantum.bit)
    // CHECK: [[C2:%.+]] = arith.constant 0.785
    // CHECK: [[C:%.+]] = arith.constant -0.392
    // CHECK: [[C0:%.+]] = arith.constant -0.785
    // CHECK: [[C1:%.+]] = arith.constant -1.570
    // CHECK: quantum.gphase([[C1]])
    // CHECK: [[q1_0_0:%.+]] = pbc.ppr ["Z"](4) [[q1]]
    // CHECK: [[q1_0_1:%.+]] = pbc.ppr ["X"](4) [[q1_0_0]]
    // CHECK: [[q1_0_2:%.+]] = pbc.ppr ["Z"](4) [[q1_0_1]]
    // CHECK: quantum.gphase([[C0]])
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["Z"](4) [[q1_0_2]]
    // CHECK: quantum.gphase([[C]])
    // CHECK: [[q1_2:%.+]] = pbc.ppr ["Z"](8) [[q1_1]]
    // CHECK: quantum.gphase([[C2]])
    // CHECK: [[q1_3:%.+]]:2 = pbc.ppr ["Z", "X"](4) [[q1_2]], [[q2]]
    %q1_0 = quantum.custom "H"() %q1 : !quantum.bit
    %q1_1 = quantum.custom "S"() %q1_0 : !quantum.bit
    %q1_2 = quantum.custom "T"() %q1_1 : !quantum.bit
    %q1_3:2 = quantum.custom "CNOT"() %q1_2, %q2 : !quantum.bit, !quantum.bit
    // CHECK: [[q1_4:%.+]] = pbc.ppr ["Z"](-4) [[q1_3]]#0
    // CHECK: [[q1_5:%.+]] = pbc.ppr ["X"](-4) [[q1_3]]#1
    // CHECK-NOT: quantum.custom
    // CHECK-NEXT: return
    func.return
}

// -----

func.func public @test_clifford_t_to_ppr_1() -> (tensor<i1>, tensor<i1>) {
    // CHECK: [[C2:%.+]] = arith.constant 0.785
    // CHECK: [[C:%.+]] = arith.constant -0.392
    // CHECK: [[C0:%.+]] = arith.constant -1.570
    // CHECK: [[C1:%.+]] = arith.constant -0.785
    // CHECK: [[q0:%.+]] = quantum.alloc( 2) : !quantum.reg
    %0 = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[q1_0:%.+]] = quantum.extract [[q0]][ 1]
    %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.gphase([[C1]])
    // CHECK: [[q1_1:%.+]] = pbc.ppr ["Z"](4) [[q1_0]]
    %out_qubits = quantum.custom "S"() %1 : !quantum.bit
    // CHECK: [[q0_0:%.+]] = quantum.extract [[q0]][ 0]
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: quantum.gphase([[C0]])
    // CHECK: [[q0_1_0:%.+]] = pbc.ppr ["Z"](4) [[q0_0]]
    // CHECK: [[q0_1_1:%.+]] = pbc.ppr ["X"](4) [[q0_1_0]]
    // CHECK: [[q0_1_2:%.+]] = pbc.ppr ["Z"](4) [[q0_1_1]]
    %out_qubits_0 = quantum.custom "Hadamard"() %2 : !quantum.bit
    // CHECK: quantum.gphase([[C]])
    // CHECK: [[q0_2:%.+]] = pbc.ppr ["Z"](8) [[q0_1_2]]
    %out_qubits_1 = quantum.custom "T"() %out_qubits_0 : !quantum.bit
    // CHECK: quantum.gphase([[C2]])
    // CHECK: [[q_3:%.+]]:2 = pbc.ppr ["Z", "X"](4) [[q0_2]], [[q1_1]]
    // CHECK: [[q_4:%.+]] = pbc.ppr ["Z"](-4) [[q_3]]#0
    // CHECK: [[q_5:%.+]] = pbc.ppr ["X"](-4) [[q_3]]#1
    %out_qubits_2:2 = quantum.custom "CNOT"() %out_qubits_1, %out_qubits : !quantum.bit, !quantum.bit

    // CHECK: [[mres_0:%.+]], [[q0_4:%.+]] = pbc.ppm ["Z"] [[q_4]]
    // CHECK: [[tensor_0:%.+]] = tensor.from_elements [[mres_0]] : tensor<i1>
    %mres_0, %out_qubit_0 = quantum.measure %out_qubits_2#0 : i1, !quantum.bit
    %from_elements_0 = tensor.from_elements %mres_0 : tensor<i1>

    // CHECK: [[mres_1:%.+]], [[q0_5:%.+]] = pbc.ppm ["Z"] [[q_5]]
    // CHECK: [[tensor_1:%.+]] = tensor.from_elements [[mres_1]] : tensor<i1>
    %mres_1, %out_qubit_1 = quantum.measure %out_qubits_2#1 : i1, !quantum.bit
    %from_elements_1 = tensor.from_elements %mres_1 : tensor<i1>

    %3 = quantum.insert %0[ 0], %out_qubit_0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_qubit_1 : !quantum.reg, !quantum.bit
    quantum.dealloc %4 : !quantum.reg
    // CHECK: return [[tensor_0]], [[tensor_1]] : tensor<i1>, tensor<i1>
    return %from_elements_0, %from_elements_1 : tensor<i1>, tensor<i1>
}

// -----

func.func @test_clifford_t_to_ppr_2(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    // expected-error @+1 {{failed to legalize operation 'quantum.custom' that was explicitly marked illegal}}
    %0 = quantum.custom "SOME_UNKNOWN_GATE"() %q1 : !quantum.bit // expected-error @+0 {{Unsupported gate for PBC conversion. Supported gates: }}
    %1 = quantum.custom "S"() %0 : !quantum.bit
    %2 = quantum.custom "T"() %1 : !quantum.bit
    %3:2 = quantum.custom "CNOT"() %2, %q2 : !quantum.bit, !quantum.bit
    func.return
}

// -----

func.func @test_controlled_gate_to_ppr_unsupported(%q0 : !quantum.bit, %q1 : !quantum.bit,
                                                   %ctrlval : i1) {
    %theta_t = stablehlo.constant dense<0.42> : tensor<f64>
    %theta = tensor.extract %theta_t[] : tensor<f64>
    // expected-error @+1 {{failed to legalize operation 'quantum.custom' that was explicitly marked illegal}}
    %out_q, %out_ctrl = quantum.custom "RX"(%theta) %q0 ctrls (%q1) ctrlvals (%ctrlval) : !quantum.bit ctrls !quantum.bit // expected-error @+0 {{Unsupported controlled gate. Supported gates: }}
    func.return
}

// -----

func.func @test_unsupported_operation_to_ppr(%q : !quantum.bit,
                                            %u : memref<2x2xcomplex<f64>>) {
    // expected-error @+1 {{failed to legalize operation 'quantum.unitary' that was explicitly marked illegal}}
    %out = quantum.unitary(%u : memref<2x2xcomplex<f64>>) %q : !quantum.bit // expected-error @+0 {{Unsupported operation for PBC conversion. Supported gates: }}
    func.return
}

// -----

func.func @test_clifford_t_to_ppr_3(%q1 : !quantum.bit, %q2 : !quantum.bit) {
    %0 = quantum.custom "PauliX"() %q1 : !quantum.bit
    %1 = quantum.custom "PauliY"() %0 : !quantum.bit
    %2 = quantum.custom "PauliZ"() %1 : !quantum.bit
    %3 = quantum.custom "S"() %2 {adjoint} : !quantum.bit
    %4 = quantum.custom "T"() %3 adj : !quantum.bit

    // CHECK: pbc.ppr ["X"](2)
    // CHECK: pbc.ppr ["Y"](2)
    // CHECK: pbc.ppr ["Z"](2)
    // CHECK: pbc.ppr ["Z"](-4)
    // CHECK: pbc.ppr ["Z"](-8)
    func.return
}

// -----

func.func @test_standard_pauli_rot_to_ppr(%q1 : !quantum.bit){
    %cst = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
    %extracted = tensor.extract %cst[] : tensor<f64>
    %out_qubits = quantum.paulirot ["Z"](%extracted) %q1 : !quantum.bit
    func.return
    // CHECK-NOT: quantum.gphase
    // CHECK: pbc.ppr ["Z"](2)
}

// -----

func.func @test_arbitrary_pauli_rot_to_ppr(%q1 : !quantum.bit){
    %cst = stablehlo.constant dense<0.42> : tensor<f64>
    %extracted = tensor.extract %cst[] : tensor<f64>
    %out_qubits = quantum.paulirot ["Z"](%extracted) %q1 : !quantum.bit
    func.return

    // CHECK: [[div:%.+]] = arith.constant 2.100000e-01 : f64
    // CHECK: [[q0:%.+]] = pbc.ppr.arbitrary ["Z"]([[div]])
}

// -----

func.func @test_dynamic_pauli_rot_to_ppr(%q1 : !quantum.bit, %arg0 : tensor<f64>){
    %extracted = tensor.extract %arg0[] : tensor<f64>
    %out_qubits_4 = quantum.paulirot ["Z"](%extracted) %q1 : !quantum.bit
    func.return

    // CHECK: [[cst:%.+]] = arith.constant 2.000000e+00 : f64
    // CHECK: [[extracted:%.+]] = tensor.extract
    // CHECK: [[div:%.+]] = arith.divf [[extracted]], [[cst]] : f64
    // CHECK: [[q0:%.+]] = pbc.ppr.arbitrary ["Z"]([[div]])
}

// -----

func.func @test_arbitrary_pauli_rot_to_ppr_2(%q1 : !quantum.bit, %q2 : !quantum.bit){
    %cst = stablehlo.constant dense<0.42> : tensor<f64>
    %extracted = tensor.extract %cst[] : tensor<f64>
    %out_qubits:2 = quantum.paulirot ["I", "I"](%extracted) %q1, %q2: !quantum.bit, !quantum.bit
    // CHECK-NOT: quantum.paulirot
    // CHECK-NOT: pbc.ppr.arbitrary
    func.return
}

// -----

func.func @test_parametrized_gate_fixed_angles_to_ppr(%q0 : !quantum.bit, %q1 : !quantum.bit,%q2 : !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    %cst = stablehlo.constant dense<3.1415926535897931> : tensor<f64>
    %theta = tensor.extract %cst[] : tensor<f64>
    %rx = quantum.custom "RX"(%theta) %q0 : !quantum.bit
    %ry = quantum.custom "RY"(%theta) %rx : !quantum.bit
    %rz = quantum.custom "RZ"(%theta) %ry : !quantum.bit
    %xx:2 = quantum.custom "IsingXX"(%theta) %rz, %q1 : !quantum.bit, !quantum.bit
    %yy:2 = quantum.custom "IsingYY"(%theta) %xx#0, %xx#1 : !quantum.bit, !quantum.bit
    %zz:2 = quantum.custom "IsingZZ"(%theta) %yy#0, %yy#1 : !quantum.bit, !quantum.bit
    %mrz:3 = quantum.multirz(%theta) %zz#0, %zz#1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
    func.return %mrz#0, %mrz#1, %mrz#2 : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.custom
    // CHECK: [[RX:%.+]] = pbc.ppr ["X"](2) [[q0:%.+]]
    // CHECK: [[RY:%.+]] = pbc.ppr ["Y"](2) [[RX]]
    // CHECK: [[RZ:%.+]] = pbc.ppr ["Z"](2) [[RY]]
    // CHECK: [[XX:%.+]]:2 = pbc.ppr ["X", "X"](2) [[RZ]], [[q1:%.+]]
    // CHECK: [[YY:%.+]]:2 = pbc.ppr ["Y", "Y"](2) [[XX]]#0, [[XX]]#1
    // CHECK: [[ZZ:%.+]]:2 = pbc.ppr ["Z", "Z"](2) [[YY]]#0, [[YY]]#1
    // CHECK: [[MRZ:%.+]]:3 = pbc.ppr ["Z", "Z", "Z"](2) [[ZZ]]#0, [[ZZ]]#1, [[q2:%.+]]
}

// -----

func.func @test_parametrized_gate_arbitrary_angles_to_ppr(%q0 : !quantum.bit, %q1 : !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %cst = stablehlo.constant dense<0.42> : tensor<f64>
    %theta = tensor.extract %cst[] : tensor<f64>
    %rx = quantum.custom "RX"(%theta) %q0 {adjoint} : !quantum.bit
    %ry = quantum.custom "RY"(%theta) %rx : !quantum.bit
    %rz = quantum.custom "RZ"(%theta) %ry : !quantum.bit
    %xx:2 = quantum.custom "IsingXX"(%theta) %rz, %q1 : !quantum.bit, !quantum.bit
    %yy:2 = quantum.custom "IsingYY"(%theta) %xx#0, %xx#1 : !quantum.bit, !quantum.bit
    %zz:2 = quantum.custom "IsingZZ"(%theta) %yy#0, %yy#1 : !quantum.bit, !quantum.bit
    %mrz:2 = quantum.multirz(%theta) %zz#0, %zz#1 : !quantum.bit, !quantum.bit
    func.return %mrz#0, %mrz#1 : !quantum.bit, !quantum.bit

    // CHECK-DAG: [[NEG:%.+]] = arith.constant -2.100000e-01 : f64
    // CHECK-DAG: [[POS:%.+]] = arith.constant 2.100000e-01 : f64
    // CHECK: [[RX:%.+]] = pbc.ppr.arbitrary ["X"]([[NEG]]) [[q0:%.+]]
    // CHECK: [[RY:%.+]] = pbc.ppr.arbitrary ["Y"]([[POS]]) [[RX]]
    // CHECK: [[RZ:%.+]] = pbc.ppr.arbitrary ["Z"]([[POS]]) [[RY]]
    // CHECK: [[XX:%.+]]:2 = pbc.ppr.arbitrary ["X", "X"]([[POS]]) [[RZ]], [[q1:%.+]]
    // CHECK: [[YY:%.+]]:2 = pbc.ppr.arbitrary ["Y", "Y"]([[POS]]) [[XX]]#0, [[XX]]#1
    // CHECK: [[ZZ:%.+]]:2 = pbc.ppr.arbitrary ["Z", "Z"]([[POS]]) [[YY]]#0, [[YY]]#1
    // CHECK: [[MRZ:%.+]]:2 = pbc.ppr.arbitrary ["Z", "Z"]([[POS]]) [[ZZ]]#0, [[ZZ]]#1
}

// -----

func.func @test_parametrized_gate_identity_angle_elision_to_ppr(%q0 : !quantum.bit, %q1 : !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %cst = stablehlo.constant dense<0.0> : tensor<f64>
    %theta = tensor.extract %cst[] : tensor<f64>
    %rx = quantum.custom "RX"(%theta) %q0 : !quantum.bit
    %ry = quantum.custom "RY"(%theta) %rx : !quantum.bit
    %rz = quantum.custom "RZ"(%theta) %ry : !quantum.bit
    %xx:2 = quantum.custom "IsingXX"(%theta) %rz, %q1 : !quantum.bit, !quantum.bit
    %yy:2 = quantum.custom "IsingYY"(%theta) %xx#0, %xx#1 : !quantum.bit, !quantum.bit
    %zz:2 = quantum.custom "IsingZZ"(%theta) %yy#0, %yy#1 : !quantum.bit, !quantum.bit
    %mrz:2 = quantum.multirz(%theta) %zz#0, %zz#1 : !quantum.bit, !quantum.bit
    func.return %mrz#0, %mrz#1 : !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.custom
    // CHECK-NOT: quantum.multirz
    // CHECK-NOT: pbc.ppr
    // CHECK: return
}
