// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Test conversion to reference semantics quantum dialect for flat circuits.

// RUN: quantum-opt --convert-to-reference-semantics --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_static_alloc
func.func @test_static_alloc() attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK-NOT: quantum.alloc
    %0 = quantum.alloc( 2) : !quantum.reg

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %0 : !quantum.reg

    return
}


// -----


// CHECK-LABEL: test_dynamic_alloc
func.func @test_dynamic_alloc(%arg0: i64) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc(%arg0) : !qref.reg<?>
    // CHECK-NOT: quantum.alloc
    %0 = quantum.alloc(%arg0) : !quantum.reg

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<?>
    quantum.dealloc %0 : !quantum.reg

    return
}


// -----


// CHECK-LABEL: test_set_state
func.func @test_set_state(%arg0: tensor<4xcomplex<f64>>) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc( 2) : !quantum.reg

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    // CHECK-NOT: quantum.extract
    // CHECK: qref.set_state(%arg0) [[q0]], [[q1]] : tensor<4xcomplex<f64>>, !qref.bit, !qref.bit
    // CHECK-NOT: quantum.insert
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3:2 = quantum.set_state(%arg0) %1, %2 : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    %4 = quantum.insert %0[ 0], %3#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %3#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %5 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_set_basis_state
func.func @test_set_basis_state(%arg0: tensor<2xi1>) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc( 2) : !quantum.reg

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    // CHECK-NOT: quantum.extract
    // CHECK: qref.set_basis_state(%arg0) [[q0]], [[q1]] : tensor<2xi1>, !qref.bit, !qref.bit
    // CHECK-NOT: quantum.insert
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3:2 = quantum.set_basis_state(%arg0) %1, %2 : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    %4 = quantum.insert %0[ 0], %3#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %3#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %5 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_custom_op
func.func @test_custom_op(%arg0: f64, %arg1: f64, %arg2: i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.custom
    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: qref.custom "RX"(%arg0) [[q0]] : !qref.bit
    // CHECK-NOT: quantum.custom
    %out_qubits_0 = quantum.custom "RX"(%arg0) %out_qubits#0 : !quantum.bit

    // CHECK: qref.custom "some_gate"(%arg0, %arg1) [[q0]] adj ctrls([[q1]]) ctrlvals(%arg2) : !qref.bit ctrls !qref.bit
    // CHECK-NOT: quantum.custom
    %out_qubits_1, %out_ctrl_qubits = quantum.custom "some_gate"(%arg0, %arg1) %out_qubits_0 adj ctrls(%out_qubits#1) ctrlvals(%arg2) : !quantum.bit ctrls !quantum.bit

    // CHECK-NOT: quantum.insert
    %3 = quantum.insert %0[ 0], %out_qubits_1 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_ctrl_qubits : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %4 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_paulirot_op
func.func @test_paulirot_op(%arg0: f64, %arg1: i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.paulirot ["Z", "X"](%arg0) [[q0]], [[q1]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.paulirot
    %out_qubits:2 = quantum.paulirot ["Z", "X"](%arg0) %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: qref.paulirot ["Y"](%arg0) [[q0]] adj ctrls([[q1]]) ctrlvals(%arg1) : !qref.bit ctrls !qref.bit
    // CHECK-NOT: quantum.paulirot
    %out_qubits_0, %out_ctrl_qubits = quantum.paulirot ["Y"](%arg0) %out_qubits#0 adj ctrls(%out_qubits#1) ctrlvals(%arg1) : !quantum.bit ctrls !quantum.bit

    // CHECK-NOT: quantum.insert
    %3 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_ctrl_qubits : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %4 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_gphase
func.func @test_gphase(%arg0: f64, %arg1: i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc( 2) : !quantum.reg

    // CHECK: qref.gphase(%arg0)
    // CHECK-NOT: quantum.gphase
    quantum.gphase(%arg0)

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: qref.gphase(%arg0) ctrls([[q0]]) ctrlvals(%arg1) : ctrls !qref.bit
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.gphase(%arg0) ctrls(%1) ctrlvals(%arg1) : ctrls !quantum.bit

    // CHECK-NOT: quantum.insert
    %3 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %3 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_multirz_op
func.func @test_multirz_op(%arg0: f64, %arg1: i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.multirz(%arg0) [[q0]], [[q1]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.multirz
    %out_qubits:2 = quantum.multirz(%arg0) %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: qref.multirz(%arg0) [[q0]] ctrls([[q1]]) ctrlvals(%arg1) : !qref.bit ctrls !qref.bit
    // CHECK-NOT: quantum.multirz
    %out_qubits_0, %out_ctrl_qubits = quantum.multirz(%arg0) %out_qubits#0 ctrls(%out_qubits#1) ctrlvals(%arg1) : !quantum.bit ctrls !quantum.bit

    // CHECK-NOT: quantum.insert
    %3 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_ctrl_qubits : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %4 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_pcphase_op
func.func @test_pcphase_op(%arg0: f64, %arg1: i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.pcphase(%arg0, %arg0) [[q0]], [[q1]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.pcphase
    %out_qubits:2 = quantum.pcphase(%arg0, %arg0) %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: qref.pcphase(%arg0, %arg0) [[q0]] ctrls([[q1]]) ctrlvals(%arg1) : !qref.bit ctrls !qref.bit
    // CHECK-NOT: quantum.pcphase
    %out_qubits_0, %out_ctrl_qubits = quantum.pcphase(%arg0, %arg0) %out_qubits#0 ctrls(%out_qubits#1) ctrlvals(%arg1) : !quantum.bit ctrls !quantum.bit

    // CHECK-NOT: quantum.insert
    %3 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_ctrl_qubits : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %4 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_unitary_op
func.func @test_unitary_op(%arg0: tensor<4x4xcomplex<f64>>, %arg1: i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q2:%.+]] = qref.get [[qreg]][ 2] : !qref.reg<3> -> !qref.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    // CHECK: qref.unitary(%arg0 : tensor<4x4xcomplex<f64>>) [[q0]], [[q1]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.unitary
    %out_qubits:2 = quantum.unitary(%arg0 : tensor<4x4xcomplex<f64>>) %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: qref.unitary(%arg0 : tensor<4x4xcomplex<f64>>) [[q0]], [[q1]] adj ctrls([[q2]]) ctrlvals(%arg1) : !qref.bit, !qref.bit ctrls !qref.bit
    // CHECK-NOT: quantum.unitary
    %out_qubits_0:2, %out_ctrl_qubits = quantum.unitary(%arg0 : tensor<4x4xcomplex<f64>>) %out_qubits#0, %out_qubits#1 adj ctrls(%3) ctrlvals(%arg1) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // CHECK-NOT: quantum.insert
    %4 = quantum.insert %0[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 2], %out_ctrl_qubits : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc %6 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_compbasis_op
func.func @test_compbasis_op() -> (!quantum.obs, !quantum.obs) attributes {quantum.node} {
    // CHECK: [[qreg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[qreg3:%.+]] = qref.alloc( 3) : !qref.reg<3>
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.alloc( 3) : !quantum.reg

    // CHECK: [[q20:%.+]] = qref.get [[qreg2]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q21:%.+]] = qref.get [[qreg2]][ 1] : !qref.reg<2> -> !qref.bit
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.custom "CNOT"() [[q20]], [[q21]] : !qref.bit, !qref.bit
    %out_qubits:2 = quantum.custom "CNOT"() %2, %3 : !quantum.bit, !quantum.bit

    // CHECK: [[obs_qubits:%.+]] = qref.compbasis qubits [[q20]], [[q21]] : !quantum.obs
    // CHECK-NOT: quantum.compbasis
    %4 = quantum.compbasis qubits %out_qubits#0, %out_qubits#1 : !quantum.obs

    // CHECK-NOT: quantum.insert
    %5 = quantum.insert %0[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %out_qubits#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg2]] : !qref.reg<2>
    quantum.dealloc %6 : !quantum.reg

    // CHECK: [[obs_qreg:%.+]] = qref.compbasis(qreg [[qreg3]] : !qref.reg<3>) : !quantum.obs
    // CHECK-NOT: quantum.compbasis
    // CHECK: qref.dealloc [[qreg3]] : !qref.reg<3>
    %7 = quantum.compbasis qreg %1 : !quantum.obs
    quantum.dealloc %1 : !quantum.reg

    // CHECK: return [[obs_qubits]], [[obs_qreg]] : !quantum.obs, !quantum.obs
    return %4, %7 : !quantum.obs, !quantum.obs
}


// -----


// CHECK-LABEL: test_namedobs_op
func.func @test_namedobs_op() -> (!quantum.obs, !quantum.obs) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.insert
    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %3 = quantum.insert %0[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out_qubits#1 : !quantum.reg, !quantum.bit

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[obs_x:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    // CHECK-NOT: quantum.namedobs
    // CHECK-NOT: quantum.insert
    %5 = quantum.extract %4[ 0] : !quantum.reg -> !quantum.bit
    %6 = quantum.namedobs %5[ PauliX] : !quantum.obs
    %7 = quantum.insert %4[ 0], %5 : !quantum.reg, !quantum.bit

    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    // CHECK: [[obs_z:%.+]] = qref.namedobs [[q1]][ PauliZ] : !quantum.obs
    // CHECK-NOT: quantum.namedobs
    // CHECK-NOT: quantum.insert
    %8 = quantum.extract %7[ 1] : !quantum.reg -> !quantum.bit
    %9 = quantum.namedobs %8[ PauliZ] : !quantum.obs
    %10 = quantum.insert %7[ 1], %8 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %10 : !quantum.reg

    // CHECK: return [[obs_x]], [[obs_z]] : !quantum.obs, !quantum.obs
    return %6, %9 : !quantum.obs, !quantum.obs
}


// -----


// CHECK-LABEL: test_hermitian_op
func.func @test_hermitian_op(%arg0: tensor<4x4xcomplex<f64>>) -> !quantum.obs attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: [[obs:%.+]] = qref.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) [[q0]], [[q1]] : !quantum.obs
    // CHECK-NOT: quantum.hermitian
    // CHECK-NOT: quantum.insert
    %3 = quantum.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) %out_qubits#0, %out_qubits#1 : !quantum.obs
    %4 = quantum.insert %0[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %out_qubits#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %5 : !quantum.reg

    // CHECK: return [[obs]] : !quantum.obs
    return %3 : !quantum.obs
}


// -----


// CHECK-LABEL: test_measure_op
func.func @test_measure_op() -> (i1, i1, i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 1) : !qref.reg<1>
    // CHECK: [[qubit:%.+]] = qref.alloc_qb : !qref.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.alloc_qb : !quantum.bit

    // CHECK: [[mres:%.+]] = qref.measure [[qubit]] : i1
    %mres, %out_qubit = quantum.measure %1 : i1, !quantum.bit

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<1> -> !qref.bit
    // CHECK: [[mres_0:%.+]] = qref.measure [[q0]] : i1
    // CHECK: [[mres_2:%.+]] = qref.measure [[q0]] : i1
    // CHECK-NOT: quantum.insert
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %mres_0, %out_qubit_1 = quantum.measure %2 : i1, !quantum.bit
    %mres_2, %out_qubit_3 = quantum.measure %out_qubit_1 : i1, !quantum.bit
    %3 = quantum.insert %0[ 0], %out_qubit_3 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<1>
    // CHECK: qref.dealloc_qb [[qubit]] : !qref.bit
    quantum.dealloc %3 : !quantum.reg
    quantum.dealloc_qb %out_qubit : !quantum.bit

    // CHECK: return [[mres]], [[mres_0]], [[mres_2]] : i1, i1, i1
    return %mres, %mres_0, %mres_2 : i1, i1, i1
}


// -----


// CHECK-LABEL: test_multiple_allocs
func.func @test_multiple_allocs() attributes {quantum.node} {
    // CHECK: [[qreg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[qreg1:%.+]] = qref.alloc( 1) : !qref.reg<1>
    // CHECK: [[q10:%.+]] = qref.get [[qreg1]][ 0] : !qref.reg<1> -> !qref.bit
    // CHECK: [[q20:%.+]] = qref.get [[qreg2]][ 0] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.alloc( 1) : !quantum.reg
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: qref.custom "CNOT"() [[q10]], [[q20]] : !qref.bit, !qref.bit
    %out_qubits:2 = quantum.custom "CNOT"() %2, %3 : !quantum.bit, !quantum.bit

    // CHECK: [[q21:%.+]] = qref.get [[qreg2]][ 1] : !qref.reg<2> -> !qref.bit
    // CHECK: qref.custom "CNOT"() [[q10]], [[q21]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.insert
    %4 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#0, %4 : !quantum.bit, !quantum.bit
    %5 = quantum.insert %1[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit

    // CHECK: qref.custom "CNOT"() [[q20]], [[q21]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.insert
    %out_qubits_1:2 = quantum.custom "CNOT"() %out_qubits#1, %out_qubits_0#1 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %0[ 0], %out_qubits_1#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 1], %out_qubits_1#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg2]] : !qref.reg<2>
    // CHECK: qref.dealloc [[qreg1]] : !qref.reg<1>
    quantum.dealloc %7 : !quantum.reg
    quantum.dealloc %5 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_dynamic_wire_index
func.func @test_dynamic_wire_index(%arg0: i64) -> f64 attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc( 2) : !quantum.reg

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: qref.custom "PauliX"() [[q0]] : !qref.bit
    // CHECK-NOT: quantum.insert
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit

    // CHECK: [[qi:%.+]] = qref.get [[qreg]][%arg0] : !qref.reg<2>, i64 -> !qref.bit
    // CHECK: qref.custom "PauliZ"() [[qi]] : !qref.bit
    // CHECK: qref.custom "PauliY"() [[qi]] : !qref.bit
    // CHECK-NOT: quantum.insert
    %3 = quantum.extract %2[%arg0] : !quantum.reg -> !quantum.bit
    %out_qubits_0 = quantum.custom "PauliZ"() %3 : !quantum.bit
    %out_qubits_1 = quantum.custom "PauliY"() %out_qubits_0 : !quantum.bit
    %4 = quantum.insert %2[%arg0], %out_qubits_1 : !quantum.reg, !quantum.bit

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: qref.custom "PauliX"() [[q0]] : !qref.bit
    // CHECK: qref.custom "PauliY"() [[q0]] : !qref.bit
    // CHECK-NOT: quantum.insert
    %5 = quantum.extract %4[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits_2 = quantum.custom "PauliX"() %5 : !quantum.bit
    %out_qubits_3 = quantum.custom "PauliY"() %out_qubits_2 : !quantum.bit
    %6 = quantum.insert %4[ 0], %out_qubits_3 : !quantum.reg, !quantum.bit

    // CHECK: [[qi:%.+]] = qref.get [[qreg]][%arg0] : !qref.reg<2>, i64 -> !qref.bit
    // CHECK: [[obs:%.+]] = qref.namedobs [[qi]][ PauliX] : !quantum.obs
    // CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    // CHECK-NOT: quantum.insert
    %7 = quantum.extract %6[%arg0] : !quantum.reg -> !quantum.bit
    %8 = quantum.namedobs %7[ PauliX] : !quantum.obs
    %9 = quantum.insert %6[%arg0], %7 : !quantum.reg, !quantum.bit
    %10 = quantum.expval %8 : f64

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %9 : !quantum.reg

    // CHECK: return [[expval]] : f64
    return %10 : f64
}


// -----


// CHECK-LABEL: test_alloc_qb
func.func @test_alloc_qb() attributes {quantum.node} {
    // CHECK: [[qubit:%.+]] = qref.alloc_qb : !qref.bit
    // CHECK: qref.custom "X"() [[qubit]] : !qref.bit
    // CHECK: qref.dealloc_qb [[qubit]] : !qref.bit

    %0 = quantum.alloc_qb : !quantum.bit
    %out_qubits = quantum.custom "X"() %0 : !quantum.bit
    quantum.dealloc_qb %out_qubits : !quantum.bit
    return
}


// -----


// CHECK-LABEL: test_adjoint_op
func.func @test_adjoint_op() attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
    %3:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit

    // CHECK: qref.adjoint {
    // CHECK-NOT: ^bb
    // CHECK:    qref.custom "Hadamard"() [[q0]] : !qref.bit
    // CHECK:    qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
    %4:2 = quantum.adjoint(%3#0, %3#1) : !quantum.bit, !quantum.bit {
    ^bb0(%arg0: !quantum.bit, %arg1: !quantum.bit):
        %out_qubits = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits, %arg1 : !quantum.bit, !quantum.bit

        // CHECK-NOT: quantum.yield
        quantum.yield %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
    }

    // CHECK: qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
    // CHECK-NOT: quantum.insert
    %5:2 = quantum.custom "CNOT"() %4#0, %4#1 : !quantum.bit, !quantum.bit
    %6 = quantum.insert %0[ 0], %5#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 1], %5#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %7 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_adjoint_op_nested
func.func @test_adjoint_op_nested() attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: qref.adjoint {
    // CHECK-NOT: ^bb
    %3:2 = quantum.adjoint(%1, %2) : !quantum.bit, !quantum.bit {
    ^bb0(%arg0: !quantum.bit, %arg1: !quantum.bit):

        // CHECK: qref.custom "Hadamard"() [[q0]] : !qref.bit
        %out_qubits = quantum.custom "Hadamard"() %arg0 : !quantum.bit

        // CHECK: qref.adjoint {
        // CHECK-NOT: ^bb
        %6:2 = quantum.adjoint(%out_qubits, %arg1) : !quantum.bit, !quantum.bit {
        ^bb0(%arg2: !quantum.bit, %arg3: !quantum.bit):

            // CHECK: qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
            %out_qubits_1:2 = quantum.custom "CNOT"() %arg2, %arg3 : !quantum.bit, !quantum.bit

            // CHECK-NOT: quantum.yield
            quantum.yield %out_qubits_1#0, %out_qubits_1#1 : !quantum.bit, !quantum.bit
        }

        // CHECK: qref.custom "SWAP"() [[q0]], [[q1]] : !qref.bit, !qref.bit
        %out_qubits_0:2 = quantum.custom "SWAP"() %6#0, %6#1 : !quantum.bit, !quantum.bit

        // CHECK-NOT: quantum.yield
        quantum.yield %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
    }

    %4 = quantum.insert %0[ 0], %3#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %3#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %5 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_adjoint_with_allocation
func.func public @test_adjoint_with_allocation(%arg0: i64) attributes {quantum.node} {
    // CHECK: [[qreg4:%.+]] = qref.alloc( 4) : !qref.reg<4>
    // CHECK: [[qreg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc( 4) : !quantum.reg
    %1 = quantum.alloc( 2) : !quantum.reg

    // CHECK: qref.adjoint {
    // CHECK:   [[q4i:%.+]] = qref.get [[qreg4]][%arg0] : !qref.reg<4>, i64 -> !qref.bit
    // CHECK:   qref.custom "PauliX"() [[q4i]] : !qref.bit
    // CHECK: }
    %2 = quantum.adjoint(%0) : !quantum.reg {
    ^bb0(%arg1: !quantum.reg):
        %9 = quantum.extract %arg1[%arg0] : !quantum.reg -> !quantum.bit
        %out_qubits = quantum.custom "PauliX"() %9 : !quantum.bit
        %10 = quantum.insert %arg1[%arg0], %out_qubits : !quantum.reg, !quantum.bit
        quantum.yield %10 : !quantum.reg
    }


    // CHECK: [[q21:%.+]] = qref.get [[qreg2]][ 1] : !qref.reg<2> -> !qref.bit
    // CHECK: qref.adjoint {
    // CHECK:   qref.custom "PauliZ"() [[q21]] : !qref.bit
    // CHECK: }
    // CHECK: qref.dealloc [[qreg2]] : !qref.reg<2>
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    %4 = quantum.adjoint(%3) : !quantum.bit {
    ^bb0(%arg1: !quantum.bit):
        %out_qubits = quantum.custom "PauliZ"() %arg1 : !quantum.bit
        quantum.yield %out_qubits : !quantum.bit
    }
    %5 = quantum.insert %1[ 1], %4 : !quantum.reg, !quantum.bit
    quantum.dealloc %5 : !quantum.reg

    // CHECK: [[q43:%.+]] = qref.get [[qreg4]][ 3] : !qref.reg<4> -> !qref.bit
    %6 = quantum.extract %2[ 3] : !quantum.reg -> !quantum.bit

    // CHECK: qref.adjoint {
    %7 = quantum.adjoint(%6) : !quantum.bit {
    ^bb0(%arg1: !quantum.bit):

        // CHECK: [[qreg_inner:%.+]] = qref.alloc( 2) : !qref.reg<2>
        // CHECK: [[q0_inner:%.+]] = qref.get [[qreg_inner]][ 0] : !qref.reg<2> -> !qref.bit
        %9 = quantum.alloc( 2) : !quantum.reg
        %10 = quantum.extract %9[ 0] : !quantum.reg -> !quantum.bit

        // CHECK: qref.custom "PauliX"() [[q0_inner]] : !qref.bit
        %out_qubits = quantum.custom "PauliX"() %10 : !quantum.bit
        %11 = quantum.insert %9[ 0], %out_qubits : !quantum.reg, !quantum.bit

        // CHECK: [[q1_inner:%.+]] = qref.get [[qreg_inner]][ 1] : !qref.reg<2> -> !qref.bit
        // CHECK: qref.custom "CNOT"() [[q43]], [[q1_inner]] : !qref.bit, !qref.bit
        %12 = quantum.extract %11[ 1] : !quantum.reg -> !quantum.bit
        %out_qubits_0:2 = quantum.custom "CNOT"() %arg1, %12 : !quantum.bit, !quantum.bit
        %13 = quantum.insert %11[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit

        // CHECK: qref.dealloc [[qreg_inner]] : !qref.reg<2>
        quantum.dealloc %13 : !quantum.reg
        quantum.yield %out_qubits_0#0 : !quantum.bit
    }
    %8 = quantum.insert %2[ 3], %7 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg4]] : !qref.reg<4>
    quantum.dealloc %8 : !quantum.reg
    return
}


// -----

// CHECK-LABEL: test_operator_qubits
func.func @test_operator_qubits(%arg0: f64, %cv: i1, %fwd: i64) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc( 2) : !quantum.reg

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK:      qref.operator "MyOp"(%arg0: f64) adj qubits([[q0]], [[q1]])
    // CHECK-NEXT: static_data = {pauli_word = "XY"}
    // CHECK-NEXT: param_map = {theta = [0]} qubit_map = {pair = [0, 1]}
    %out1:2 = quantum.operator "MyOp"(%arg0: f64) adj qubits(%1, %2)
      static_data = {pauli_word = "XY"}
      param_map = {theta = [0]} qubit_map = {pair = [0, 1]}

    // CHECK:      qref.operator "MyOp"() qubits([[q0]])
    // CHECK-NEXT: ctrls([[q1]]) ctrl_vals(%arg1)
    %oq, %ocq = quantum.operator "MyOp"() qubits(%out1#0)
      ctrls(%out1#1) ctrl_vals(%cv)

    // CHECK:      qref.operator "MyOp"() qubits([[q0]], [[q1]])
    // CHECK-NEXT: UID(42) forward(%arg2: i64)
    %out3:2 = quantum.operator "MyOp"() qubits(%oq, %ocq)
      UID(42) forward(%fwd : i64)

    // CHECK-NOT: quantum.insert
    %3 = quantum.insert %0[ 0], %out3#0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %out3#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %4 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_operator_register
func.func @test_operator_register(%idx0: tensor<2xi64>, %idx1: tensor<1xi64>, %cidx: tensor<2xi64>, %cval: tensor<2xi1>) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 4) : !qref.reg<4>
    %0 = quantum.alloc( 4) : !quantum.reg

    // CHECK:      qref.operator "MyOp"()
    // CHECK-NEXT: quregs([[qreg]] : !qref.reg<4>) indices(%arg0: tensor<2xi64>, %arg1: tensor<1xi64>)
    // CHECK-NEXT: qubit_map = {qi0 = [0], qi1 = [1]}
    %out1 = quantum.operator "MyOp"()
      quregs(%0) indices(%idx0 : tensor<2xi64>, %idx1 : tensor<1xi64>)
      qubit_map = {qi0 = [0], qi1 = [1]}

    // CHECK:      qref.operator "MyOp"()
    // CHECK-NEXT: quregs([[qreg]] : !qref.reg<4>) indices(%arg0: tensor<2xi64>)
    // CHECK-NEXT: ctrls(%arg2: tensor<2xi64>) ctrl_vals(%arg3: tensor<2xi1>)
    %out2 = quantum.operator "MyOp"()
      quregs(%out1) indices(%idx0 : tensor<2xi64>)
      ctrls(%cidx : tensor<2xi64>) ctrl_vals(%cval : tensor<2xi1>)

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<4>
    quantum.dealloc %out2 : !quantum.reg
    return
}
