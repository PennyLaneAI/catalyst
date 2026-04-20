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

// Test conversion to value semantics quantum dialect for flat circuits with only classical arguments.

// RUN: quantum-opt --convert-to-value-semantics --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_set_state
func.func @test_set_state(%arg0: tensor<4xcomplex<f64>>) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[SET_STATE:%.+]]:2 = quantum.set_state(%arg0) [[bit0]], [[bit1]] : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[SET_STATE]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[SET_STATE]]#1 : !quantum.reg, !quantum.bit
    qref.set_state(%arg0) %q0, %q1 : tensor<4xcomplex<f64>>, !qref.bit, !qref.bit

    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_set_basis_state
func.func @test_set_basis_state(%arg0: tensor<2xi1>) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[SET_BASIS_STATE:%.+]]:2 = quantum.set_basis_state(%arg0) [[bit0]], [[bit1]] : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[SET_BASIS_STATE]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[SET_BASIS_STATE]]#1 : !quantum.reg, !quantum.bit
    qref.set_basis_state(%arg0) %q0, %q1 : tensor<2xi1>, !qref.bit, !qref.bit

    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_custom_op
func.func @test_custom_op(%arg0: f64, %arg1: f64, %arg2: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[bit0]], [[bit1]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[RX:%.+]] = quantum.custom "RX"(%arg0) [[CNOT]]#0 : !quantum.bit
    qref.custom "RX"(%arg0) %q0 : !qref.bit

    // CHECK: [[ROT:%.+]], [[ROTctrl:%.+]] = quantum.custom "some_gate"(%arg0, %arg1) [[RX]] adj ctrls([[CNOT]]#1) ctrlvals(%arg2) : !quantum.bit ctrls !quantum.bit
    qref.custom "some_gate"(%arg0, %arg1) %q0 adj ctrls (%q1) ctrlvals (%arg2) : !qref.bit ctrls !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[ROT]] : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[ROTctrl]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_paulirot_op
func.func @test_paulirot_op(%arg0: f64, %arg1: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[ZX:%.+]]:2 = quantum.paulirot ["Z", "X"](%arg0) [[bit0]], [[bit1]] : !quantum.bit, !quantum.bit
    qref.paulirot ["Z", "X"](%arg0) %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[Y:%.+]], [[Yctrl:%.+]] = quantum.paulirot ["Y"](%arg0) [[ZX]]#0 adj ctrls([[ZX]]#1) ctrlvals(%arg1) : !quantum.bit ctrls !quantum.bit
    qref.paulirot ["Y"](%arg0) %q0 adj ctrls (%q1) ctrlvals (%arg1) : !qref.bit ctrls !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[Y]] : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[Yctrl]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_gphase
func.func @test_gphase(%arg0: f64, %arg1: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit

    // CHECK: quantum.gphase(%arg0)
    qref.gphase(%arg0)

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[GPHASEctrl:%.+]] = quantum.gphase(%arg0) ctrls([[bit0]]) ctrlvals(%arg1) : ctrls !quantum.bit
    qref.gphase(%arg0) ctrls (%q0) ctrlvals (%arg1) : ctrls !qref.bit

    // CHECK: [[insert:%.+]] = quantum.insert [[qreg]][ 0], [[GPHASEctrl]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_multirz_op
func.func @test_multirz_op(%arg0: f64, %arg1: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[gate0:%.+]]:2 = quantum.multirz(%arg0) [[bit0]], [[bit1]] : !quantum.bit, !quantum.bit
    qref.multirz (%arg0) %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[gate1:%.+]], [[gate1Ctrl:%.+]] = quantum.multirz(%arg0) [[gate0]]#0 ctrls([[gate0]]#1) ctrlvals(%arg1) : !quantum.bit ctrls !quantum.bit
    qref.multirz (%arg0) %q0 ctrls (%q1) ctrlvals (%arg1) : !qref.bit ctrls !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[gate1]] : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[gate1Ctrl]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_pcphase_op
func.func @test_pcphase_op(%arg0: f64, %arg1: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[gate0:%.+]]:2 = quantum.pcphase(%arg0, %arg0) [[bit0]], [[bit1]] : !quantum.bit, !quantum.bit
    qref.pcphase (%arg0, %arg0) %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[gate1:%.+]], [[gate1Ctrl:%.+]] = quantum.pcphase(%arg0, %arg0) [[gate0]]#0 ctrls([[gate0]]#1) ctrlvals(%arg1) : !quantum.bit ctrls !quantum.bit
    qref.pcphase (%arg0, %arg0) %q0 ctrls (%q1) ctrlvals (%arg1) : !qref.bit ctrls !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[gate1]] : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[gate1Ctrl]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_unitary_op
func.func @test_unitary_op(%arg0: tensor<4x4xcomplex<f64>>, %arg1: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 3) : !quantum.reg
    %a = qref.alloc(3) : !qref.reg<3>
    %q0 = qref.get %a[0] : !qref.reg<3> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<3> -> !qref.bit
    %q2 = qref.get %a[2] : !qref.reg<3> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[gate0:%.+]]:2 = quantum.unitary(%arg0 : tensor<4x4xcomplex<f64>>) [[bit0]], [[bit1]] : !quantum.bit, !quantum.bit
    qref.unitary (%arg0 : tensor<4x4xcomplex<f64>>) %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[bit2:%.+]] = quantum.extract [[qreg]][ 2] : !quantum.reg -> !quantum.bit
    // CHECK: [[gate1:%.+]]:2, [[gate1Ctrl:%.+]] = quantum.unitary(%arg0 : tensor<4x4xcomplex<f64>>) [[gate0]]#0, [[gate0]]#1
    // CHECK-SAME:  adj ctrls([[bit2]]) ctrlvals(%arg1)
    qref.unitary (%arg0 : tensor<4x4xcomplex<f64>>) %q0, %q1 adj ctrls (%q2) ctrlvals (%arg1) : !qref.bit, !qref.bit ctrls !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[gate1]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[gate1]]#1 : !quantum.reg, !quantum.bit
    // CHECK: [[insert2:%.+]] = quantum.insert [[insert1]][ 2], [[gate1Ctrl]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert2]] : !quantum.reg
    qref.dealloc %a : !qref.reg<3>
    return
}

// -----


// CHECK-LABEL: test_compbasis_op
func.func @test_compbasis_op() -> (!quantum.obs, !quantum.obs) attributes {quantum.node} {

    // CHECK: [[r2:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r3:%.+]] = quantum.alloc( 3) : !quantum.reg
    %r2 = qref.alloc(2) : !qref.reg<2>
    %r3 = qref.alloc(3) : !qref.reg<3>
    %q20 = qref.get %r2[0] : !qref.reg<2> -> !qref.bit
    %q21 = qref.get %r2[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[q20:%.+]] = quantum.extract [[r2]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q21:%.+]] = quantum.extract [[r2]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[q20]], [[q21]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q20, %q21 : !qref.bit, !qref.bit

    // CHECK: [[obs_q:%.+]] = quantum.compbasis qubits [[CNOT]]#0, [[CNOT]]#1 : !quantum.obs
    %obs_q = qref.compbasis qubits %q20, %q21 : !quantum.obs

    // CHECK: [[insert0:%.+]] = quantum.insert [[r2]][ 0], [[CNOT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[CNOT]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %r2 : !qref.reg<2>

    // CHECK: [[obs_r:%.+]] = quantum.compbasis qreg [[r3]] : !quantum.obs
    %obs_r = qref.compbasis (qreg %r3 : !qref.reg<3>) : !quantum.obs

    // CHECK: quantum.dealloc [[r3]] : !quantum.reg
    qref.dealloc %r3 : !qref.reg<3>

    // CHECK: return [[obs_q]], [[obs_r]] : !quantum.obs, !quantum.obs
    return %obs_q, %obs_r : !quantum.obs, !quantum.obs
}


// -----


// CHECK-LABEL: test_namedobs_op
func.func @test_namedobs_op() -> (!quantum.obs, !quantum.obs) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[q0]], [[q1]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // COM: TODO: improve canonicalization patterns to recognize inverse extract-insert pairs where
    // COM: inserts are delayed past guaranteed distinct extracts (or vice versa), via statically
    // COM: different indices
    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[CNOT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[CNOT]]#1 : !quantum.reg, !quantum.bit

    // CHECK: [[extract:%.+]] = quantum.extract [[insert1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[obs_x:%.+]] = quantum.namedobs [[extract]][ PauliX] : !quantum.obs
    // CHECK: [[insertX:%.+]] = quantum.insert [[insert1]][ 0], [[extract]] : !quantum.reg, !quantum.bit
    %obs_x = qref.namedobs %q0 [ PauliX] : !quantum.obs

    // CHECK: [[extract:%.+]] = quantum.extract [[insertX]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[obs_z:%.+]] = quantum.namedobs [[extract]][ PauliZ] : !quantum.obs
    // CHECK: [[insertZ:%.+]] = quantum.insert [[insertX]][ 1], [[extract]] : !quantum.reg, !quantum.bit
    %obs_z = qref.namedobs %q1 [ PauliZ] : !quantum.obs

    // CHECK: quantum.dealloc [[insertZ]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>

    // CHECK: return [[obs_x]], [[obs_z]]
    return %obs_x, %obs_z : !quantum.obs , !quantum.obs
}


// -----


// CHECK-LABEL: test_hermitian_op
func.func @test_hermitian_op(%arg0: tensor<4x4xcomplex<f64>>) -> !quantum.obs attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[q0]], [[q1]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[obs:%.+]] = quantum.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) [[CNOT]]#0, [[CNOT]]#1 : !quantum.obs
    %obs = qref.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) %q0, %q1 : !quantum.obs

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[CNOT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[CNOT]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>

    // CHECK: return [[obs]]
    return %obs : !quantum.obs
}


// -----


// CHECK-LABEL: test_measure_op
func.func @test_measure_op() -> (i1, i1, i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 1) : !quantum.reg
    // CHECK: [[qb:%.+]] = quantum.alloc_qb : !quantum.bit
    %a = qref.alloc(1) : !qref.reg<1>
    %q0 = qref.get %a[0] : !qref.reg<1> -> !qref.bit
    %qb = qref.alloc_qb : !qref.bit

    // CHECK: [[mres0:%.+]], [[qb_0:%.+]] = quantum.measure [[qb]] : i1, !quantum.bit
    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[mres1:%.+]], [[q0_0:%.+]] = quantum.measure [[q0]] : i1, !quantum.bit
    // CHECK: [[mres2:%.+]], [[q0_1:%.+]] = quantum.measure [[q0_0]] : i1, !quantum.bit
    %mres0 = qref.measure %qb : i1
    %mres1 = qref.measure %q0 : i1
    %mres2 = qref.measure %q0 : i1

    // CHECK: [[insert:%.+]] = quantum.insert [[qreg]][ 0], [[q0_1]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    // CHECK: quantum.dealloc_qb [[qb_0]] : !quantum.bit
    qref.dealloc %a : !qref.reg<1>
    qref.dealloc_qb %qb : !qref.bit

    // CHECK: return [[mres0]], [[mres1]], [[mres2]] : i1, i1, i1
    return %mres0, %mres1, %mres2 : i1, i1, i1
}


// -----


// CHECK-LABEL: test_multiple_allocs
func.func @test_multiple_allocs() attributes {quantum.node} {

    // CHECK: [[r2:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r1:%.+]] = quantum.alloc( 1) : !quantum.reg
    %r2 = qref.alloc(2) : !qref.reg<2>
    %r1 = qref.alloc(1) : !qref.reg<1>
    %q20 = qref.get %r2[0] : !qref.reg<2> -> !qref.bit
    %q21 = qref.get %r2[1] : !qref.reg<2> -> !qref.bit
    %q10 = qref.get %r1[0] : !qref.reg<1> -> !qref.bit

    // CHECK: [[q10:%.+]] = quantum.extract [[r1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q20:%.+]] = quantum.extract [[r2]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT0:%.+]]:2 = quantum.custom "CNOT"() [[q10]], [[q20]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q10, %q20 : !qref.bit, !qref.bit

    // CHECK: [[q21:%.+]] = quantum.extract [[r2]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT1:%.+]]:2 = quantum.custom "CNOT"() [[CNOT0]]#0, [[q21]] : !quantum.bit, !quantum.bit
    // CHECK: [[insert10:%.+]] = quantum.insert [[r1]][ 0], [[CNOT1]]#0 : !quantum.reg, !quantum.bit
    qref.custom "CNOT"() %q10, %q21: !qref.bit, !qref.bit

    // CHECK: [[CNOT2:%.+]]:2 = quantum.custom "CNOT"() [[CNOT0]]#1, [[CNOT1]]#1 : !quantum.bit, !quantum.bit
    // CHECK: [[insert20:%.+]] = quantum.insert [[r2]][ 0], [[CNOT2]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert21:%.+]] = quantum.insert [[insert20]][ 1], [[CNOT2]]#1 : !quantum.reg, !quantum.bit
    qref.custom "CNOT"() %q20, %q21 : !qref.bit, !qref.bit


    // CHECK: quantum.dealloc [[insert21]] : !quantum.reg
    qref.dealloc %r2 : !qref.reg<2>

    // CHECK: quantum.dealloc [[insert10]] : !quantum.reg
    qref.dealloc %r1 : !qref.reg<1>

    return
}


// -----


// CHECK-LABEL: test_dynamic_wire_index
func.func @test_dynamic_wire_index(%arg0: i64) -> f64 attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[%arg0] : !qref.reg<2>, i64 -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[bit0]] : !quantum.bit
    // CHECK: [[insertX:%.+]] = quantum.insert [[qreg]][ 0], [[X]] : !quantum.reg, !quantum.bit
    qref.custom "PauliX"() %q0 : !qref.bit

    // CHECK: [[bit_dynamic:%.+]] = quantum.extract [[insertX]][%arg0] : !quantum.reg -> !quantum.bit
    // CHECK: [[Z:%.+]] = quantum.custom "PauliZ"() [[bit_dynamic]] : !quantum.bit
    // CHECK: [[Y:%.+]] = quantum.custom "PauliY"() [[Z]] : !quantum.bit
    // CHECK: [[insert_dynamic:%.+]] = quantum.insert [[insertX]][%arg0], [[Y]] : !quantum.reg, !quantum.bit
    qref.custom "PauliZ"() %q1 : !qref.bit
    qref.custom "PauliY"() %q1 : !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[insert_dynamic]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[bit0]] : !quantum.bit
    // CHECK: [[Y:%.+]] = quantum.custom "PauliY"() [[X]] : !quantum.bit
    // CHECK: [[insertY:%.+]] = quantum.insert [[insert_dynamic]][ 0], [[Y]] : !quantum.reg, !quantum.bit
    qref.custom "PauliX"() %q0 : !qref.bit
    qref.custom "PauliY"() %q0 : !qref.bit

    // CHECK: [[bit_dynamic:%.+]] = quantum.extract [[insertY]][%arg0] : !quantum.reg -> !quantum.bit
    // CHECK: [[namedobs:%.+]] = quantum.namedobs [[bit_dynamic]][ PauliX] : !quantum.obs
    // CHECK: [[insert_dealloc:%.+]] = quantum.insert [[insertY]][%arg0], [[bit_dynamic]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.expval [[namedobs]]
    %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: quantum.dealloc [[insert_dealloc]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return %expval : f64
}


// -----


// CHECK-LABEL: test_alloc_qb
func.func @test_alloc_qb() attributes {quantum.node} {

    // CHECK: [[qubit:%.+]] = quantum.alloc_qb : !quantum.bit
    %q = qref.alloc_qb : !qref.bit

    // CHECK: [[X:%.+]] = quantum.custom "X"() [[qubit]] : !quantum.bit
    qref.custom "X"() %q : !qref.bit

    // CHECK: quantum.dealloc_qb [[X]] : !quantum.bit
    qref.dealloc_qb %q : !qref.bit
    return
}


// -----


// CHECK-LABEL: test_adjoint_op
func.func @test_adjoint_op() attributes {quantum.node}
{
    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %r = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[ADJOINT:%.+]]:2 = quantum.adjoint([[q0]], [[q1]]) : !quantum.bit, !quantum.bit {
    // CHECK: ^bb0(%arg0: !quantum.bit, %arg1: !quantum.bit):
    qref.adjoint {
    ^bb0():
        %q0 = qref.get %r[0] : !qref.reg<2> -> !qref.bit
        %q1 = qref.get %r[1] : !qref.reg<2> -> !qref.bit
        qref.custom "Hadamard"() %q0 : !qref.bit
        qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

        // CHECK: [[HADAMARD:%.+]] = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[HADAMARD]], %arg1 : !quantum.bit, !quantum.bit
        // CHECK: quantum.yield [[CNOT]]#0, [[CNOT]]#1 : !quantum.bit, !quantum.bit
    }
    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[ADJOINT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[ADJOINT]]#1 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %r : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_adjoint_op_nested
func.func @test_adjoint_op_nested() attributes {quantum.node}
{
    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %r = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[OUTER_ADJ:%.+]]:2 = quantum.adjoint([[q0]], [[q1]]) : !quantum.bit, !quantum.bit {
    // CHECK: ^bb0(%arg0: !quantum.bit, %arg1: !quantum.bit):
    qref.adjoint {
    ^bb0():
        %q0 = qref.get %r[0] : !qref.reg<2> -> !qref.bit

        // CHECK: [[HADAMARD:%.+]] = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        qref.custom "Hadamard"() %q0 : !qref.bit

        // CHECK: [[INNER_ADJ:%.+]]:2 = quantum.adjoint([[HADAMARD]], %arg1) : !quantum.bit, !quantum.bit {
        // CHECK: ^bb0(%arg2: !quantum.bit, %arg3: !quantum.bit):
        qref.adjoint {
        ^bb0():
            %q0_cnot = qref.get %r[0] : !qref.reg<2> -> !qref.bit
            %q1_cnot = qref.get %r[1] : !qref.reg<2> -> !qref.bit

            // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() %arg2, %arg3 : !quantum.bit, !quantum.bit
            qref.custom "CNOT"() %q0_cnot, %q1_cnot : !qref.bit, !qref.bit

            // CHECK: quantum.yield [[CNOT]]#0, [[CNOT]]#1 : !quantum.bit, !quantum.bit
        }

        %q0_swap = qref.get %r[0] : !qref.reg<2> -> !qref.bit
        %q1_swap = qref.get %r[1] : !qref.reg<2> -> !qref.bit

        // CHECK: [[SWAP:%.+]]:2 = quantum.custom "SWAP"() [[INNER_ADJ]]#0, [[INNER_ADJ]]#1 : !quantum.bit, !quantum.bit
        qref.custom "SWAP"() %q0_swap, %q1_swap : !qref.bit, !qref.bit

        // CHECK: quantum.yield [[SWAP]]#0, [[SWAP]]#1 : !quantum.bit, !quantum.bit
    }
    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[OUTER_ADJ]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[OUTER_ADJ]]#1 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %r : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_aliasing_getops
func.func @test_aliasing_getops() attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT1:%.+]]:2 = quantum.custom "CNOT"() [[bit0]], [[bit1]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    %q0_again = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1_again = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[CNOT2:%.+]]:2 = quantum.custom "CNOT"() [[CNOT1]]#0, [[CNOT1]]#1 : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0_again, %q1_again : !qref.bit, !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[CNOT2]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[CNOT2]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}
