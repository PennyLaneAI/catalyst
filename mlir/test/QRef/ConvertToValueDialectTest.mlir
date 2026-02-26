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

// Test conversion to value semantics quantum dialect.
//
// RUN: quantum-opt --convert-to-value-semantics --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_flat_circuit
func.func @test_flat_circuit(%arg0: f64, %arg1: f64, %arg2: i1) -> f64 attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[bit0]], [[bit1]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[RX:%.+]] = quantum.custom "RX"(%arg0) [[CNOT]]#0 : !quantum.bit
    qref.custom "RX"(%arg0) %q0 : !qref.bit

    // CHECK: [[ROT:%.+]], [[ROTctrl:%.+]] = quantum.custom "Rot"(%arg0, %arg1) [[RX]] adj ctrls([[CNOT]]#1) ctrlvals(%arg2) : !quantum.bit ctrls !quantum.bit
    qref.custom "Rot"(%arg0, %arg1) %q0 adj ctrls (%q1) ctrlvals (%arg2) : !qref.bit ctrls !qref.bit

    // CHECK: [[PPR:%.+]], [[PPRctrl:%.+]] = quantum.paulirot ["Y"](%arg0) [[ROT]] ctrls([[ROTctrl]]) ctrlvals(%arg2) : !quantum.bit ctrls !quantum.bit
    qref.paulirot ["Y"](%arg0) %q0 ctrls (%q1) ctrlvals (%arg2) : !qref.bit ctrls !qref.bit

    // CHECK: quantum.gphase(%arg0)
    qref.gphase(%arg0) : f64

    // CHECK: [[GPHASEctrl:%.+]] = quantum.gphase(%arg0) ctrls([[PPR]]) ctrlvals(%arg2) : ctrls !quantum.bit
    qref.gphase(%arg0) ctrls (%q0) ctrlvals (%arg2) : f64 ctrls !qref.bit

    // CHECK: [[MULTIRZ:%.+]]:2 = quantum.multirz(%arg0) [[GPHASEctrl]], [[PPRctrl]] : !quantum.bit, !quantum.bit
    qref.multirz (%arg0) %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[PCPHASE:%.+]]:2 = quantum.pcphase(%arg0, %arg0) [[MULTIRZ]]#0, [[MULTIRZ]]#1 : !quantum.bit, !quantum.bit
    qref.pcphase (%arg0, %arg0) %q0, %q1 : !qref.bit, !qref.bit

    %matrix44 = tensor.empty() : tensor<4x4xcomplex<f64>>
    // CHECK: [[UNITARY:%.+]]:2 = quantum.unitary({{.+}}) [[PCPHASE]]#0, [[PCPHASE]]#1 : !quantum.bit, !quantum.bit
    qref.unitary (%matrix44 : tensor<4x4xcomplex<f64>>) %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[namedobs:%.+]] = quantum.namedobs [[UNITARY]]#1[ PauliX] : !quantum.obs
    // CHECK: quantum.expval [[namedobs]]
    %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[UNITARY]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[UNITARY]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return %expval : f64
}


// -----


// CHECK-LABEL: test_set_state
func.func @test_set_state(%arg0: tensor<4xcomplex<f64>>) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[SET_STATE:%.+]]:2 = quantum.set_state(%arg0) [[bit0]], [[bit1]] : (tensor<4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    qref.set_state(%arg0) %q0, %q1 : tensor<4xcomplex<f64>>, !qref.bit, !qref.bit

    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[SET_STATE]]#0, [[SET_STATE]]#1 : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[CNOT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[CNOT]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_set_basis_state
func.func @test_set_basis_state(%arg0: tensor<2xi1>) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[SET_BASIS_STATE:%.+]]:2 = quantum.set_basis_state(%arg0) [[bit0]], [[bit1]] : (tensor<2xi1>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    qref.set_basis_state(%arg0) %q0, %q1 : tensor<2xi1>, !qref.bit, !qref.bit

    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[SET_BASIS_STATE]]#0, [[SET_BASIS_STATE]]#1 : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[CNOT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[CNOT]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return
}


// -----


// CHECK-LABEL: test_dynamic_wire_index
func.func @test_dynamic_wire_index(%arg0: i64) -> f64 attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit_dynamic:%.+]] = quantum.extract [[qreg]][%arg0] : !quantum.reg -> !quantum.bit
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[%arg0] : !qref.reg<2>, i64 -> !qref.bit

    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[bit0]] : !quantum.bit
    qref.custom "PauliX"() %q0 : !qref.bit

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[X]] : !quantum.reg, !quantum.bit
    // CHECK: [[qreg:%.+]] = quantum.insert [[insert0]][%arg0], [[bit_dynamic]] : !quantum.reg, !quantum.bit
    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit_dynamic:%.+]] = quantum.extract [[qreg]][%arg0] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[bit0]], [[bit_dynamic]] : !quantum.bit, !quantum.bit
    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[CNOT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[qreg:%.+]] = quantum.insert [[insert0]][%arg0], [[CNOT]]#1 : !quantum.reg, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[Y:%.+]] = quantum.custom "PauliY"() [[bit0]] : !quantum.bit
    qref.custom "PauliY"() %q0 : !qref.bit

    // CHECK: [[bit_dynamic:%.+]] = quantum.extract [[qreg]][%arg0] : !quantum.reg -> !quantum.bit
    // CHECK: [[namedobs:%.+]] = quantum.namedobs [[bit_dynamic]][ PauliX] : !quantum.obs
    // CHECK: quantum.expval [[namedobs]]
    %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[Y]] : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][%arg0], [[bit_dynamic]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>
    return %expval : f64
}


// -----


// CHECK-LABEL: test_multiple_allocs
func.func @test_multiple_allocs() attributes {quantum.node} {

    // CHECK: [[r2:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r1:%.+]] = quantum.alloc( 1) : !quantum.reg
    %r2 = qref.alloc(2) : !qref.reg<2>
    %r1 = qref.alloc(1) : !qref.reg<1>

    // CHECK: [[q20:%.+]] = quantum.extract [[r2]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q21:%.+]] = quantum.extract [[r2]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[q10:%.+]] = quantum.extract [[r1]][ 0] : !quantum.reg -> !quantum.bit
    %q20 = qref.get %r2[0] : !qref.reg<2> -> !qref.bit
    %q21 = qref.get %r2[1] : !qref.reg<2> -> !qref.bit
    %q10 = qref.get %r1[0] : !qref.reg<1> -> !qref.bit

    // CHECK: [[CNOT0:%.+]]:2 = quantum.custom "CNOT"() [[q10]], [[q20]] : !quantum.bit, !quantum.bit
    // CHECK: [[CNOT1:%.+]]:2 = quantum.custom "CNOT"() [[CNOT0]]#0, [[q21]] : !quantum.bit, !quantum.bit
    // CHECK: [[CNOT2:%.+]]:2 = quantum.custom "CNOT"() [[CNOT0]]#1, [[CNOT1]]#1 : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q10, %q20 : !qref.bit, !qref.bit
    qref.custom "CNOT"() %q10, %q21: !qref.bit, !qref.bit
    qref.custom "CNOT"() %q20, %q21 : !qref.bit, !qref.bit

    // CHECK: [[insert20:%.+]] = quantum.insert [[r2]][ 0], [[CNOT2]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert21:%.+]] = quantum.insert [[insert20]][ 1], [[CNOT2]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert21]] : !quantum.reg
    qref.dealloc %r2 : !qref.reg<2>

    // CHECK: [[insert10:%.+]] = quantum.insert [[r1]][ 0], [[CNOT1]]#0 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert10]] : !quantum.reg
    qref.dealloc %r1 : !qref.reg<1>

    return
}


// -----


// CHECK-LABEL: test_for_loop
func.func @test_for_loop(%nqubits: i64) -> f64 attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = quantum.alloc(%arg0) : !quantum.reg
    %a = qref.alloc(%nqubits) : !qref.reg<?>

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[q0]] : !quantum.bit
    %q0 = qref.get %a[0] : !qref.reg<?> -> !qref.bit
    qref.custom "PauliX"() %q0 : !qref.bit

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = index.casts %nqubits : i64 to index

    // CHECK: [[preLoopInsert:%.+]] = quantum.insert [[qreg]][ 0], [[X]] : !quantum.reg, !quantum.bit
    // CHECK: [[loopOutReg:%.+]] = scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg2 = [[preLoopInsert]]) -> (!quantum.reg) {
    scf.for %i = %start to %stop step %step {
        // CHECK: [[indexCast:%.+]] = index.casts %arg1 : index to i64
        %int = index.casts %i : index to i64

        // CHECK: [[dynamicQ:%.+]] = quantum.extract %arg2[[[indexCast]]] : !quantum.reg -> !quantum.bit
        %this_q = qref.get %a[%int] : !qref.reg<?>, i64 -> !qref.bit

        // CHECK: [[dynamicIndexInsert:%.+]] = quantum.insert %arg2[[[indexCast]]], [[dynamicQ]] : !quantum.reg, !quantum.bit
        // CHECK: [[dynamicQ:%.+]] = quantum.extract [[dynamicIndexInsert]][[[indexCast]]] : !quantum.reg -> !quantum.bit
        // CHECK: [[Hadamard:%.+]] = quantum.custom "Hadamard"() [[dynamicQ]] : !quantum.bit
        // CHECK: [[dynamicQInsert:%.+]] = quantum.insert [[dynamicIndexInsert]][[[indexCast]]], [[Hadamard]] : !quantum.reg, !quantum.bit
        qref.custom "Hadamard"() %this_q : !qref.bit

        // CHECK: [[staticQ:%.+]] = quantum.extract [[dynamicQInsert]][ 0] : !quantum.reg -> !quantum.bit
        // CHECK: [[Z:%.+]] = quantum.custom "PauliZ"() [[staticQ]] : !quantum.bit
        qref.custom "PauliZ"() %q0 : !qref.bit

        // CHECK: [[staticQInsert:%.+]] = quantum.insert [[dynamicQInsert]][ 0], [[Z]] : !quantum.reg, !quantum.bit
        // CHECK: scf.yield [[staticQInsert]] : !quantum.reg
        scf.yield
    }

    // CHECK: [[extract:%.+]] = quantum.extract [[loopOutReg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[obs:%.+]] = quantum.namedobs [[extract]][ PauliX] : !quantum.obs
    // CHECK: quantum.expval [[obs]]
    %obs = qref.namedobs %q0 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[insert:%.+]] = quantum.insert [[loopOutReg]][ 0], [[extract]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc %a : !qref.reg<?>
    return %expval : f64
}


// -----


// CHECK-LABEL: test_for_loop_multiple_allocs
func.func @test_for_loop_multiple_allocs() -> f64 attributes {quantum.node} {
    // CHECK: [[r2:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r1:%.+]] = quantum.alloc( 1) : !quantum.reg
    %r2 = qref.alloc(2) : !qref.reg<2>
    %r1 = qref.alloc(1) : !qref.reg<1>

    // CHECK: [[q10:%.+]] = quantum.extract [[r1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[q10]] : !quantum.bit
    %q10 = qref.get %r1[0] : !qref.reg<1> -> !qref.bit
    qref.custom "PauliX"() %q10 : !qref.bit

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    // CHECK: [[preLoopInsert:%.+]] = quantum.insert [[r1]][ 0], [[X]] : !quantum.reg, !quantum.bit
    // CHECK: [[loopOutRegs:%.+]]:2 = scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg1 = [[r2]], %arg2 = [[preLoopInsert]])
    // CHECK-SAME:    -> (!quantum.reg, !quantum.reg) {
    scf.for %i = %start to %stop step %step {
        // CHECK: [[q20:%.+]] = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit
        %q20 = qref.get %r2[0] : !qref.reg<2> -> !qref.bit

        // CHECK: [[r3:%.+]] = quantum.alloc( 3) : !quantum.reg
        // CHECK: [[q30:%.+]] = quantum.extract [[r3]][ 0] : !quantum.reg -> !quantum.bit
        %r3 = qref.alloc(3) : !qref.reg<3>
        %q30 = qref.get %r3[0] : !qref.reg<3> -> !qref.bit

        // CHECK: [[q10:%.+]] = quantum.extract %arg2[ 0] : !quantum.reg -> !quantum.bit
        // CHECK: [[Toffoli:%.+]]:3 = quantum.custom "Toffoli"() [[q10]], [[q20]], [[q30]] : !quantum.bit, !quantum.bit, !quantum.bit
        qref.custom "Toffoli"() %q10, %q20, %q30 : !qref.bit, !qref.bit, !qref.bit

        // CHECK: [[insert3:%.+]] = quantum.insert [[r3]][ 0], [[Toffoli]]#2 : !quantum.reg, !quantum.bit
        // CHECK: quantum.dealloc [[insert3]] : !quantum.reg
        qref.dealloc %r3 : !qref.reg<3>

        // CHECK: [[insert2:%.+]] = quantum.insert %arg1[ 0], [[Toffoli]]#1 : !quantum.reg, !quantum.bit
        // CHECK: [[insert1:%.+]] = quantum.insert %arg2[ 0], [[Toffoli]]#0 : !quantum.reg, !quantum.bit
        // CHECK: scf.yield [[insert2]], [[insert1]] : !quantum.reg, !quantum.reg
        scf.yield
    }

    // CHECK: [[extract:%.+]] = quantum.extract [[loopOutRegs]]#1[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[obs:%.+]] = quantum.namedobs [[extract]][ PauliX] : !quantum.obs
    // CHECK: quantum.expval [[obs]]
    %obs = qref.namedobs %q10 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: quantum.dealloc [[loopOutRegs]]#0 : !quantum.reg
    // CHECK: [[insert:%.+]] = quantum.insert [[loopOutRegs]]#1[ 0], [[extract]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc %r2 : !qref.reg<2>
    qref.dealloc %r1 : !qref.reg<1>
    return %expval : f64
}


// -----


// CHECK-LABEL: test_for_loop_nested
func.func @test_for_loop_nested() -> f64 attributes {quantum.node} {
    // CHECK: [[r1:%.+]] = quantum.alloc( 1) : !quantum.reg
    %r1 = qref.alloc(1) : !qref.reg<1>

    // CHECK: [[q10:%.+]] = quantum.extract [[r1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[q10]] : !quantum.bit
    %q10 = qref.get %r1[0] : !qref.reg<1> -> !qref.bit
    qref.custom "PauliX"() %q10 : !qref.bit

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    // CHECK: [[r1Insert:%.+]] = quantum.insert [[r1]][ 0], [[X]] : !quantum.reg, !quantum.bit
    // CHECK: [[r1LoopOut:%.+]] = scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg1 = [[r1Insert]]) -> (!quantum.reg) {
    scf.for %i = %start to %stop step %step {
        // CHECK: [[r2:%.+]] = quantum.alloc( 2) : !quantum.reg
        // CHECK: [[q20:%.+]] = quantum.extract [[r2]][ 0] : !quantum.reg -> !quantum.bit
        %r2 = qref.alloc(2) : !qref.reg<2>
        %q20 = qref.get %r2[0] : !qref.reg<2> -> !qref.bit

        // CHECK: [[q10:%.+]] = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit
        // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[q10]], [[q20]] : !quantum.bit, !quantum.bit
        qref.custom "CNOT"() %q10, %q20 : !qref.bit, !qref.bit

        // CHECK: [[r1InsertInner:%.+]] = quantum.insert %arg1[ 0], [[CNOT]]#0 : !quantum.reg, !quantum.bit
        // CHECK: [[r2Insert:%.+]] = quantum.insert [[r2]][ 0], [[CNOT]]#1 : !quantum.reg, !quantum.bit
        // CHECK: [[innerLoopOuts:%.+]]:2 = scf.for %arg2 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg3 = [[r1InsertInner]], %arg4 = [[r2Insert]])
        // CHECK-SAME:   -> (!quantum.reg, !quantum.reg) {
        scf.for %j = %start to %stop step %step {
            // CHECK: [[r3:%.+]] = quantum.alloc( 3) : !quantum.reg
            // CHECK: [[q30:%.+]] = quantum.extract [[r3]][ 0] : !quantum.reg -> !quantum.bit
            %r3 = qref.alloc(3) : !qref.reg<3>
            %q30 = qref.get %r3[0] : !qref.reg<3> -> !qref.bit

            // CHECK: [[q10:%.+]] = quantum.extract %arg3[ 0] : !quantum.reg -> !quantum.bit
            // CHECK: [[q20:%.+]] = quantum.extract %arg4[ 0] : !quantum.reg -> !quantum.bit
            // CHECK: [[Toffoli:%.+]]:3 = quantum.custom "Toffoli"() [[q10]], [[q20]], [[q30]] : !quantum.bit, !quantum.bit, !quantum.bit
            qref.custom "Toffoli"() %q10, %q20, %q30 : !qref.bit, !qref.bit, !qref.bit

            // CHECK: [[insert3:%.+]] = quantum.insert [[r3]][ 0], [[Toffoli]]#2 : !quantum.reg, !quantum.bit
            // CHECK: quantum.dealloc [[insert3]] : !quantum.reg
            qref.dealloc %r3 : !qref.reg<3>

            // CHECK: [[insert1:%.+]] = quantum.insert %arg3[ 0], [[Toffoli]]#0 : !quantum.reg, !quantum.bit
            // CHECK: [[insert2:%.+]] = quantum.insert %arg4[ 0], [[Toffoli]]#1 : !quantum.reg, !quantum.bit
            // CHECK: scf.yield [[insert1]], [[insert2]] : !quantum.reg, !quantum.reg
            scf.yield
        }

        // CHECK: quantum.dealloc [[innerLoopOuts]]#1 : !quantum.reg
        qref.dealloc %r2 : !qref.reg<2>

        // CHECK: scf.yield [[innerLoopOuts]]#0 : !quantum.reg
        scf.yield
    }

    // CHECK: [[extract:%.+]] = quantum.extract [[r1LoopOut]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[obs:%.+]] = quantum.namedobs [[extract]][ PauliX] : !quantum.obs
    // CHECK: quantum.expval [[obs]]
    %obs = qref.namedobs %q10 [ PauliX] : !quantum.obs
    %expval = quantum.expval %obs : f64

    // CHECK: [[insert:%.+]] = quantum.insert [[r1LoopOut]][ 0], [[extract]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc %r1 : !qref.reg<1>
    return %expval : f64
}
