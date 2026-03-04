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

// Test conversion to value semantics quantum dialect for subroutines.

// RUN: quantum-opt --convert-to-value-semantics --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK: func.func @test_qubit_args(%arg0: !quantum.bit, %arg1: !quantum.bit, %arg2: i1) ->
// CHECK-SAME:   (!quantum.obs, !quantum.bit, !quantum.bit) attributes {quantum.node}
func.func @test_qubit_args(%q0: !qref.bit, %q1: !qref.bit, %arg2: i1) -> !quantum.obs attributes {quantum.node} {

    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() %arg0, %arg1 : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

    // CHECK: [[X:%.+]] = quantum.custom "X"() [[CNOT]]#0 : !quantum.bit
    qref.custom "X"() %q0 : !qref.bit

    // CHECK: [[ROT:%.+]], [[ROTctrl:%.+]] = quantum.custom "some_gate"() [[X]] adj ctrls([[CNOT]]#1) ctrlvals(%arg2) : !quantum.bit ctrls !quantum.bit
    qref.custom "some_gate"() %q0 adj ctrls (%q1) ctrlvals (%arg2) : !qref.bit ctrls !qref.bit

    // CHECK: [[obs:%.+]] = quantum.compbasis qubits [[ROT]], [[ROTctrl]] : !quantum.obs
    %obs_q = qref.compbasis qubits %q0, %q1 : !quantum.obs

    // CHECK: return [[obs]], [[ROT]], [[ROTctrl]] : !quantum.obs, !quantum.bit, !quantum.bit
    return %obs_q : !quantum.obs
}


// -----


// CHECK: func.func @test_qreg_args(%arg0: !quantum.reg, %arg1: !quantum.reg, %arg2: f64) ->
// CHECK-SAME:   (!quantum.reg, !quantum.reg) attributes {quantum.node}
func.func @test_qreg_args(%r0: !qref.reg<2>, %r1: !qref.reg<?>, %arg2: f64) attributes {quantum.node} {

    %q00 = qref.get %r0[0] : !qref.reg<2> -> !qref.bit
    %q01 = qref.get %r0[1] : !qref.reg<2> -> !qref.bit
    %q11 = qref.get %r1[1] : !qref.reg<?> -> !qref.bit

    // CHECK: [[q00:%.+]] = quantum.extract %arg0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q01:%.+]] = quantum.extract %arg0[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[q00]], [[q01]] : !quantum.bit, !quantum.bit
    qref.custom "CNOT"() %q00, %q01 : !qref.bit, !qref.bit

    // CHECK: [[q11:%.+]] = quantum.extract %arg1[ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[RX:%.+]] = quantum.custom "RX"(%arg2) [[q11]] : !quantum.bit
    qref.custom "RX"(%arg2) %q11 : !qref.bit

    // CHECK: [[Toffoli:%.+]]:3 = quantum.custom "Toffoli"() [[CNOT]]#0, [[CNOT]]#1, [[RX]] : !quantum.bit, !quantum.bit, !quantum.bit
    qref.custom "Toffoli"() %q00, %q01, %q11 : !qref.bit, !qref.bit, !qref.bit

    // CHECK: [[insert00:%.+]] = quantum.insert %arg0[ 0], [[Toffoli]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert01:%.+]] = quantum.insert [[insert00]][ 1], [[Toffoli]]#1 : !quantum.reg, !quantum.bit
    // CHECK: [[insert11:%.+]] = quantum.insert %arg1[ 1], [[Toffoli]]#2 : !quantum.reg, !quantum.bit
    // CHECK: return [[insert01]], [[insert11]] : !quantum.reg, !quantum.reg
    return
}


// -----


// CHECK: func.func @test_qreg_and_qubit_args(%arg0: f64, %arg1: !quantum.reg, %arg2: !quantum.bit, %arg3: !quantum.bit) ->
// CHECK-SAME:   (!quantum.reg, !quantum.bit, !quantum.bit) attributes {quantum.node}
func.func @test_qreg_and_qubit_args(%arg0: f64, %r0: !qref.reg<2>, %q0: !qref.bit, %q1: !qref.bit) attributes {quantum.node} {

    %q00 = qref.get %r0[0] : !qref.reg<2> -> !qref.bit

    // CHECK: [[q00:%.+]] = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[ROT:%.+]]:3 = quantum.custom "Rot"(%arg0) [[q00]], %arg2, %arg3 : !quantum.bit, !quantum.bit, !quantum.bit
    qref.custom "Rot"(%arg0) %q00, %q0, %q1 : !qref.bit, !qref.bit, !qref.bit

    // CHECK: [[insert:%.+]] = quantum.insert %arg1[ 0], [[ROT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: return [[insert]], [[ROT]]#1, [[ROT]]#2 : !quantum.reg, !quantum.bit, !quantum.bit
    return
}


// -----


// CHECK: func.func @test_qubit_args_with_loop(%arg0: !quantum.bit) -> !quantum.bit
func.func @test_qubit_args_with_loop(%q: !qref.bit) attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    scf.for %i = %start to %stop step %step {
        qref.custom "Hadamard"() %q : !qref.bit
        scf.yield
    }
    return

    // CHECK: [[loopOut:%.+]] = scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg2 = %arg0) -> (!quantum.bit) {
    // CHECK:   [[out_qubits:%.+]] = quantum.custom "Hadamard"() %arg2 : !quantum.bit
    // CHECK:   scf.yield [[out_qubits]] : !quantum.bit
    // CHECK: }
    // CHECK: return [[loopOut]] : !quantum.bit
}
