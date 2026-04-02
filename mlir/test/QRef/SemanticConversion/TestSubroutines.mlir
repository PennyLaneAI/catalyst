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



// CHECK: func.func @test_extract_before_call(%arg0: f64, %arg1: !quantum.bit, %arg2: !quantum.bit, %arg3: !quantum.bit) ->
// CHECK-SAME:    (!quantum.bit, !quantum.bit, !quantum.bit)
// CHECK:     [[CNOT:%.+]]:2 = quantum.custom "CNOT"() %arg1, %arg2 : !quantum.bit, !quantum.bit
// CHECK:     [[RX:%.+]] = quantum.custom "RX"(%arg0) %arg3 : !quantum.bit
// CHECK:     [[TOFFOLI:%.+]]:3 = quantum.custom "Toffoli"() [[CNOT]]#0, [[CNOT]]#1, [[RX]] : !quantum.bit, !quantum.bit, !quantum.bit
// CHECK:     return [[TOFFOLI]]#0, [[TOFFOLI]]#1, [[TOFFOLI]]#2 : !quantum.bit, !quantum.bit, !quantum.bit
// CHECK: }

func.func @test_extract_before_call(%r0: !qref.reg<2>, %r1: !qref.reg<?>, %arg2: f64) {
    %q00 = qref.get %r0[0] : !qref.reg<2> -> !qref.bit
    %q01 = qref.get %r0[1] : !qref.reg<2> -> !qref.bit
    %q11 = qref.get %r1[1] : !qref.reg<?> -> !qref.bit

    qref.custom "CNOT"() %q00, %q01 : !qref.bit, !qref.bit
    qref.custom "RX"(%arg2) %q11 : !qref.bit
    qref.custom "Toffoli"() %q00, %q01, %q11 : !qref.bit, !qref.bit, !qref.bit
    return
}

// CHECK: func.func @main(%arg0: i64, %arg1: f64) -> (!quantum.obs, !quantum.obs) attributes {quantum.node}
func.func @main(%arg0: i64, %arg1: f64) -> (!quantum.obs, !quantum.obs) attributes {quantum.node} {
    // CHECK: [[r2:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r_dyn:%.+]] = quantum.alloc(%arg0) : !quantum.reg
    %r2 = qref.alloc(2) : !qref.reg<2>
    %r_dyn = qref.alloc(%arg0) : !qref.reg<?>


    // CHECK: [[q20:%.+]] = quantum.extract [[r2]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q21:%.+]] = quantum.extract [[r2]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[q_dyn1:%.+]] = quantum.extract [[r_dyn]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[first_call:%.+]]:3 = call @test_extract_before_call(%arg1, [[q20]], [[q21]], [[q_dyn1]])
    // CHECK-SAME:   (f64, !quantum.bit, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit)
    // CHECK: [[second_call:%.+]]:3 = call @test_extract_before_call(%arg1, [[first_call]]#0, [[first_call]]#1, [[first_call]]#2)
    // CHECK-SAME:   (f64, !quantum.bit, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit)
    func.call @test_extract_before_call(%r2, %r_dyn, %arg1) : (!qref.reg<2>, !qref.reg<?>, f64) -> ()
    func.call @test_extract_before_call(%r2, %r_dyn, %arg1) : (!qref.reg<2>, !qref.reg<?>, f64) -> ()
    // CHECK: [[insert_20:%.+]] = quantum.insert [[r2]][ 0], [[second_call]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert_21:%.+]] = quantum.insert [[insert_20]][ 1], [[second_call]]#1 : !quantum.reg, !quantum.bit
    // CHECK: [[insert_dyn1:%.+]] = quantum.insert [[r_dyn]][ 1], [[second_call]]#2 : !quantum.reg, !quantum.bit


    // CHECK: [[q20:%.+]] = quantum.extract [[insert_21]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[obs_q:%.+]] = quantum.compbasis qubits [[q20]] : !quantum.obs
    %q20 = qref.get %r2[0] : !qref.reg<2> -> !qref.bit
    %obs_q = qref.compbasis qubits %q20 : !quantum.obs
    // CHECK: [[insert_r2:%.+]] = quantum.insert [[insert_21]][ 0], [[q20]] : !quantum.reg, !quantum.bit

    // CHECK: [[obs_r:%.+]] = quantum.compbasis qreg [[insert_dyn1]] : !quantum.obs
    %obs_r = qref.compbasis (qreg %r_dyn : !qref.reg<?>) : !quantum.obs

    // CHECK: quantum.dealloc [[insert_r2]] : !quantum.reg
    // CHECK: quantum.dealloc [[insert_dyn1]] : !quantum.reg
    qref.dealloc %r2 : !qref.reg<2>
    qref.dealloc %r_dyn : !qref.reg<?>

    // CHECK: return [[obs_q]], [[obs_r]] : !quantum.obs, !quantum.obs
    return %obs_q, %obs_r : !quantum.obs, !quantum.obs
}


// -----


// CHECK: func.func @test_no_extract_before_call(%arg0: f64, %arg1: i64, %arg2: !quantum.reg) -> !quantum.reg {
// CHECK:     [[extract:%.+]] = quantum.extract %arg2[%arg1] : !quantum.reg -> !quantum.bit
// CHECK:     [[RX:%.+]] = quantum.custom "RX"(%arg0) [[extract]] : !quantum.bit
// CHECK:     [[insert:%.+]] = quantum.insert %arg2[%arg1], [[RX]] : !quantum.reg, !quantum.bit
// CHECK:     return [[insert]] : !quantum.reg
// CHECK: }

func.func @test_no_extract_before_call(%reg: !qref.reg<2>, %param: f64, %idx: i64) -> () {
    %q = qref.get %reg[%idx] : !qref.reg<2>, i64 -> !qref.bit
    qref.custom "RX"(%param) %q : !qref.bit
    return
}

// CHECK: func.func @main(%arg0: f64, %arg1: i64) attributes {quantum.node}
func.func @main(%arg0: f64, %arg1: i64) -> () attributes {quantum.node} {
    // CHECK: [[r2_0:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r2_1:%.+]] = quantum.alloc( 2) : !quantum.reg
    %r2_0 = qref.alloc(2) : !qref.reg<2>
    %r2_1 = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[first_call:%.+]] = call @test_no_extract_before_call(%arg0, %arg1, [[r2_0]]) : (f64, i64, !quantum.reg) -> !quantum.reg
    // CHECK: [[second_call:%.+]] = call @test_no_extract_before_call(%arg0, %arg1, [[r2_1]]) : (f64, i64, !quantum.reg) -> !quantum.reg
    func.call @test_no_extract_before_call(%r2_0, %arg0, %arg1) : (!qref.reg<2>, f64, i64) -> ()
    func.call @test_no_extract_before_call(%r2_1, %arg0, %arg1) : (!qref.reg<2>, f64, i64) -> ()

    // CHECK: quantum.dealloc [[first_call]] : !quantum.reg
    // CHECK: quantum.dealloc [[second_call]] : !quantum.reg
    qref.dealloc %r2_0 : !qref.reg<2>
    qref.dealloc %r2_1 : !qref.reg<2>

    return
}


// -----


// CHECK: func.func @test_single_qubit_alloc(%arg0: f64, %arg1: !quantum.bit) -> !quantum.bit {
// CHECK:   %out_qubits = quantum.custom "RX"(%arg0) %arg1 : !quantum.bit
// CHECK:   return %out_qubits : !quantum.bit
// CHECK: }


func.func @test_single_qubit_alloc(%q: !qref.bit, %param: f64) -> () {
    qref.custom "RX"(%param) %q : !qref.bit
    return
}

// CHECK: func.func @main(%arg0: f64) attributes {quantum.node}
func.func @main(%arg0: f64) -> () attributes {quantum.node} {
    // CHECK: [[q:%.+]] = quantum.alloc_qb : !quantum.bit
    %q = qref.alloc_qb : !qref.bit

    // CHECK: [[first_call:%.+]] = call @test_single_qubit_alloc(%arg0, [[q]]) : (f64, !quantum.bit) -> !quantum.bit
    // CHECK: [[second_call:%.+]] = call @test_single_qubit_alloc(%arg0, [[first_call]]) : (f64, !quantum.bit) -> !quantum.bit
    func.call @test_single_qubit_alloc(%q, %arg0) : (!qref.bit, f64) -> ()
    func.call @test_single_qubit_alloc(%q, %arg0) : (!qref.bit, f64) -> ()

    // CHECK: quantum.dealloc_qb [[second_call]] : !quantum.bit
    qref.dealloc_qb %q : !qref.bit

    return
}
