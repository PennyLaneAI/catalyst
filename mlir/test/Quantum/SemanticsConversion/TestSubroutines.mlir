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

// Test conversion to reference semantics quantum dialect where control flow is present.

// RUN: quantum-opt --convert-to-reference-semantics --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK: func.func @test_qreg_arg(%arg0: f64, %arg1: i64, %arg2: !qref.reg<2>) {
// CHECK:     [[q:%.+]] = qref.get %arg2[%arg1] : !qref.reg<2>, i64 -> !qref.bit
// CHECK:     qref.custom "RX"(%arg0) [[q]] : !qref.bit
// CHECK:     return
// CHECK: }

func.func @test_qreg_arg(%arg0: f64, %arg1: i64, %arg2: !quantum.reg) -> !quantum.reg {
    %0 = quantum.extract %arg2[%arg1] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "RX"(%arg0) %0 : !quantum.bit
    %1 = quantum.insert %arg2[%arg1], %out_qubits : !quantum.reg, !quantum.bit
    return %1 : !quantum.reg
}

// CHECK: func.func @main(%arg0: f64, %arg1: i64) attributes {quantum.node} {
// CHECK:     [[reg1:%.+]] = qref.alloc( 2) : !qref.reg<2>
// CHECK:     [[reg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
// CHECK:     call @test_qreg_arg(%arg0, %arg1, [[reg1]]) : (f64, i64, !qref.reg<2>) -> ()
// CHECK:     call @test_qreg_arg(%arg0, %arg1, [[reg2]]) : (f64, i64, !qref.reg<2>) -> ()
// CHECK:     qref.dealloc [[reg1]] : !qref.reg<2>
// CHECK:     qref.dealloc [[reg2]] : !qref.reg<2>
// CHECK:     return
// CHECK: }

func.func @main(%arg0: f64, %arg1: i64) attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.alloc( 2) : !quantum.reg
    %2 = call @test_qreg_arg(%arg0, %arg1, %0) : (f64, i64, !quantum.reg) -> !quantum.reg
    %3 = call @test_qreg_arg(%arg0, %arg1, %1) : (f64, i64, !quantum.reg) -> !quantum.reg
    quantum.dealloc %2 : !quantum.reg
    quantum.dealloc %3 : !quantum.reg
    return
}


// -----


// CHECK: func.func @test_multi_qreg_arg(%arg0: !qref.reg<2>, %arg1: i64, %arg2: !qref.reg<?>) {
// CHECK:     [[q0:%.+]] = qref.get %arg0[%arg1] : !qref.reg<2>, i64 -> !qref.bit
// CHECK:     [[q2:%.+]] = qref.get %arg2[%arg1] : !qref.reg<?>, i64 -> !qref.bit
// CHECK:     qref.custom "gate"() [[q0]], [[q2]] : !qref.bit, !qref.bit
// CHECK:     return
// CHECK: }

func.func @test_multi_qreg_arg(%arg0: !quantum.reg, %arg1: i64, %arg2: !quantum.reg) -> (!quantum.reg, !quantum.reg) {
    %0 = quantum.extract %arg0[%arg1] : !quantum.reg -> !quantum.bit
    %1 = quantum.extract %arg2[%arg1] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.custom "gate"() %0, %1 : !quantum.bit, !quantum.bit
    %2 = quantum.insert %arg0[%arg1], %out_qubits#0 : !quantum.reg, !quantum.bit
    %3 = quantum.insert %arg2[%arg1], %out_qubits#1 : !quantum.reg, !quantum.bit
    return %2, %3 : !quantum.reg, !quantum.reg
}

// CHECK: func.func @main(%arg0: i64) attributes {quantum.node} {
// CHECK:     [[reg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
// CHECK:     [[reg_dyn:%.+]] = qref.alloc(%arg0) : !qref.reg<?>
// CHECK:     call @test_multi_qreg_arg([[reg2]], %arg0, [[reg_dyn]]) : (!qref.reg<2>, i64, !qref.reg<?>) -> ()
// CHECK:     qref.dealloc [[reg2]] : !qref.reg<2>
// CHECK:     qref.dealloc [[reg_dyn]] : !qref.reg<?>
// CHECK:     return
// CHECK: }

func.func @main(%arg0: i64) attributes {quantum.node} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.alloc(%arg0) : !quantum.reg
    %2:2 = call @test_multi_qreg_arg(%0, %arg0, %1) : (!quantum.reg, i64, !quantum.reg) -> (!quantum.reg, !quantum.reg)
    quantum.dealloc %2#0 : !quantum.reg
    quantum.dealloc %2#1 : !quantum.reg
    return
}


// -----


// CHECK: func.func @test_qubit_args(%arg0: f64, %arg1: !qref.bit, %arg2: !qref.bit, %arg3: !qref.bit) {
// CHECK:     qref.custom "CNOT"() %arg1, %arg2 : !qref.bit, !qref.bit
// CHECK:     qref.custom "RX"(%arg0) %arg3 : !qref.bit
// CHECK:     qref.custom "Toffoli"() %arg1, %arg2, %arg3 : !qref.bit, !qref.bit, !qref.bit
// CHECK:     return
// CHECK: }

func.func @test_qubit_args(%arg0: f64, %arg1: !quantum.bit, %arg2: !quantum.bit, %arg3: !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
    %out_qubits:2 = quantum.custom "CNOT"() %arg1, %arg2 : !quantum.bit, !quantum.bit
    %out_qubits_0 = quantum.custom "RX"(%arg0) %arg3 : !quantum.bit
    %out_qubits_1:3 = quantum.custom "Toffoli"() %out_qubits#0, %out_qubits#1, %out_qubits_0 : !quantum.bit, !quantum.bit, !quantum.bit
    return %out_qubits_1#0, %out_qubits_1#1, %out_qubits_1#2 : !quantum.bit, !quantum.bit, !quantum.bit
}

func.func @main(%arg0: i64, %arg1: f64) -> (!quantum.obs, !quantum.obs) attributes {quantum.node} {
    // CHECK: [[reg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[reg_dyn:%.+]] = qref.alloc(%arg0) : !qref.reg<?>
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.alloc(%arg0) : !quantum.reg

    // CHECK: [[q20:%.+]] = qref.get [[reg2]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q21:%.+]] = qref.get [[reg2]][ 1] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q_dyn_1:%.+]] = qref.get [[reg_dyn]][ 1] : !qref.reg<?> -> !qref.bit
    // CHECK: call @test_qubit_args(%arg1, [[q20]], [[q21]], [[q_dyn_1]]) : (f64, !qref.bit, !qref.bit, !qref.bit) -> ()
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %4 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit
    %5:3 = call @test_qubit_args(%arg1, %2, %3, %4) : (f64, !quantum.bit, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit)
    %6 = quantum.insert %0[ 0], %5#0 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 1], %5#1 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %1[ 1], %5#2 : !quantum.reg, !quantum.bit

    // CHECK: [[q20:%.+]] = qref.get [[reg2]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q21:%.+]] = qref.get [[reg2]][ 1] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q_dyn_1:%.+]] = qref.get [[reg_dyn]][ 1] : !qref.reg<?> -> !qref.bit
    // CHECK: call @test_qubit_args(%arg1, [[q20]], [[q21]], [[q_dyn_1]]) : (f64, !qref.bit, !qref.bit, !qref.bit) -> ()
    %9 = quantum.extract %7[ 0] : !quantum.reg -> !quantum.bit
    %10 = quantum.extract %7[ 1] : !quantum.reg -> !quantum.bit
    %11 = quantum.extract %8[ 1] : !quantum.reg -> !quantum.bit
    %12:3 = call @test_qubit_args(%arg1, %9, %10, %11) : (f64, !quantum.bit, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit)
    %13 = quantum.insert %7[ 0], %12#0 : !quantum.reg, !quantum.bit
    %14 = quantum.insert %13[ 1], %12#1 : !quantum.reg, !quantum.bit
    %15 = quantum.insert %8[ 1], %12#2 : !quantum.reg, !quantum.bit

    // CHECK: [[q20:%.+]] = qref.get [[reg2]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[obs0:%.+]] = qref.compbasis qubits [[q20]] : !quantum.obs
    // CHECK: [[obs1:%.+]] = qref.compbasis(qreg [[reg_dyn]] : !qref.reg<?>) : !quantum.obs
    %16 = quantum.extract %14[ 0] : !quantum.reg -> !quantum.bit
    %17 = quantum.compbasis qubits %16 : !quantum.obs
    %18 = quantum.insert %14[ 0], %16 : !quantum.reg, !quantum.bit
    %19 = quantum.compbasis qreg %15 : !quantum.obs

    // CHECK: qref.dealloc [[reg2]] : !qref.reg<2>
    // CHECK: qref.dealloc [[reg_dyn]] : !qref.reg<?>
    quantum.dealloc %18 : !quantum.reg
    quantum.dealloc %15 : !quantum.reg

    // return [[obs0]], [[obs1]] : !quantum.obs, !quantum.obs
    return %17, %19 : !quantum.obs, !quantum.obs
}


// -----


// CHECK: func.func @test_single_qubit_alloc(%arg0: f64, %arg1: !qref.bit) {
// CHECK:     qref.custom "RX"(%arg0) %arg1 : !qref.bit
// CHECK:     return
// CHECK: }

func.func @test_single_qubit_alloc(%arg0: f64, %arg1: !quantum.bit) -> !quantum.bit {
    %out_qubits = quantum.custom "RX"(%arg0) %arg1 : !quantum.bit
    return %out_qubits : !quantum.bit
}

// CHECK: func.func @main(%arg0: f64) attributes {quantum.node} {
// CHECK:     [[q0:%.+]] = qref.alloc_qb : !qref.bit
// CHECK:     call @test_single_qubit_alloc(%arg0, [[q0]]) : (f64, !qref.bit) -> ()
// CHECK:     call @test_single_qubit_alloc(%arg0, [[q0]]) : (f64, !qref.bit) -> ()
// CHECK:     qref.dealloc_qb [[q0]] : !qref.bit
// CHECK:     return
// CHECK: }

func.func @main(%arg0: f64) attributes {quantum.node} {
    %0 = quantum.alloc_qb : !quantum.bit
    %2 = call @test_single_qubit_alloc(%arg0, %0) : (f64, !quantum.bit) -> !quantum.bit
    %3 = call @test_single_qubit_alloc(%arg0, %2) : (f64, !quantum.bit) -> !quantum.bit
    quantum.dealloc_qb %3 : !quantum.bit
    return
}


// -----


// CHECK: func.func @test_mixed_args(%arg0: i64, %arg1: !qref.bit, %arg2: !qref.reg<3>, %arg3: !qref.bit) -> (i1, i1, i1) {
func.func @test_mixed_args(%arg0: i64, %arg1: !quantum.bit, %arg2: !quantum.reg, %arg3: !quantum.bit) -> (i1, i1, i1, !quantum.bit, !quantum.reg, !quantum.bit) {
    // CHECK: [[q_from_reg:%.+]] = qref.get %arg2[%arg0] : !qref.reg<3>, i64 -> !qref.bit
    // CHECK: qref.custom "gate"() %arg1, [[q_from_reg]], %arg3 : !qref.bit, !qref.bit, !qref.bit
    %0 = quantum.extract %arg2[%arg0] : !quantum.reg -> !quantum.bit
    %out_qubits:3 = quantum.custom "gate"() %arg1, %0, %arg3 : !quantum.bit, !quantum.bit, !quantum.bit
    %1 = quantum.insert %arg2[%arg0], %out_qubits#1 : !quantum.reg, !quantum.bit

    // CHECK: [[mres1:%.+]] = qref.measure %arg1 : i1
    %mres, %out_qubit = quantum.measure %out_qubits#0 : i1, !quantum.bit

    // CHECK: [[q_from_reg:%.+]] = qref.get %arg2[%arg0] : !qref.reg<3>, i64 -> !qref.bit
    // CHECK: [[mres2:%.+]] = qref.measure [[q_from_reg]] : i1
    %2 = quantum.extract %1[%arg0] : !quantum.reg -> !quantum.bit
    %mres_0, %out_qubit_1 = quantum.measure %2 : i1, !quantum.bit
    %3 = quantum.insert %1[%arg0], %out_qubit_1 : !quantum.reg, !quantum.bit

    // CHECK: [[mres3:%.+]] = qref.measure %arg3 : i1
    %mres_2, %out_qubit_3 = quantum.measure %out_qubits#2 : i1, !quantum.bit

    // CHECK: return [[mres1]], [[mres2]], [[mres3]] : i1, i1, i1
    return %mres, %mres_0, %mres_2, %out_qubit, %3, %out_qubit_3 : i1, i1, i1, !quantum.bit, !quantum.reg, !quantum.bit
}


func.func @main(%arg0: i64) -> (i1, i1, i1, i1, i1, i1) attributes {quantum.node} {
    // CHECK: [[q:%.+]] = qref.alloc_qb : !qref.bit
    // CHECK: [[reg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[reg3:%.+]] = qref.alloc( 3) : !qref.reg<3>
    %0 = quantum.alloc_qb : !quantum.bit
    %2 = quantum.alloc( 2) : !quantum.reg
    %3 = quantum.alloc( 3) : !quantum.reg

    // CHECK: [[q20:%.+]] = qref.get [[reg2]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[first_call:%.+]]:3 = call @test_mixed_args(%arg0, [[q20]], [[reg3]], [[q]]) : (i64, !qref.bit, !qref.reg<3>, !qref.bit) -> (i1, i1, i1)
    %4 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
    %5:6 = call @test_mixed_args(%arg0, %4, %3, %0) : (i64, !quantum.bit, !quantum.reg, !quantum.bit) -> (i1, i1, i1, !quantum.bit, !quantum.reg, !quantum.bit)
    %6 = quantum.insert %2[ 0], %5#3 : !quantum.reg, !quantum.bit

    // CHECK: [[q20:%.+]] = qref.get [[reg2]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[second_call:%.+]]:3 = call @test_mixed_args(%arg0, [[q20]], [[reg3]], [[q]]) : (i64, !qref.bit, !qref.reg<3>, !qref.bit) -> (i1, i1, i1)
    %7 = quantum.extract %6[ 0] : !quantum.reg -> !quantum.bit
    %8:6 = call @test_mixed_args(%arg0, %7, %5#4, %5#5) : (i64, !quantum.bit, !quantum.reg, !quantum.bit) -> (i1, i1, i1, !quantum.bit, !quantum.reg, !quantum.bit)
    %9 = quantum.insert %6[ 0], %8#3 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc_qb [[q]] : !qref.bit
    // CHECK: qref.dealloc [[reg2]] : !qref.reg<2>
    // CHECK: qref.dealloc [[reg3]] : !qref.reg<3>
    quantum.dealloc_qb %8#5 : !quantum.bit
    quantum.dealloc %9 : !quantum.reg
    quantum.dealloc %8#4 : !quantum.reg

    // CHECK: return [[first_call]]#0, [[first_call]]#1, [[first_call]]#2, [[second_call]]#0, [[second_call]]#1, [[second_call]]#2 : i1, i1, i1, i1, i1, i1
    return %5#0, %5#1, %5#2, %8#0, %8#1, %8#2 : i1, i1, i1, i1, i1, i1
}


// -----


// CHECK: func.func @test_loop_callsite(%arg0: !qref.bit, %arg1: !qref.bit) {
// CHECK:     qref.custom "CNOT"() %arg0, %arg1 : !qref.bit, !qref.bit
// CHECK:     return
// CHECK: }

func.func @test_loop_callsite(%arg0: !quantum.bit, %arg1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %out_qubits:2 = quantum.custom "CNOT"() %arg0, %arg1 : !quantum.bit, !quantum.bit
    return %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
}

func.func @main() attributes {quantum.node} {
    // CHECK: [[q:%.+]] = qref.alloc_qb : !qref.bit
    // CHECK: [[reg0:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[reg1:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc_qb : !quantum.bit
    %2 = quantum.alloc( 2) : !quantum.reg
    %3 = quantum.alloc( 2) : !quantum.reg

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c37 = arith.constant 37 : index

    // CHECK: [[q0:%.+]] = qref.get [[reg0]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[reg1]][ 0] : !qref.reg<2> -> !qref.bit
    %4 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} {
    // CHECK:     func.call @test_loop_callsite([[q]], [[q0]]) : (!qref.bit, !qref.bit) -> ()
    // CHECK:     func.call @test_loop_callsite([[q]], [[q1]]) : (!qref.bit, !qref.bit) -> ()
    // CHECK: }
    %6:3 = scf.for %arg0 = %c0 to %c37 step %c1 iter_args(%arg1 = %0, %arg2 = %4, %arg3 = %5) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
        %9:2 = func.call @test_loop_callsite(%arg1, %arg2) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %10:2 = func.call @test_loop_callsite(%9#0, %arg3) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        scf.yield %10#0, %9#1, %10#1 : !quantum.bit, !quantum.bit, !quantum.bit
    }
    %7 = quantum.insert %2[ 0], %6#1 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %3[ 0], %6#2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc_qb [[q]] : !qref.bit
    // CHECK: qref.dealloc [[reg0]] : !qref.reg<2>
    // CHECK: qref.dealloc [[reg1]] : !qref.reg<2>
    quantum.dealloc_qb %6#0 : !quantum.bit
    quantum.dealloc %7 : !quantum.reg
    quantum.dealloc %8 : !quantum.reg
    return
}


// -----


// CHECK: func.func @test_cond_callsite(%arg0: !qref.bit, %arg1: !qref.bit) {
// CHECK:     qref.custom "CNOT"() %arg0, %arg1 : !qref.bit, !qref.bit
// CHECK:     return
// CHECK: }

func.func @test_cond_callsite(%arg0: !quantum.bit, %arg1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %out_qubits:2 = quantum.custom "CNOT"() %arg0, %arg1 : !quantum.bit, !quantum.bit
    return %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
}

func.func @main(%arg0: i1) attributes {quantum.node} {
    // CHECK: [[q:%.+]] = qref.alloc_qb : !qref.bit
    // CHECK: [[reg0:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[reg1:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc_qb : !quantum.bit
    %2 = quantum.alloc( 2) : !quantum.reg
    %3 = quantum.alloc( 2) : !quantum.reg

    // CHECK: [[q0:%.+]] = qref.get [[reg0]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[reg1]][ 0] : !qref.reg<2> -> !qref.bit
    %4 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: scf.if %arg0 {
    // CHECK:   func.call @test_cond_callsite([[q]], [[q0]]) : (!qref.bit, !qref.bit) -> ()
    // CHECK:   func.call @test_cond_callsite([[q]], [[q1]]) : (!qref.bit, !qref.bit) -> ()
    // CHECK: } else {
    // CHECK: }
    %6:3 = scf.if %arg0 -> (!quantum.bit, !quantum.bit, !quantum.bit) {
        %9:2 = func.call @test_cond_callsite(%0, %4) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %10:2 = func.call @test_cond_callsite(%9#0, %5) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        scf.yield %10#0, %9#1, %10#1 : !quantum.bit, !quantum.bit, !quantum.bit
    } else {
        scf.yield %0, %4, %5 : !quantum.bit, !quantum.bit, !quantum.bit
    }
    %7 = quantum.insert %2[ 0], %6#1 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %3[ 0], %6#2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc_qb [[q]] : !qref.bit
    // CHECK: qref.dealloc [[reg0]] : !qref.reg<2>
    // CHECK: qref.dealloc [[reg1]] : !qref.reg<2>
    quantum.dealloc_qb %6#0 : !quantum.bit
    quantum.dealloc %7 : !quantum.reg
    quantum.dealloc %8 : !quantum.reg
    return
}


// -----

// CHECK: func.func @test_callsite_in_nested_region(%arg0: !qref.bit, %arg1: !qref.bit) {
// CHECK:     qref.custom "CNOT"() %arg0, %arg1 : !qref.bit, !qref.bit
// CHECK:     return
// CHECK: }
func.func @test_callsite_in_nested_region(%arg0: !quantum.bit, %arg1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    %out_qubits:2 = quantum.custom "CNOT"() %arg0, %arg1 : !quantum.bit, !quantum.bit
    return %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
}

func.func @main(%arg0: i1) attributes {quantum.node} {
    // CHECK: [[q:%.+]] = qref.alloc_qb : !qref.bit
    // CHECK: [[reg0:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[reg1:%.+]] = qref.alloc( 2) : !qref.reg<2>
    %0 = quantum.alloc_qb : !quantum.bit
    %2 = quantum.alloc( 2) : !quantum.reg
    %3 = quantum.alloc( 2) : !quantum.reg

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c37 = arith.constant 37 : index

    // CHECK: [[q0:%.+]] = qref.get [[reg0]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[reg1]][ 0] : !qref.reg<2> -> !qref.bit
    %4 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
    %5 = quantum.extract %3[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: scf.if %arg0 {
    // CHECK:   scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} {
    // CHECK:     func.call @test_callsite_in_nested_region([[q]], [[q0]]) : (!qref.bit, !qref.bit) -> ()
    // CHECK:     func.call @test_callsite_in_nested_region([[q]], [[q1]]) : (!qref.bit, !qref.bit) -> ()
    // CHECK:   }
    // CHECK: } else {
    // CHECK: }
    %6:3 = scf.if %arg0 -> (!quantum.bit, !quantum.bit, !quantum.bit) {
        %9:3 = scf.for %arg1 = %c0 to %c37 step %c1 iter_args(%arg2 = %0, %arg3 = %4, %arg4 = %5) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
        %10:2 = func.call @test_callsite_in_nested_region(%arg2, %arg3) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        %11:2 = func.call @test_callsite_in_nested_region(%10#0, %arg4) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        scf.yield %11#0, %10#1, %11#1 : !quantum.bit, !quantum.bit, !quantum.bit
        }
        scf.yield %9#0, %9#1, %9#2 : !quantum.bit, !quantum.bit, !quantum.bit
    } else {
        scf.yield %0, %4, %5 : !quantum.bit, !quantum.bit, !quantum.bit
    }
    %7 = quantum.insert %2[ 0], %6#1 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %3[ 0], %6#2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc_qb [[q]] : !qref.bit
    // CHECK: qref.dealloc [[reg0]] : !qref.reg<2>
    // CHECK: qref.dealloc [[reg1]] : !qref.reg<2>
    quantum.dealloc_qb %6#0 : !quantum.bit
    quantum.dealloc %7 : !quantum.reg
    quantum.dealloc %8 : !quantum.reg
    return
}


// -----

// CHECK: func.func @callee(%arg0: !qref.bit) {
// CHECK:     qref.custom "PauliX"() %arg0 : !qref.bit
// CHECK:     return
// CHECK: }
// CHECK: func.func @caller(%arg0: !qref.bit) {
// CHECK:     call @callee(%arg0) : (!qref.bit) -> ()
// CHECK:     return
// CHECK: }

func.func @callee(%arg0: !quantum.bit) -> !quantum.bit {
    %out_qubits = quantum.custom "PauliX"() %arg0 : !quantum.bit
    return %out_qubits : !quantum.bit
}
func.func @caller(%arg0: !quantum.bit) -> !quantum.bit {
    %0 = call @callee(%arg0) : (!quantum.bit) -> !quantum.bit
    return %0 : !quantum.bit
}

// CHECK: func.func @main() attributes {quantum.node} {
// CHECK:     [[reg:%.+]] = qref.alloc( 1) : !qref.reg<1>
// CHECK:     [[q:%.+]] = qref.get [[reg]][ 0] : !qref.reg<1> -> !qref.bit
// CHECK:     call @caller([[q]]) : (!qref.bit) -> ()
// CHECK:     qref.dealloc [[reg]] : !qref.reg<1>
// CHECK:     return
// CHECK: }
func.func @main() attributes {quantum.node} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = call @caller(%1) : (!quantum.bit) -> !quantum.bit
    %3 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
}
