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


//
// for loop
//


// CHECK-LABEL: test_basic_for_loop
func.func @test_basic_for_loop() attributes {quantum.node} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c37 = arith.constant 37 : index

    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
    // CHECK: [[qb:%.+]] = qref.alloc_qb : !qref.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.alloc_qb : !quantum.bit

    // CHECK: scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} {
    // CHECK:   qref.custom "gate"() [[q0]], [[q1]], [[qb]] : !qref.bit, !qref.bit, !qref.bit
    // CHECK:   qref.custom "gate"() [[q0]], [[q1]], [[qb]] : !qref.bit, !qref.bit, !qref.bit
    // CHECK-NOT: scf.yield
    // CHECK: }
    %4:3 = scf.for %arg0 = %c0 to %c37 step %c1 iter_args(%arg1 = %1, %arg2 = %2, %arg3 = %3) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
        %5:3 = quantum.custom "gate"() %arg1, %arg2, %arg3 : !quantum.bit, !quantum.bit, !quantum.bit
        %6:3 = quantum.custom "gate"() %5#0, %5#1, %5#2 : !quantum.bit, !quantum.bit, !quantum.bit
        scf.yield %6#0, %6#1, %6#2 : !quantum.bit, !quantum.bit, !quantum.bit
    }

    // CHECK-NOT: quantum.insert
    %7 = quantum.insert %0[ 0], %4#0 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %4#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc_qb [[qb]] : !qref.bit
    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc_qb %4#2 : !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_for_loop_dynamic_index
func.func @test_for_loop_dynamic_index() attributes {quantum.node} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c37 = arith.constant 37 : index

    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    %1 = quantum.alloc( 3) : !quantum.reg

    // CHECK: scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} {
    %2 = scf.for %arg0 = %c0 to %c37 step %c1 iter_args(%arg1 = %1) -> (!quantum.reg) {
        // CHECK: [[i:%.+]] = index.casts %arg0 : index to i64
        // CHECK: [[qi:%.+]] = qref.get [[qreg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
        %3 = index.casts %arg0 : index to i64
        %4 = quantum.extract %arg1[%3] : !quantum.reg -> !quantum.bit

        // CHECK: qref.custom "Hadamard"() [[qi]] : !qref.bit
        %out_qubits = quantum.custom "Hadamard"() %4 : !quantum.bit

        // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
        %5 = quantum.extract %arg1[ 0] : !quantum.reg -> !quantum.bit

        // CHECK: qref.custom "CNOT"() [[qi]], [[q0]] : !qref.bit, !qref.bit
        %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits, %5 : !quantum.bit, !quantum.bit

        // CHECK-NOT: quantum.insert
        // CHECK-NOT: scf.yield
        %6 = quantum.insert %arg1[%3], %out_qubits_0#0 : !quantum.reg, !quantum.bit
        %7 = quantum.insert %6[ 0], %out_qubits_0#1 : !quantum.reg, !quantum.bit
        scf.yield %7 : !quantum.reg
    }

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc %2 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_for_loop_with_existing_args
func.func @test_for_loop_with_existing_args() -> (f64, f32) attributes {quantum.node} {
    // CHECK: [[cst:%.+]] = arith.constant 3.742000e+01 : f32
    // CHECK: [[cst_0:%.+]] = arith.constant 0.000000e+00 : f32
    %cst = arith.constant 3.742000e+01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32

    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c37 = arith.constant 37 : index

    // CHECK: %0 = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[loopOut:%.+]] = scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg1 = [[cst_0]]) -> (f32) {
    %4:3 = scf.for %arg1 = %c0 to %c37 step %c1 iter_args(%arg2 = %cst_0, %arg3 = %1, %arg4 = %2) -> (f32, !quantum.bit, !quantum.bit) {
        // CHECK: qref.custom "Hadamard"() [[q0]] : !qref.bit
        // CHECK: qref.custom "PauliZ"() [[q1]] : !qref.bit
        // CHECK: qref.custom "CNOT"() [[q1]], [[q0]] : !qref.bit, !qref.bit
        %out_qubits_1 = quantum.custom "Hadamard"() %arg3 : !quantum.bit
        %out_qubits_2 = quantum.custom "PauliZ"() %arg4 : !quantum.bit
        %out_qubits_3:2 = quantum.custom "CNOT"() %out_qubits_2, %out_qubits_1 : !quantum.bit, !quantum.bit

        // CHECK: [[add:%.+]] = arith.addf %arg1, [[cst]] : f32
        // CHECK: scf.yield [[add]] : f32
        %9 = arith.addf %arg2, %cst : f32
        scf.yield %9, %out_qubits_3#1, %out_qubits_3#0 : f32, !quantum.bit, !quantum.bit
    }

    // CHECK-NOT: quantum.insert
    %5 = quantum.insert %0[ 0], %4#1 : !quantum.reg, !quantum.bit

    // CHECK: [[obs:%.+]] = qref.namedobs [[q1]][ PauliX] : !quantum.obs
    %6 = quantum.namedobs %4#2[ PauliX] : !quantum.obs
    %7 = quantum.insert %5[ 1], %4#2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %7 : !quantum.reg

    // CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    // CHECK: return [[expval]], [[loopOut]] : f64, f32
    %8 = quantum.expval %6 : f64
    return %8, %4#0 : f64, f32
}


// -----


// CHECK-LABEL: test_for_loop_nested
func.func @test_for_loop_nested() -> f64 attributes {quantum.node} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c37 = arith.constant 37 : index

    // CHECK: [[reg1:%.+]] = qref.alloc( 1) : !qref.reg<1>
    // CHECK: [[q10:%.+]] = qref.get [[reg1]][ 0] : !qref.reg<1> -> !qref.bit
    // CHECK: qref.custom "PauliX"() [[q10]] : !qref.bit
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit

    // CHECK: scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} {
    %2 = scf.for %arg0 = %c0 to %c37 step %c1 iter_args(%arg1 = %out_qubits) -> (!quantum.bit) {

        // CHECK: [[reg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
        // CHECK: [[q20:%.+]] = qref.get [[reg2]][ 0] : !qref.reg<2> -> !qref.bit
        // CHECK: qref.custom "CNOT"() [[q10]], [[q20]] : !qref.bit, !qref.bit
        %6 = quantum.alloc( 2) : !quantum.reg
        %7 = quantum.extract %6[ 0] : !quantum.reg -> !quantum.bit
        %out_qubits_0:2 = quantum.custom "CNOT"() %arg1, %7 : !quantum.bit, !quantum.bit

        // CHECK: scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} {
        %8:2 = scf.for %arg2 = %c0 to %c37 step %c1 iter_args(%arg3 = %out_qubits_0#0, %arg4 = %out_qubits_0#1) -> (!quantum.bit, !quantum.bit) {

            // CHECK: [[reg3:%.+]] = qref.alloc( 3) : !qref.reg<3>
            // CHECK: [[q30:%.+]] = qref.get [[reg3]][ 0] : !qref.reg<3> -> !qref.bit
            // CHECK: qref.custom "Toffoli"() [[q10]], [[q20]], [[q30]] : !qref.bit, !qref.bit, !qref.bit
            %10 = quantum.alloc( 3) : !quantum.reg
            %11 = quantum.extract %10[ 0] : !quantum.reg -> !quantum.bit
            %out_qubits_1:3 = quantum.custom "Toffoli"() %arg3, %arg4, %11 : !quantum.bit, !quantum.bit, !quantum.bit
            %12 = quantum.insert %10[ 0], %out_qubits_1#2 : !quantum.reg, !quantum.bit

            // CHECK: qref.dealloc [[reg3]] : !qref.reg<3>
            // CHECK-NOT: scf.yield
            quantum.dealloc %12 : !quantum.reg
            scf.yield %out_qubits_1#0, %out_qubits_1#1 : !quantum.bit, !quantum.bit
        }
        %9 = quantum.insert %6[ 0], %8#1 : !quantum.reg, !quantum.bit

        // CHECK: qref.dealloc [[reg2]] : !qref.reg<2>
        // CHECK-NOT: scf.yield
        quantum.dealloc %9 : !quantum.reg
        scf.yield %8#0 : !quantum.bit
    }

    // CHECK: [[obs:%.+]] = qref.namedobs [[q10]][ PauliX] : !quantum.obs
    // CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    %3 = quantum.namedobs %2[ PauliX] : !quantum.obs
    %4 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit
    %5 = quantum.expval %3 : f64

    // CHECK: qref.dealloc [[reg1]] : !qref.reg<1>
    // CHECK: return [[expval]] : f64
    quantum.dealloc %4 : !quantum.reg
    return %5 : f64
}


// -----


// CHECK-LABEL: test_for_loop_with_dynamic_allocation
func.func public @test_for_loop_with_dynamic_allocation() attributes {quantum.node} {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index

    // CHECK: [[reg3:%.+]] = qref.alloc( 3) : !qref.reg<3>
    %0 = quantum.alloc( 3) : !quantum.reg

    // CHECK: scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} {
    %1 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %0) -> (!quantum.reg) {
        // CHECK: [[i:%.+]] = arith.index_cast %arg0 : index to i64
        %10 = arith.index_cast %arg0 : index to i64

        // CHECK: [[reg2:%.+]] = qref.alloc( 2) : !qref.reg<2>
        // CHECK: [[q20:%.+]] = qref.get [[reg2]][ 0] : !qref.reg<2> -> !qref.bit
        %11 = quantum.alloc( 2) : !quantum.reg
        %12 = quantum.extract %11[ 0] : !quantum.reg -> !quantum.bit

        // CHECK: [[q3i:%.+]] = qref.get [[reg3]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
        %13 = quantum.extract %arg1[%10] : !quantum.reg -> !quantum.bit

        // CHECK: qref.custom "CNOT"() [[q20]], [[q3i]] : !qref.bit, !qref.bit
        %out_qubits:2 = quantum.custom "CNOT"() %12, %13 : !quantum.bit, !quantum.bit

        // CHECK-NOT: quantum.insert
        %14 = quantum.insert %11[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
        %15 = quantum.insert %arg1[%10], %out_qubits#1 : !quantum.reg, !quantum.bit

        // CHECK: qref.dealloc [[reg2]] : !qref.reg<2>
        quantum.dealloc %14 : !quantum.reg

        // CHECK-NOT: scf.yield
        scf.yield %15 : !quantum.reg
    }

    // CHECK: [[reg1:%.+]] = qref.alloc( 1) : !qref.reg<1>
    // CHECK: [[q10:%.+]] = qref.get [[reg1]][ 0] : !qref.reg<1> -> !qref.bit
    %2 = quantum.alloc( 1) : !quantum.reg
    %3 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} {
    %4:2 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %3, %arg2 = %1) -> (!quantum.bit, !quantum.reg) {
        // CHECK: [[i:%.+]] = arith.index_cast %arg0 : index to i64
        %10 = arith.index_cast %arg0 : index to i64

        // CHECK: [[q3i:%.+]] = qref.get [[reg3]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
        // CHECK: qref.custom "CNOT"() [[q10]], [[q3i]] : !qref.bit, !qref.bit
        %11 = quantum.extract %arg2[%10] : !quantum.reg -> !quantum.bit
        %out_qubits:2 = quantum.custom "CNOT"() %arg1, %11 : !quantum.bit, !quantum.bit

        // CHECK-NOT: quantum.insert
        // CHECK-NOT: scf.yield
        %12 = quantum.insert %arg2[%10], %out_qubits#1 : !quantum.reg, !quantum.bit
        scf.yield %out_qubits#0, %12 : !quantum.bit, !quantum.reg
    }
    %5 = quantum.insert %2[ 0], %4#0 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[reg1]] : !qref.reg<1>
    // CHECK: qref.dealloc [[reg3]] : !qref.reg<3>
    quantum.dealloc %5 : !quantum.reg
    quantum.dealloc %4#1 : !quantum.reg
    return
}


// -----


//
// while loop
//


// CHECK-LABEL: test_basic_while_loop
func.func @test_basic_while_loop(%arg0: f64) attributes {quantum.node} {

    // CHECK: [[q:%.+]] = qref.alloc_qb : !qref.bit
    %0 = quantum.alloc_qb : !quantum.bit

    // CHECK: scf.while : () -> () {
    %1 = scf.while (%arg1 = %0) : (!quantum.bit) -> !quantum.bit {

        // CHECK: [[mres:%.+]] = qref.measure [[q]] : i1
        // CHECK: scf.condition([[mres]])
        %mres, %out_qubit = quantum.measure %arg1 : i1, !quantum.bit
        scf.condition(%mres) %out_qubit : !quantum.bit

    // CHECK: } do {
    // CHECK-NOT: ^bb
    } do {
    ^bb0(%arg1: !quantum.bit):

        // CHECK: qref.custom "gate"(%arg0) [[q]] : !qref.bit
        // CHECK: scf.yield
        %out_qubits = quantum.custom "gate"(%arg0) %arg1 : !quantum.bit
        scf.yield %out_qubits : !quantum.bit
    }

    // CHECK: qref.dealloc_qb [[q]] : !qref.bit
    quantum.dealloc_qb %1 : !quantum.bit
    return
}


// -----


// CHECK-LABEL: test_while_loop_with_existing_args
func.func @test_while_loop_with_existing_args(%arg0: i1, %arg1: f64) -> (f64, f64) attributes {quantum.node} {
    // CHECK: [[cst:%.+]] = arith.constant 1.000000e-01 : f64
    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q2:%.+]] = qref.get [[qreg]][ 2] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
    %cst = arith.constant 1.000000e-01 : f64
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[loopOut:%.+]]:2 = scf.while (%arg2 = %arg1) : (f64) -> (f64, f64) {
    %4:5 = scf.while (%arg2 = %arg1, %arg3 = %1, %arg4 = %2, %arg5 = %3) : (f64, !quantum.bit, !quantum.bit, !quantum.bit) -> (f64, f64, !quantum.bit, !quantum.bit, !quantum.bit) {

        // CHECK: [[mres:%.+]] = qref.measure [[q2]] : i1
        // CHECK: [[neg:%.+]] = arith.negf %arg2 : f64
        // CHECK: scf.condition([[mres]]) %arg2, [[neg]] : f64, f64
        %mres, %out_qubit = quantum.measure %arg3 : i1, !quantum.bit
        %8 = arith.negf %arg2 : f64
        scf.condition(%mres) %arg2, %8, %out_qubit, %arg4, %arg5 : f64, f64, !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: } do {
    // CHECK: ^bb0(%arg2: f64, %arg3: f64):
    } do {
    ^bb0(%arg2: f64, %arg3: f64, %arg4: !quantum.bit, %arg5: !quantum.bit, %arg6: !quantum.bit):

        // CHECK: qref.custom "gate"(%arg2, %arg3) [[q0]], [[q1]], [[q2]] : !qref.bit, !qref.bit, !qref.bit
        // CHECK: [[add:%.+]] = arith.addf %arg2, [[cst]] : f64
        // CHECK: scf.yield [[add]] : f64
        %out_qubits:3 = quantum.custom "gate"(%arg2, %arg3) %arg5, %arg6, %arg4 : !quantum.bit, !quantum.bit, !quantum.bit
        %8 = arith.addf %arg2, %cst : f64
        scf.yield %8, %out_qubits#2, %out_qubits#0, %out_qubits#1 : f64, !quantum.bit, !quantum.bit, !quantum.bit
    }

    // CHECK-NOT: quantum.insert
    %5 = quantum.insert %0[ 2], %4#2 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 0], %4#3 : !quantum.reg, !quantum.bit
    %7 = quantum.insert %6[ 1], %4#4 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    // CHECK: return [[loopOut]]#0, [[loopOut]]#1 : f64, f64
    quantum.dealloc %7 : !quantum.reg
    return %4#0, %4#1 : f64, f64
}


// -----


// CHECK-LABEL: test_while_dynamic_index
func.func @test_while_dynamic_index(%arg0: i1, %arg1: i64) -> i64 attributes {quantum.node} {

    // CHECK: [[one:%.+]] = arith.constant 1 : i64
    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    %c1_i64 = arith.constant 1 : i64
    %0 = quantum.alloc( 3) : !quantum.reg

    // CHECK: [[loopOut:%.+]] = scf.while (%arg2 = %arg1) : (i64) -> i64 {
    %1:2 = scf.while (%arg2 = %arg1, %arg3 = %0) : (i64, !quantum.reg) -> (i64, !quantum.reg) {

        // CHECK: scf.condition(%arg0) %arg2 : i64
        scf.condition(%arg0) %arg2, %arg3 : i64, !quantum.reg

    // CHECK: } do {
    // CHECK: ^bb0(%arg2: i64):
    } do {
    ^bb0(%arg2: i64, %arg3: !quantum.reg):

        // CHECK: [[qi:%.+]] = qref.get [[qreg]][%arg2] : !qref.reg<3>, i64 -> !qref.bit
        // CHECK: qref.custom "gate"() [[qi]] : !qref.bit
        // CHECK: [[add:%.+]] = arith.addi %arg2, [[one]] : i64
        // CHECK: scf.yield [[add]] : i64
        %2 = quantum.extract %arg3[%arg2] : !quantum.reg -> !quantum.bit
        %out_qubits = quantum.custom "gate"() %2 : !quantum.bit
        %3 = quantum.insert %arg3[%arg2], %out_qubits : !quantum.reg, !quantum.bit
        %4 = arith.addi %arg2, %c1_i64 : i64
        scf.yield %4, %3 : i64, !quantum.reg
    }

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    // CHECK: return [[loopOut]] : i64
    quantum.dealloc %1#1 : !quantum.reg
    return %1#0 : i64
}


// -----


// CHECK-LABEL: test_while_loop_nested
func.func @test_while_loop_nested(%arg0: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: scf.while : () -> () {
    // CHECK:     scf.condition(%arg0)
    // CHECK: } do {
    // CHECK-NOT: bb0
    %2 = scf.while (%arg1 = %1) : (!quantum.bit) -> !quantum.bit {
        scf.condition(%arg0) %arg1 : !quantum.bit
    } do {
    ^bb0(%arg1: !quantum.bit):

        // CHECK: scf.while : () -> () {
        // CHECK:     scf.condition(%arg0)
        // CHECK: } do {
        // CHECK-NOT: bb0
        %6 = scf.while (%arg2 = %arg1) : (!quantum.bit) -> !quantum.bit {
            scf.condition(%arg0) %arg2 : !quantum.bit
        } do {
        ^bb0(%arg2: !quantum.bit):

            // CHECK: qref.custom "X"() [[q0]] : !qref.bit
            // CHECK: scf.yield
            %out_qubits = quantum.custom "X"() %arg2 : !quantum.bit
            scf.yield %out_qubits : !quantum.bit
        }

        // CHECK: scf.yield
        scf.yield %6 : !quantum.bit
    }
    %3 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc %3 : !quantum.reg
    return
}


// -----


//
// if
//


// CHECK-LABEL: test_if_non_root_no_else
func.func @test_if_non_root_no_else(%arg0: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit

    // CHECK: scf.if %arg0 {
    // CHECK:   qref.custom "Hadamard"() [[q0]] : !qref.bit
    // CHECK:   qref.custom "X"() [[q0]] : !qref.bit
    // CHECK: } else {
    // CHECK-NEXT: }
    %2 = scf.if %arg0 -> (!quantum.bit) {
        %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
        %out_qubits_0 = quantum.custom "X"() %out_qubits : !quantum.bit
        scf.yield %out_qubits_0 : !quantum.bit
    } else {
        scf.yield %1 : !quantum.bit
    }

    // CHECK-NOT: quantum.insert
    %3 = quantum.insert %0[ 0], %2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc %3 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_if_non_root_with_else
func.func @test_if_non_root_with_else(%arg0: i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: scf.if %arg0 {
    // CHECK:   qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
    // CHECK: } else {
    // CHECK:   qref.custom "Y"() [[q1]] : !qref.bit
    // CHECK: }
    %3:2 = scf.if %arg0 -> (!quantum.bit, !quantum.bit) {
        %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
        scf.yield %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
    } else {
        %out_qubits = quantum.custom "Y"() %2 : !quantum.bit
        scf.yield %1, %out_qubits : !quantum.bit, !quantum.bit
    }

    // CHECK-NOT: quantum.insert
    %4 = quantum.insert %0[ 0], %3#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %3#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc %5 : !quantum.reg
    return
}


// -----


// CHECK-LABEL: test_if_root_no_else
func.func @test_if_root_no_else(%arg0: i1) attributes {quantum.node} {

    // CHECK: [[qb:%.+]] = qref.alloc_qb : !qref.bit
    %0 = quantum.alloc_qb : !quantum.bit

    // CHECK: scf.if %arg0 {
    // CHECK:   qref.custom "Hadamard"() [[qb]] : !qref.bit
    // CHECK: } else {
    // CHECK-NEXT: }
    %1 = scf.if %arg0 -> (!quantum.bit) {
        %out_qubits = quantum.custom "Hadamard"() %0 : !quantum.bit
        scf.yield %out_qubits : !quantum.bit
    } else {
        scf.yield %0 : !quantum.bit
    }

    // CHECK: qref.dealloc_qb [[qb]] : !qref.bit
    quantum.dealloc_qb %1 : !quantum.bit
    return
}


// -----


// CHECK-LABEL: test_if_root_with_else
func.func @test_if_root_with_else(%arg0: i1, %arg1: f64) attributes {quantum.node} {

    // CHECK: [[qb:%.+]] = qref.alloc_qb : !qref.bit
    %0 = quantum.alloc_qb : !quantum.bit

    // CHECK: scf.if %arg0 {
    // CHECK:   qref.custom "Hadamard"() [[qb]] : !qref.bit
    // CHECK: } else {
    // CHECK:   qref.custom "RX"(%arg1) [[qb]] : !qref.bit
    // CHECK: }
    %1 = scf.if %arg0 -> (!quantum.bit) {
        %out_qubits = quantum.custom "Hadamard"() %0 : !quantum.bit
        scf.yield %out_qubits : !quantum.bit
    } else {
        %out_qubits = quantum.custom "RX"(%arg1) %0 : !quantum.bit
        scf.yield %out_qubits : !quantum.bit
    }

    // CHECK: qref.dealloc_qb [[qb]] : !qref.bit
    quantum.dealloc_qb %1 : !quantum.bit
    return
}


// -----


// CHECK-LABEL: test_if_with_existing_results
func.func @test_if_with_existing_results(%arg0: i1, %arg1: f64) -> i1 attributes {quantum.node} {
    // CHECK: [[qb:%.+]] = qref.alloc_qb : !qref.bit
    %0 = quantum.alloc_qb : !quantum.bit

    // CHECK: [[ifOut:%.+]] = scf.if %arg0 -> (i1) {
    // CHECK:   qref.custom "Hadamard"() [[qb]] : !qref.bit
    // CHECK:   [[mres_t:%.+]] = qref.measure [[qb]] : i1
    // CHECK:   scf.yield [[mres_t]] : i1
    // CHECK: } else {
    // CHECK:   qref.custom "RX"(%arg1) [[qb]] : !qref.bit
    // CHECK:   [[mres_f:%.+]] = qref.measure [[qb]] : i1
    // CHECK:   scf.yield [[mres_f]] : i1
    // CHECK: }
    %1:2 = scf.if %arg0 -> (i1, !quantum.bit) {
        %out_qubits = quantum.custom "Hadamard"() %0 : !quantum.bit
        %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
        scf.yield %mres, %out_qubit : i1, !quantum.bit
    } else {
        %out_qubits = quantum.custom "RX"(%arg1) %0 : !quantum.bit
        %mres, %out_qubit = quantum.measure %out_qubits : i1, !quantum.bit
        scf.yield %mres, %out_qubit : i1, !quantum.bit
    }

    // CHECK: qref.dealloc_qb [[qb]] : !qref.bit
    // CHECK: return [[ifOut]] : i1
    quantum.dealloc_qb %1#1 : !quantum.bit
    return %1#0 : i1
}


// -----


// CHECK-LABEL: test_if_nested_for
func.func @test_if_nested_for(%arg0: i1) attributes {quantum.node} {
    %c37 = arith.constant 37 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index

    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    %0 = quantum.alloc( 3) : !quantum.reg

    // CHECK: scf.if %arg0 {
    %1 = scf.if %arg0 -> (!quantum.reg) {

        // CHECK: scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} {
        %2 = scf.for %arg1 = %c0 to %c37 step %c1 iter_args(%arg2 = %0) -> (!quantum.reg) {
            // CHECK: [[i:%.+]] = index.casts %arg1 : index to i64
            // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
            // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
            // CHECK: [[qi:%.+]] = qref.get [[qreg]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
            // CHECK: qref.custom "Toffoli"() [[q0]], [[q1]], [[qi]] : !qref.bit, !qref.bit, !qref.bit
            %7 = index.casts %arg1 : index to i64
            %8 = quantum.extract %arg2[ 0] : !quantum.reg -> !quantum.bit
            %9 = quantum.extract %arg2[ 1] : !quantum.reg -> !quantum.bit
            %10 = quantum.extract %arg2[%7] : !quantum.reg -> !quantum.bit
            %out_qubits_0:3 = quantum.custom "Toffoli"() %8, %9, %10 : !quantum.bit, !quantum.bit, !quantum.bit

            // CHECK-NOT: quantum.insert
            %11 = quantum.insert %arg2[ 0], %out_qubits_0#0 : !quantum.reg, !quantum.bit
            %12 = quantum.insert %11[ 1], %out_qubits_0#1 : !quantum.reg, !quantum.bit
            %13 = quantum.insert %12[%7], %out_qubits_0#2 : !quantum.reg, !quantum.bit

            // CHECK-NOT: scf.yield
            scf.yield %13 : !quantum.reg
        }

        // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
        // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
        // CHECK: qref.custom "CNOT"() [[q0]], [[q1]] : !qref.bit, !qref.bit
        %3 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
        %4 = quantum.extract %2[ 1] : !quantum.reg -> !quantum.bit
        %out_qubits:2 = quantum.custom "CNOT"() %3, %4 : !quantum.bit, !quantum.bit

        // CHECK-NOT: quantum.insert
        %5 = quantum.insert %2[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
        %6 = quantum.insert %5[ 1], %out_qubits#1 : !quantum.reg, !quantum.bit

        // CHECK-NOT: scf.yield
        scf.yield %6 : !quantum.reg

    // CHECK: } else {
    // CHECK-NEXT: }
    } else {
        scf.yield %0 : !quantum.reg
    }

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc %1 : !quantum.reg
    return
}


// -----


//
// switch
//


// CHECK-LABEL: test_switch
func.func @test_switch(%arg0: index, %arg1: f64) -> f64 attributes {quantum.node} {
    // CHECK: [[qb:%.+]] = qref.alloc_qb : !qref.bit
    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
    %0 = quantum.alloc_qb : !quantum.bit
    %1 = quantum.alloc( 3) : !quantum.reg
    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %1[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[switchOut:%.+]] = scf.index_switch %arg0 -> f64
    %4:4 = scf.index_switch %arg0 -> !quantum.bit, !quantum.bit, !quantum.bit, f64
    // CHECK: case 10 {
    // CHECK:   qref.custom "Hadamard"() [[qb]] : !qref.bit
    // CHECK:   qref.custom "Toffoli"() [[qb]], [[q0]], [[q1]] : !qref.bit, !qref.bit, !qref.bit
    // CHECK:   scf.yield %arg1 : f64
    // CHECK: }
    case 10 {
        %out_qubits = quantum.custom "Hadamard"() %0 : !quantum.bit
        %out_qubits_0:3 = quantum.custom "Toffoli"() %out_qubits, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
        scf.yield %out_qubits_0#0, %out_qubits_0#1, %out_qubits_0#2, %arg1 : !quantum.bit, !quantum.bit, !quantum.bit, f64
    }
    // CHECK: case 20 {
    // CHECK:   qref.custom "RX"(%arg1) [[qb]] : !qref.bit
    // CHECK:   qref.custom "CNOT"() [[qb]], [[q1]] : !qref.bit, !qref.bit
    // CHECK:   scf.yield %arg1 : f64
    // CHECK: }
    case 20 {
        %out_qubits = quantum.custom "RX"(%arg1) %0 : !quantum.bit
        %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits, %3 : !quantum.bit, !quantum.bit
        scf.yield %out_qubits_0#0, %2, %out_qubits_0#1, %arg1 : !quantum.bit, !quantum.bit, !quantum.bit, f64
    }

    // CHECK: default {
    // CHECK:   scf.yield %arg1 : f64
    // CHECK: }
    default {
        scf.yield %0, %2, %3, %arg1 : !quantum.bit, !quantum.bit, !quantum.bit, f64
    }

    // CHECK-NOT: quantum.insert
    %5 = quantum.insert %1[ 0], %4#1 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 1], %4#2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc_qb [[qb]] : !qref.bit
    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    // CHECK: return [[switchOut]] : f64
    quantum.dealloc_qb %4#0 : !quantum.bit
    quantum.dealloc %6 : !quantum.reg
    return %4#3 : f64
}
