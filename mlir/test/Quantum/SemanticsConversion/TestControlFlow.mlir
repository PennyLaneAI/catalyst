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
