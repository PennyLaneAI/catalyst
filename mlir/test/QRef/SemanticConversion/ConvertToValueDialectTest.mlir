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
