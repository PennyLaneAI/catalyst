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

// Test conversion to value semantics quantum dialect where control flow is present.
// For each control flow construct, the following cases are tested:
// 1. Purely classical: no transform should happen
// 2. Root qref values, aka qref.bit/reg values that are a result of an allocation or an argument:
// the current value semantics value needs to be taken in by the control flow op, and new values
// returned.
// 3. Non-root qref values: a value semantics qubit value needs to be extracted, then taken in and
// returned by the control flow op.

// RUN: quantum-opt --convert-to-value-semantics --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s


//
// for loop
//

// CHECK-LABEL: test_for_loop_no_qref
func.func @test_for_loop_no_qref() -> f32 attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    %sum_init = arith.constant 0.0 : f32
    %t = arith.constant 37.42 : f32
    %sum = scf.for %i = %start to %stop step %step iter_args(%sum_iter = %sum_init) -> (f32) {
        %sum_next = arith.addf %sum_iter, %t : f32
        scf.yield %sum_next : f32
    }

    return %sum : f32
}

// CHECK:     {{%.+}} = scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args([[sum_iter:%.+]] = {{%.+}}) -> (f32) {
// CHECK:         [[sum:%.+]] = arith.addf [[sum_iter]], {{%.+}} : f32
// CHECK:         scf.yield [[sum]] : f32
// CHECK:     }


// -----


// CHECK-LABEL: test_for_loop_non_root
func.func @test_for_loop_non_root() attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    // CHECK: [[qreg:%.+]] = quantum.alloc( 3) : !quantum.reg
    %a = qref.alloc(3) : !qref.reg<3>
    %q0 = qref.get %a[0] : !qref.reg<3> -> !qref.bit

    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[loopOut:%.+]]:2 = scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg1 = [[bit1]], %arg2 = [[bit0]]) ->
    // CHECK-SAME:     (!quantum.bit, !quantum.bit) {
    scf.for %i = %start to %stop step %step {
        %q1 = qref.get %a[1] : !qref.reg<3> -> !qref.bit

        // CHECK: [[HADAMARD:%.+]] = quantum.custom "Hadamard"() %arg1 : !quantum.bit
        // CHECK: [[Z:%.+]] = quantum.custom "PauliZ"() %arg2 : !quantum.bit
        // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[Z]], [[HADAMARD]] : !quantum.bit, !quantum.bit
        qref.custom "Hadamard"() %q1 : !qref.bit
        qref.custom "PauliZ"() %q0 : !qref.bit
        qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

        // CHECK: scf.yield [[CNOT]]#1, [[CNOT]]#0 : !quantum.bit, !quantum.bit
        scf.yield
    }
    // CHECK: [[insert1:%.+]] = quantum.insert [[qreg]][ 1], [[loopOut]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert0:%.+]] = quantum.insert [[insert1]][ 0], [[loopOut]]#1 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert0]] : !quantum.reg
    qref.dealloc %a : !qref.reg<3>
    return
}


// -----


// CHECK-LABEL: test_for_loop_root
func.func @test_for_loop_root() attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    // CHECK: [[qubit:%.+]] = quantum.alloc_qb : !quantum.bit
    %q = qref.alloc_qb : !qref.bit

    // CHECK: [[loopOut:%.+]] = scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg1 = [[qubit]]) ->
    // CHECK-SAME:     (!quantum.bit) {
    scf.for %i = %start to %stop step %step {

        // CHECK: [[HADAMARD:%.+]] = quantum.custom "Hadamard"() %arg1 : !quantum.bit
        qref.custom "Hadamard"() %q : !qref.bit

        // CHECK: scf.yield [[HADAMARD]] : !quantum.bit
        scf.yield
    }

    // CHECK: quantum.dealloc_qb [[loopOut]] : !quantum.bit
    qref.dealloc_qb %q : !qref.bit
    return
}


// -----


// CHECK-LABEL: test_for_loop_dynamic_index
func.func @test_for_loop_dynamic_index(%nqubits: i64) attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = index.casts %nqubits : i64 to index

    // CHECK: [[qreg:%.+]] = quantum.alloc(%arg0) : !quantum.reg
    %a = qref.alloc(%nqubits) : !qref.reg<?>
    %q0 = qref.get %a[0] : !qref.reg<?> -> !qref.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[loopOut:%.+]]:2 = scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg2 = [[qreg]], %arg3 = [[q0]])
    // CHECK-SAME:     -> (!quantum.reg, !quantum.bit) {
    scf.for %i = %start to %stop step %step {

        // CHECK: [[i:%.+]] = index.casts %arg1 : index to i64
        %int = index.casts %i : index to i64

        %this_q = qref.get %a[%int] : !qref.reg<?>, i64 -> !qref.bit

        // CHECK: [[this_q:%.+]] = quantum.extract %arg2[[[i]]] : !quantum.reg -> !quantum.bit
        // CHECK: [[HADAMARD:%.+]] = quantum.custom "Hadamard"() [[this_q]] : !quantum.bit
        qref.custom "Hadamard"() %this_q : !qref.bit

        // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[HADAMARD]], %arg3 : !quantum.bit, !quantum.bit
        // CHECK: [[insertCNOT:%.+]] = quantum.insert %arg2[[[i]]], [[CNOT]]#0 : !quantum.reg, !quantum.bit
        qref.custom "CNOT"() %this_q, %q0 : !qref.bit, !qref.bit

        // CHECK: scf.yield [[insertCNOT]], [[CNOT]]#1 : !quantum.reg, !quantum.bit
        scf.yield
    }
    // CHECK: [[insertLoop:%.+]] = quantum.insert [[loopOut]]#0[ 0], [[loopOut]]#1 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insertLoop]] : !quantum.reg
    qref.dealloc %a : !qref.reg<?>
    return
}


// -----


// CHECK-LABEL: test_for_loop_with_existing_args
func.func @test_for_loop_with_existing_args(%nqubits: i64) -> (f64, f32) attributes {quantum.node} {
    // CHECK: [[sum_step:%.+]] = arith.constant 3.742000e+01 : f32
    // CHECK: [[sum_init:%.+]] = arith.constant 0.000000e+00 : f32

    // CHECK: [[qreg:%.+]] = quantum.alloc(%arg0) : !quantum.reg
    %a = qref.alloc(%nqubits) : !qref.reg<?>
    %q0 = qref.get %a[0] : !qref.reg<?> -> !qref.bit

    // CHECK: [[bit0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[bit0]] : !quantum.bit
    qref.custom "PauliX"() %q0 : !qref.bit

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = index.casts %nqubits : i64 to index

    %sum_init = arith.constant 0.0 : f32

    // CHECK: [[bit1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[loopOut:%.+]]:3 = scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg2 = [[sum_init]], %arg3 = [[bit1]], %arg4 = [[X]]) ->
    // CHECK-SAME:     (f32, !quantum.bit, !quantum.bit) {
    %sum_loop = scf.for %i = %start to %stop step %step iter_args(%sum_iter = %sum_init) -> (f32) {
        %q1 = qref.get %a[1] : !qref.reg<?> -> !qref.bit

        // CHECK: [[HADAMARD:%.+]] = quantum.custom "Hadamard"() %arg3 : !quantum.bit
        // CHECK: [[Z:%.+]] = quantum.custom "PauliZ"() %arg4 : !quantum.bit
        // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() [[Z]], [[HADAMARD]] : !quantum.bit, !quantum.bit
        qref.custom "Hadamard"() %q1 : !qref.bit
        qref.custom "PauliZ"() %q0 : !qref.bit
        qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit

        // CHECK: [[add:%.+]] = arith.addf %arg2, [[sum_step]] : f32
        // CHECK: scf.yield [[add]], [[CNOT]]#1, [[CNOT]]#0 : f32, !quantum.bit, !quantum.bit
        %t = arith.constant 37.42 : f32
        %sum_next = arith.addf %sum_iter, %t : f32
        scf.yield %sum_next : f32
    }
    // CHECK: [[insert1:%.+]] = quantum.insert [[qreg]][ 1], [[loopOut]]#1 : !quantum.reg, !quantum.bit

    // CHECK: [[obs:%.+]] = quantum.namedobs [[loopOut]]#2[ PauliX] : !quantum.obs
    // CHECK: [[insert0:%.+]] = quantum.insert [[insert1]][ 0], [[loopOut]]#2 : !quantum.reg, !quantum.bit
    %obs = qref.namedobs %q0 [ PauliX] : !quantum.obs

    // CHECK: quantum.dealloc [[insert0]] : !quantum.reg
    qref.dealloc %a : !qref.reg<?>

    // CHECK: [[expval:%.+]] = quantum.expval [[obs]]
    %expval = quantum.expval %obs : f64

    // CHECK: return [[expval]], [[loopOut]]#0
    return %expval, %sum_loop : f64, f32
}


// -----


// CHECK-LABEL: test_for_loop_nested
func.func @test_for_loop_nested() -> f64 attributes {quantum.node} {
    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    // CHECK: [[r1:%.+]] = quantum.alloc( 1) : !quantum.reg
    %r1 = qref.alloc(1) : !qref.reg<1>
    %q10 = qref.get %r1[0] : !qref.reg<1> -> !qref.bit

    // CHECK: [[q10:%.+]] = quantum.extract [[r1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[X:%.+]] = quantum.custom "PauliX"() [[q10]] : !quantum.bit
    qref.custom "PauliX"() %q10 : !qref.bit

    // CHECK: [[outerLoopResult_q10:%.+]] = scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg1 = [[X]]) -> (!quantum.bit) {
    scf.for %i = %start to %stop step %step {
        // CHECK: [[r2:%.+]] = quantum.alloc( 2) : !quantum.reg
        %r2 = qref.alloc(2) : !qref.reg<2>
        %q20 = qref.get %r2[0] : !qref.reg<2> -> !qref.bit

        // CHECK: [[q20:%.+]] = quantum.extract [[r2]][ 0] : !quantum.reg -> !quantum.bit
        // CHECK: [[CNOT:%.+]]:2 = quantum.custom "CNOT"() %arg1, [[q20]] : !quantum.bit, !quantum.bit
        qref.custom "CNOT"() %q10, %q20 : !qref.bit, !qref.bit

        // CHECK: [[innerLoopOuts_q2:%.+]]:2 = scf.for %arg2 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg3 = [[CNOT]]#0, %arg4 = [[CNOT]]#1)
        // CHECK-SAME:   -> (!quantum.bit, !quantum.bit) {
        scf.for %j = %start to %stop step %step {
            // CHECK: [[r3:%.+]] = quantum.alloc( 3) : !quantum.reg
            %r3 = qref.alloc(3) : !qref.reg<3>
            %q30 = qref.get %r3[0] : !qref.reg<3> -> !qref.bit

            // CHECK: [[q30:%.+]] = quantum.extract [[r3]][ 0] : !quantum.reg -> !quantum.bit
            // CHECK: [[Toffoli:%.+]]:3 = quantum.custom "Toffoli"() %arg3, %arg4, [[q30]] : !quantum.bit, !quantum.bit, !quantum.bit
            qref.custom "Toffoli"() %q10, %q20, %q30 : !qref.bit, !qref.bit, !qref.bit

            // CHECK: [[insert_r3:%.+]] = quantum.insert [[r3]][ 0], [[Toffoli]]#2 : !quantum.reg, !quantum.bit
            // CHECK: quantum.dealloc [[insert_r3]] : !quantum.reg
            qref.dealloc %r3 : !qref.reg<3>

            // CHECK: scf.yield [[Toffoli]]#0, [[Toffoli]]#1 : !quantum.bit, !quantum.bit
            scf.yield
        }
        // CHECK: [[insert_r2:%.+]] = quantum.insert [[r2]][ 0], [[innerLoopOuts_q2]]#1 : !quantum.reg, !quantum.bit

        // CHECK: quantum.dealloc [[insert_r2]] : !quantum.reg
        qref.dealloc %r2 : !qref.reg<2>

        // CHECK: scf.yield [[innerLoopOuts_q2]]#0 : !quantum.bit
        scf.yield
    }

    // CHECK: [[obs:%.+]] = quantum.namedobs [[outerLoopResult_q10]][ PauliX] : !quantum.obs
    // CHECK: [[insert_r1:%.+]] = quantum.insert [[r1]][ 0], [[outerLoopResult_q10]] : !quantum.reg, !quantum.bit
    %obs = qref.namedobs %q10 [ PauliX] : !quantum.obs

    // CHECK: quantum.expval [[obs]]
    %expval = quantum.expval %obs : f64

    // CHECK: quantum.dealloc [[insert_r1]] : !quantum.reg
    qref.dealloc %r1 : !qref.reg<1>
    return %expval : f64
}


// -----


//
// while loop
//


// CHECK-LABEL: test_while_loop_no_qref
func.func @test_while_loop_no_qref(%arg0 : f32, %arg1 : i1) -> f32 attributes {quantum.node} {
    %res = scf.while (%arg2 = %arg0) : (f32) -> f32 {
        scf.condition(%arg1) %arg2 : f32
    } do {
        ^bb0(%arg2: f32):
        %0 = arith.negf %arg2 : f32
        scf.yield %0 : f32
    }
    return %res : f32
}

// CHECK:    {{%.+}} = scf.while (%arg2 = %arg0) : (f32) -> f32 {
// CHECK:      scf.condition(%arg1) %arg2 : f32
// CHECK:    } do {
// CHECK:    ^bb0(%arg2: f32):
// CHECK:      [[yield:%.+]] = arith.negf %arg2 : f32
// CHECK:      scf.yield [[yield]] : f32
// CHECK:    }



// -----


// CHECK-LABEL: test_while_loop_non_root
func.func @test_while_loop_non_root(%arg0: i1, %arg1: f64) -> (f64, f64) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 3) : !quantum.reg
    %a = qref.alloc(3) : !qref.reg<3>
    %q2 = qref.get %a[2] : !qref.reg<3> -> !qref.bit

    // CHECK: [[q2:%.+]] = quantum.extract [[qreg]][ 2] : !quantum.reg -> !quantum.bit
    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[loopOut:%.+]]:5 = scf.while (%arg2 = %arg1, %arg3 = [[q2]], %arg4 = [[q0]], %arg5 = [[q1]])
    // CHECK-SAME:   (f64, !quantum.bit, !quantum.bit, !quantum.bit) -> (f64, f64, !quantum.bit, !quantum.bit, !quantum.bit)
    %final_angle, %37 = scf.while (%arg2 = %arg1) : (f64) -> (f64, f64) {

        // CHECK: [[mres:%.+]], [[MEASURE:%.+]] = quantum.measure %arg3 : i1, !quantum.bit
        %mres = qref.measure %q2 : i1
        %neg = arith.negf %arg2 : f64

        // CHECK: scf.condition([[mres]]) %arg2, {{%.+}}, [[MEASURE]], %arg4, %arg5 : f64, f64, !quantum.bit, !quantum.bit, !quantum.bit
        scf.condition(%mres) %arg2, %neg : f64, f64
    } do {
        // CHECK: ^bb0(%arg2: f64, %arg3: f64, [[q2:%.+]]: !quantum.bit, [[q0:%.+]]: !quantum.bit, [[q1:%.+]]: !quantum.bit):
        ^bb0(%arg2: f64, %arg3: f64):
        %q0 = qref.get %a[0] : !qref.reg<3> -> !qref.bit
        %q1 = qref.get %a[1] : !qref.reg<3> -> !qref.bit

        // CHECK: [[GATE:%.+]]:3 = quantum.custom "gate"(%arg2, %arg3) [[q0]], [[q1]], [[q2]] : !quantum.bit, !quantum.bit, !quantum.bit
        qref.custom "gate"(%arg2, %arg3) %q0, %q1, %q2 : !qref.bit, !qref.bit, !qref.bit

        %increment = arith.constant 0.1 : f64
        %add = arith.addf %arg2, %increment : f64

        // CHECK: scf.yield {{%.+}}, [[GATE]]#2, [[GATE]]#0, [[GATE]]#1 : f64, !quantum.bit, !quantum.bit, !quantum.bit
        scf.yield %add : f64
    }

    // CHECK: [[insert2:%.+]] = quantum.insert [[qreg]][ 2], [[loopOut]]#2 : !quantum.reg, !quantum.bit
    // CHECK: [[insert0:%.+]] = quantum.insert [[insert2]][ 0], [[loopOut]]#3 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[loopOut]]#4 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<3>

    // CHECK: return [[loopOut]]#0, [[loopOut]]#1 : f64, f64
    return %final_angle, %37 : f64, f64
}


// -----


// CHECK-LABEL: test_while_loop_root
func.func @test_while_loop_root(%arg0: i1, %arg1: f64) attributes {quantum.node} {

    // CHECK: [[qubit:%.+]] = quantum.alloc_qb : !quantum.bit
    %q = qref.alloc_qb : !qref.bit

    // CHECK: [[loopOut:%.+]] = scf.while (%arg2 = [[qubit]]) : (!quantum.bit) -> !quantum.bit {
    scf.while () : () -> () {

        // CHECK: [[mres:%.+]], [[MEASURE:%.+]] = quantum.measure %arg2 : i1, !quantum.bit
        %mres = qref.measure %q : i1

        // CHECK: scf.condition([[mres]]) [[MEASURE]] : !quantum.bit
        scf.condition(%mres)
    } do {
        // CHECK: ^bb0(%arg2: !quantum.bit):
        ^bb0():

        // CHECK: [[GATE:%.+]] = quantum.custom "gate"(%arg1) %arg2 : !quantum.bit
        qref.custom "gate"(%arg1) %q : !qref.bit

        // CHECK: scf.yield [[GATE]] : !quantum.bit
        scf.yield
    }

    // CHECK: quantum.dealloc_qb [[loopOut]] : !quantum.bit
    qref.dealloc_qb %q : !qref.bit
    return
}


// -----


// CHECK-LABEL: test_while_dynamic_index
func.func @test_while_dynamic_index(%arg0: i1, %arg1: i64) -> i64 attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 3) : !quantum.reg
    %a = qref.alloc(3) : !qref.reg<3>

    // CHECK: [[loopOut:%.+]]:2 = scf.while (%arg2 = %arg1, %arg3 = [[qreg]])
    // CHECK-SAME:   (i64, !quantum.reg) -> (i64, !quantum.reg)
    %i = scf.while (%arg2 = %arg1) : (i64) -> (i64) {

        // CHECK: scf.condition(%arg0) %arg2, %arg3 : i64, !quantum.reg
        scf.condition(%arg0) %arg2 : i64
    } do {
        // CHECK: ^bb0(%arg2: i64, %arg3: !quantum.reg):
        ^bb0(%arg2: i64):
        %q0 = qref.get %a[%arg2] : !qref.reg<3>, i64 -> !qref.bit

        // CHECK: [[q0:%.+]] = quantum.extract %arg3[%arg2] : !quantum.reg -> !quantum.bit
        // CHECK: [[GATE:%.+]] = quantum.custom "gate"() [[q0]] : !quantum.bit
        // CHECK: [[insert:%.+]] = quantum.insert %arg3[%arg2], [[GATE]] : !quantum.reg, !quantum.bit
        qref.custom "gate"() %q0 : !qref.bit

        // CHECK: scf.yield {{%.+}}, [[insert]] : i64, !quantum.reg

        %increment = arith.constant 1 : i64
        %add = arith.addi %arg2, %increment : i64
        scf.yield %add : i64
    }

    // CHECK: quantum.dealloc [[loopOut]]#1 : !quantum.reg
    qref.dealloc %a : !qref.reg<3>

    // CHECK: return [[loopOut]]#0 : i64
    return %i : i64
}


// -----


//
// if statements
//
