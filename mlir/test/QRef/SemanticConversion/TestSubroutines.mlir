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


// CHECK: func.func @test_no_extract_before_call_with_static_idx(%arg0: f64, %arg1: i64, %arg2: !quantum.reg) -> !quantum.reg {
// CHECK:     [[extract_i:%.+]] = quantum.extract %arg2[%arg1] : !quantum.reg -> !quantum.bit
// CHECK:     [[extract_0:%.+]] = quantum.extract %arg2[ 0] : !quantum.reg -> !quantum.bit
// CHECK:     [[ROT:%.+]]:2 = quantum.custom "Rot"(%arg0) [[extract_i]], [[extract_0]] : !quantum.bit, !quantum.bit
// CHECK:     [[insert_i:%.+]] = quantum.insert %arg2[%arg1], [[ROT]]#0 : !quantum.reg, !quantum.bit
// CHECK:     [[X:%.+]] = quantum.custom "X"() [[ROT]]#1 : !quantum.bit
// CHECK:     [[insert_0:%.+]] = quantum.insert [[insert_i]][ 0], [[X]] : !quantum.reg, !quantum.bit
// CHECK:     return [[insert_0]] : !quantum.reg
// CHECK: }

func.func @test_no_extract_before_call_with_static_idx(%reg: !qref.reg<2>, %param: f64, %idx: i64) -> () {
    %qi = qref.get %reg[%idx] : !qref.reg<2>, i64 -> !qref.bit
    %q0 = qref.get %reg[0] : !qref.reg<2> -> !qref.bit
    qref.custom "Rot"(%param) %qi, %q0 : !qref.bit, !qref.bit

    %q0_alias = qref.get %reg[0] : !qref.reg<2> -> !qref.bit
    qref.custom "X"() %q0_alias : !qref.bit

    return
}

// CHECK: func.func @main(%arg0: f64, %arg1: i64) attributes {quantum.node}
func.func @main(%arg0: f64, %arg1: i64) -> () attributes {quantum.node} {
    // CHECK: [[r2_0:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r2_1:%.+]] = quantum.alloc( 2) : !quantum.reg
    %r2_0 = qref.alloc(2) : !qref.reg<2>
    %r2_1 = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[first_call:%.+]] = call @test_no_extract_before_call_with_static_idx(%arg0, %arg1, [[r2_0]]) : (f64, i64, !quantum.reg) -> !quantum.reg
    // CHECK: [[second_call:%.+]] = call @test_no_extract_before_call_with_static_idx(%arg0, %arg1, [[r2_1]]) : (f64, i64, !quantum.reg) -> !quantum.reg
    func.call @test_no_extract_before_call_with_static_idx(%r2_0, %arg0, %arg1) : (!qref.reg<2>, f64, i64) -> ()
    func.call @test_no_extract_before_call_with_static_idx(%r2_1, %arg0, %arg1) : (!qref.reg<2>, f64, i64) -> ()

    // CHECK: quantum.dealloc [[first_call]] : !quantum.reg
    // CHECK: quantum.dealloc [[second_call]] : !quantum.reg
    qref.dealloc %r2_0 : !qref.reg<2>
    qref.dealloc %r2_1 : !qref.reg<2>

    return
}


// -----


// CHECK: func.func @test_single_qubit_alloc(%arg0: f64, %arg1: !quantum.bit) -> !quantum.bit {
// CHECK:   [[RX:%.+]] = quantum.custom "RX"(%arg0) %arg1 : !quantum.bit
// CHECK:   return [[RX]] : !quantum.bit
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


// -----


// CHECK: func.func @test_mixed_args(%arg0: i64, [[q20:%.+]]: !quantum.bit, %arg2: !quantum.reg, [[q:%.+]]: !quantum.bit) ->
// CHECK-SAME:   (i1, i1, i1, !quantum.bit, !quantum.reg, !quantum.bit) {
// CHECK:   [[extract:%.+]] = quantum.extract %arg2[%arg0] : !quantum.reg -> !quantum.bit
// CHECK:   [[GATE:%.+]]:3 = quantum.custom "gate"() [[q20]], [[extract]], [[q]] : !quantum.bit, !quantum.bit, !quantum.bit
// CHECK:   [[q20_mres:%.+]], [[q20_MEASURE:%.+]] = quantum.measure [[GATE]]#0 : i1, !quantum.bit
// CHECK:   [[extract_mres:%.+]], [[extract_MEASURE:%.+]] = quantum.measure [[GATE]]#1 : i1, !quantum.bit
// CHECK:   [[insert:%.+]] = quantum.insert %arg2[%arg0], [[extract_MEASURE]] : !quantum.reg, !quantum.bit
// CHECK:   [[q_mres:%.+]], [[q_MEASURE:%.+]] = quantum.measure [[GATE]]#2 : i1, !quantum.bit
// CHECK:   return [[q20_mres]], [[extract_mres]], [[q_mres]], [[q20_MEASURE]], [[insert]], [[q_MEASURE]]
// CHECK-SAME:     i1, i1, i1, !quantum.bit, !quantum.reg, !quantum.bit
// CHECK: }

func.func @test_mixed_args(%q: !qref.bit, %r2: !qref.reg<2>, %r3: !qref.reg<3>, %idx: i64) -> (i1, i1, i1) {
    %q20 = qref.get %r2[0] : !qref.reg<2> -> !qref.bit
    %q3 = qref.get %r3[%idx] : !qref.reg<3>, i64 -> !qref.bit
    qref.custom "gate"() %q20, %q3, %q : !qref.bit, !qref.bit, !qref.bit

    %mres_q20 = qref.measure %q20 : i1
    %mres_q3 = qref.measure %q3 : i1
    %mres_q = qref.measure %q : i1
    return %mres_q20, %mres_q3, %mres_q : i1, i1, i1
}

// CHECK: func.func @main(%arg0: i64) -> (i1, i1, i1, i1, i1, i1) attributes {quantum.node}
func.func @main(%arg0: i64) -> (i1, i1, i1, i1, i1, i1) attributes {quantum.node} {
    // CHECK: [[q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[r2:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r3:%.+]] = quantum.alloc( 3) : !quantum.reg
    %q = qref.alloc_qb : !qref.bit
    %r2 = qref.alloc(2) : !qref.reg<2>
    %r3 = qref.alloc(3) : !qref.reg<3>

    // CHECK: [[q20:%.+]] = quantum.extract [[r2]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[first_call:%.+]]:6 = call @test_mixed_args(%arg0, [[q20]], [[r3]], [[q]])
    // CHECK-SAME:  (i64, !quantum.bit, !quantum.reg, !quantum.bit) -> (i1, i1, i1, !quantum.bit, !quantum.reg, !quantum.bit)
    // CHECK: [[second_call:%.+]]:6 = call @test_mixed_args(%arg0, [[first_call]]#3, [[first_call]]#4, [[first_call]]#5) :
    // CHECK-SAME:  (i64, !quantum.bit, !quantum.reg, !quantum.bit) -> (i1, i1, i1, !quantum.bit, !quantum.reg, !quantum.bit)
    %call0:3 = func.call @test_mixed_args(%q, %r2, %r3, %arg0) : (!qref.bit, !qref.reg<2>, !qref.reg<3>, i64) -> (i1, i1, i1)
    %call1:3 = func.call @test_mixed_args(%q, %r2, %r3, %arg0) : (!qref.bit, !qref.reg<2>, !qref.reg<3>, i64) -> (i1, i1, i1)
    // CHECK: [[insert_r2:%.+]] = quantum.insert [[r2]][ 0], [[second_call]]#3 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc_qb [[second_call]]#5 : !quantum.bit
    // CHECK: quantum.dealloc [[insert_r2]] : !quantum.reg
    // CHECK: quantum.dealloc [[second_call]]#4 : !quantum.reg
    qref.dealloc_qb %q : !qref.bit
    qref.dealloc %r2 : !qref.reg<2>
    qref.dealloc %r3 : !qref.reg<3>

    // CHECK: return [[first_call]]#0, [[first_call]]#1, [[first_call]]#2,
    // CHECK-SAME:  [[second_call]]#0, [[second_call]]#1, [[second_call]]#2 : i1, i1, i1, i1, i1, i1
    return %call0#0, %call0#1, %call0#2, %call1#0, %call1#1, %call1#2 : i1, i1, i1, i1, i1, i1
}


// -----


// CHECK:   func.func @test_loop_callsite(%arg0: !quantum.bit, %arg1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
// CHECK:     [[CNOT:%.+]]:2 = quantum.custom "CNOT"() %arg0, %arg1 : !quantum.bit, !quantum.bit
// CHECK:     return [[CNOT]]#0, [[CNOT]]#1 : !quantum.bit, !quantum.bit
// CHECK:   }

func.func @test_loop_callsite(%reg: !qref.reg<2>, %q: !qref.bit) -> () {
    %q0 = qref.get %reg[0] : !qref.reg<2> -> !qref.bit
    qref.custom "CNOT"() %q, %q0 : !qref.bit, !qref.bit
    return
}

// CHECK: func.func @main() attributes {quantum.node}
func.func @main() -> () attributes {quantum.node} {
    // CHECK: [[q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[r0:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r1:%.+]] = quantum.alloc( 2) : !quantum.reg
    %q = qref.alloc_qb : !qref.bit
    %r0 = qref.alloc(2) : !qref.reg<2>
    %r1 = qref.alloc(2) : !qref.reg<2>

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    // CHECK: [[q0:%.+]] = quantum.extract [[r0]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[r1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[loopOut:%.+]]:3 = scf.for %arg0 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg1 = [[q]], %arg2 = [[q0]], %arg3 = [[q1]])
    // CHECK-SAME:   -> (!quantum.bit, !quantum.bit, !quantum.bit)
    scf.for %i = %start to %stop step %step {
        // CHECK: [[first_call:%.+]]:2 = func.call @test_loop_callsite(%arg1, %arg2) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        // CHECK: [[second_call:%.+]]:2 = func.call @test_loop_callsite([[first_call]]#0, %arg3) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        func.call @test_loop_callsite(%r0, %q) : (!qref.reg<2>, !qref.bit) -> ()
        func.call @test_loop_callsite(%r1, %q) : (!qref.reg<2>, !qref.bit) -> ()

        // CHECK: scf.yield [[second_call]]#0, [[first_call]]#1, [[second_call]]#1 : !quantum.bit, !quantum.bit, !quantum.bit
    }
    // CHECK: [[insert0:%.+]] = quantum.insert [[r0]][ 0], [[loopOut]]#1 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[r1]][ 0], [[loopOut]]#2 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc_qb [[loopOut]]#0 : !quantum.bit
    // CHECK: quantum.dealloc [[insert0]] : !quantum.reg
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc_qb %q : !qref.bit
    qref.dealloc %r0 : !qref.reg<2>
    qref.dealloc %r1 : !qref.reg<2>

    return
}


// -----


// CHECK:   func.func @test_cond_callsite(%arg0: !quantum.bit, %arg1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
// CHECK:     [[CNOT:%.+]]:2 = quantum.custom "CNOT"() %arg0, %arg1 : !quantum.bit, !quantum.bit
// CHECK:     return [[CNOT]]#0, [[CNOT]]#1 : !quantum.bit, !quantum.bit
// CHECK:   }

func.func @test_cond_callsite(%reg: !qref.reg<2>, %q: !qref.bit) -> () {
    %q0 = qref.get %reg[0] : !qref.reg<2> -> !qref.bit
    qref.custom "CNOT"() %q, %q0 : !qref.bit, !qref.bit
    return
}

// CHECK: func.func @main(%arg0: i1) attributes {quantum.node}
func.func @main(%arg0: i1) -> () attributes {quantum.node} {
    // CHECK: [[q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[r0:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r1:%.+]] = quantum.alloc( 2) : !quantum.reg
    %q = qref.alloc_qb : !qref.bit
    %r0 = qref.alloc(2) : !qref.reg<2>
    %r1 = qref.alloc(2) : !qref.reg<2>

    // CHECK: [[q0:%.+]] = quantum.extract [[r0]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[r1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[ifOut:%.+]]:3 = scf.if %arg0 -> (!quantum.bit, !quantum.bit, !quantum.bit)
    scf.if %arg0 {
        // CHECK: [[first_call:%.+]]:2 = func.call @test_cond_callsite([[q]], [[q0]]) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        // CHECK: [[second_call:%.+]]:2 = func.call @test_cond_callsite([[first_call]]#0, [[q1]]) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
        func.call @test_cond_callsite(%r0, %q) : (!qref.reg<2>, !qref.bit) -> ()
        func.call @test_cond_callsite(%r1, %q) : (!qref.reg<2>, !qref.bit) -> ()

        // CHECK: scf.yield [[second_call]]#0, [[first_call]]#1, [[second_call]]#1 : !quantum.bit, !quantum.bit, !quantum.bit
    }
    // CHECK: } else {
    // CHECK:   scf.yield [[q]], [[q0]], [[q1]] : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: }
    // CHECK: [[insert0:%.+]] = quantum.insert [[r0]][ 0], [[ifOut]]#1 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[r1]][ 0], [[ifOut]]#2 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc_qb [[ifOut]]#0 : !quantum.bit
    // CHECK: quantum.dealloc [[insert0]] : !quantum.reg
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc_qb %q : !qref.bit
    qref.dealloc %r0 : !qref.reg<2>
    qref.dealloc %r1 : !qref.reg<2>

    return
}


// -----


// CHECK:   func.func @test_callsite_in_nested_region(%arg0: !quantum.bit, %arg1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
// CHECK:     [[CNOT:%.+]]:2 = quantum.custom "CNOT"() %arg0, %arg1 : !quantum.bit, !quantum.bit
// CHECK:     return [[CNOT]]#0, [[CNOT]]#1 : !quantum.bit, !quantum.bit
// CHECK:   }

func.func @test_callsite_in_nested_region(%reg: !qref.reg<2>, %q: !qref.bit) -> () {
    %q0 = qref.get %reg[0] : !qref.reg<2> -> !qref.bit
    qref.custom "CNOT"() %q, %q0 : !qref.bit, !qref.bit
    return
}

// CHECK: func.func @main(%arg0: i1) attributes {quantum.node}
func.func @main(%arg0: i1) -> () attributes {quantum.node} {
    // CHECK: [[q:%.+]] = quantum.alloc_qb : !quantum.bit
    // CHECK: [[r0:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[r1:%.+]] = quantum.alloc( 2) : !quantum.reg
    %q = qref.alloc_qb : !qref.bit
    %r0 = qref.alloc(2) : !qref.reg<2>
    %r1 = qref.alloc(2) : !qref.reg<2>

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 37 : index

    // CHECK: [[q0:%.+]] = quantum.extract [[r0]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[r1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[ifOut:%.+]]:3 = scf.if %arg0 -> (!quantum.bit, !quantum.bit, !quantum.bit)
    scf.if %arg0 {
        // CHECK: [[loopOut:%.+]]:3 = scf.for %arg1 = {{%.+}} to {{%.+}} step {{%.+}} iter_args(%arg2 = [[q]], %arg3 = [[q0]], %arg4 = [[q1]])
        // CHECK-SAME:   -> (!quantum.bit, !quantum.bit, !quantum.bit)
        scf.for %i = %start to %stop step %step {
            // CHECK: [[first_call:%.+]]:2 = func.call @test_callsite_in_nested_region(%arg2, %arg3) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            // CHECK: [[second_call:%.+]]:2 = func.call @test_callsite_in_nested_region([[first_call]]#0, %arg4) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            func.call @test_callsite_in_nested_region(%r0, %q) : (!qref.reg<2>, !qref.bit) -> ()
            func.call @test_callsite_in_nested_region(%r1, %q) : (!qref.reg<2>, !qref.bit) -> ()

            // CHECK: scf.yield [[second_call]]#0, [[first_call]]#1, [[second_call]]#1 : !quantum.bit, !quantum.bit, !quantum.bit
        }

        // CHECK: scf.yield [[loopOut]]#0, [[loopOut]]#1, [[loopOut]]#2 : !quantum.bit, !quantum.bit, !quantum.bit
    }
    // CHECK: } else {
    // CHECK:   scf.yield [[q]], [[q0]], [[q1]] : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: }
    // CHECK: [[insert0:%.+]] = quantum.insert [[r0]][ 0], [[ifOut]]#1 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[r1]][ 0], [[ifOut]]#2 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc_qb [[ifOut]]#0 : !quantum.bit
    // CHECK: quantum.dealloc [[insert0]] : !quantum.reg
    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc_qb %q : !qref.bit
    qref.dealloc %r0 : !qref.reg<2>
    qref.dealloc %r1 : !qref.reg<2>

    return
}


// -----


// test subroutines calling other subroutines

// CHECK:   func.func @callee(%arg0: !quantum.bit) -> !quantum.bit {
// CHECK:     [[X:%.+]] = quantum.custom "PauliX"() %arg0 : !quantum.bit
// CHECK:     return [[X]] : !quantum.bit
// CHECK:   }
func.func @callee(%r: !qref.reg<1>) {
    %q = qref.get %r[0] : !qref.reg<1> -> !qref.bit
    qref.custom "PauliX"() %q : !qref.bit
    return
}

// CHECK:   func.func @caller(%arg0: !quantum.bit) -> !quantum.bit {
// CHECK:     [[call:%.+]] = call @callee(%arg0) : (!quantum.bit) -> !quantum.bit
// CHECK:     return [[call]] : !quantum.bit
// CHECK:   }
func.func @caller(%r: !qref.reg<1>) {
    func.call @callee(%r) : (!qref.reg<1>) -> ()
    return
}

// CHECK: func.func @main() attributes {quantum.node}
func.func @main() attributes {quantum.node} {
    // CHECK: [[r:%.+]] = quantum.alloc( 1) : !quantum.reg
    %r = qref.alloc(1) : !qref.reg<1>

    // CHECK: [[extract:%.+]] = quantum.extract [[r]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[call:%.+]] = call @caller([[extract]]) : (!quantum.bit) -> !quantum.bit
    func.call @caller(%r) : (!qref.reg<1>) -> ()
    // CHECK: [[insert:%.+]] = quantum.insert [[r]][ 0], [[call]] : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc %r : !qref.reg<1>
    return
}


// -----


func.func @callee(%r: !qref.reg<1>) {
    func.call @caller(%r) : (!qref.reg<1>) -> ()
    return
}

// expected-error@+1 {{Quantum subroutine call graphs must not have cycles}}
func.func @caller(%r: !qref.reg<1>) {
    func.call @callee(%r) : (!qref.reg<1>) -> ()
    return
}
