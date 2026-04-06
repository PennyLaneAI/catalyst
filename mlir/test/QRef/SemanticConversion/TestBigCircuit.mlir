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

// RUN: quantum-opt --convert-to-value-semantics --canonicalize --verify-diagnostics %s | FileCheck %s


// CHECK: func.func @subroutine(%arg0: f64, %arg1: i64, %arg2: !quantum.reg) -> !quantum.reg {
// CHECK:     [[extract:%.+]] = quantum.extract %arg2[%arg1] : !quantum.reg -> !quantum.bit
// CHECK:     [[RX:%.+]] = quantum.custom "RX"(%arg0) [[extract]] : !quantum.bit
// CHECK:     [[insert:%.+]] = quantum.insert %arg2[%arg1], [[RX]] : !quantum.reg, !quantum.bit
// CHECK:     return [[insert]] : !quantum.reg
// CHECK: }

func.func @subroutine(%arg0: f64, %idx: i64, %reg: !qref.reg<6>) -> () {
    %q = qref.get %reg[%idx] : !qref.reg<6>, i64 -> !qref.bit
    qref.custom "RX"(%arg0) %q : !qref.bit
    return
}


// CHECK: func.func @main(%arg0: i1, %arg1: i64, %arg2: f64) -> tensor<64xf64> attributes {quantum.node}
func.func @main(%arg0: i1, %arg1: i64, %arg2: f64) -> tensor<64xf64> attributes {quantum.node} {

    // CHECK: [[shots:%.+]] = arith.constant 1000 : i64
    // CHECK: quantum.device shots([[shots]]) ["", "", ""]
    %1000 = arith.constant 1000 : i64
    quantum.device shots(%1000) ["", "", ""]

    // CHECK: [[reg6:%.+]] = quantum.alloc( 6) : !quantum.reg
    // CHECK: [[reg1:%.+]] = quantum.alloc( 1) : !quantum.reg
    %reg = qref.alloc(6) : !qref.reg<6>
    %reg_dyn_alloc = qref.alloc(1) : !qref.reg<1>

    %start = arith.constant 0 : index
    %step = arith.constant 1 : index
    %stop = arith.constant 6 : index

    // CHECK: [[q_dyn_alloc:%.+]] = quantum.extract [[reg1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[forOut:%.+]]:2 = scf.for %arg3 = {{%.+}} to {{%.+}} step
    // CHECK-SAME: iter_args(%arg4 = [[reg6]], %arg5 = [[q_dyn_alloc]]) -> (!quantum.reg, !quantum.bit) {
    scf.for %i = %start to %stop step %step {

        // CHECK: [[i:%.+]] = index.casts %arg3 : index to i64
        %int = index.casts %i : index to i64
        %this_q = qref.get %reg[%int] : !qref.reg<6>, i64 -> !qref.bit

        // CHECK: [[this_q:%.+]] = quantum.extract %arg4[[[i]]] : !quantum.reg -> !quantum.bit
        // CHECK: [[HADAMARD:%.+]] = quantum.custom "Hadamard"() [[this_q]] : !quantum.bit
        qref.custom "Hadamard"() %this_q : !qref.bit
        // CHECK: [[H_insert:%.+]] = quantum.insert %arg4[[[i]]], [[HADAMARD]] : !quantum.reg, !quantum.bit

        // CHECK: [[ifOut:%.+]]:2 = scf.if %arg0 -> (!quantum.reg, !quantum.bit)
        scf.if %arg0 {
            // CHECK: [[callOut:%.+]] = func.call @subroutine(%arg2, [[i]], [[H_insert]]) : (f64, i64, !quantum.reg) -> !quantum.reg
            func.call @subroutine(%arg2, %int, %reg) : (f64, i64, !qref.reg<6>) -> ()

            // CHECK: scf.yield [[callOut]], %arg5 : !quantum.reg, !quantum.bit

        // CHECK: else
        } else {
            %q_arg1 = qref.get %reg[%arg1] : !qref.reg<6>, i64 -> !qref.bit
            %q_dyn_alloc = qref.get %reg_dyn_alloc[ 0] : !qref.reg<1> -> !qref.bit

            // CHECK: [[extract_i:%.+]] = quantum.extract [[H_insert]][[[i]]] : !quantum.reg -> !quantum.bit
            // CHECK: [[extract_arg1:%.+]] = quantum.extract [[H_insert]][%arg1] : !quantum.reg -> !quantum.bit
            // CHECK: [[ROT:%.+]]:3 = quantum.custom "Rot"(%arg2) [[extract_i]], [[extract_arg1]], %arg5 adj : !quantum.bit, !quantum.bit, !quantum.bit
            qref.custom "Rot"(%arg2) %this_q, %q_arg1, %q_dyn_alloc adj : !qref.bit, !qref.bit, !qref.bit
            // CHECK: [[insert_i:%.+]] = quantum.insert [[H_insert]][[[i]]], [[ROT]]#0 : !quantum.reg, !quantum.bit
            // CHECK: [[insert_arg1:%.+]] = quantum.insert [[insert_i]][%arg1], [[ROT]]#1 : !quantum.reg, !quantum.bit

            // CHECK: scf.yield [[insert_arg1]], [[ROT]]#2 : !quantum.reg, !quantum.bit
        }
        // CHECK: scf.yield [[ifOut]]#0, [[ifOut]]#1 : !quantum.reg, !quantum.bit
    }
    // CHECK: [[reg1_afterForInsert:%.+]] = quantum.insert [[reg1]][ 0], [[forOut]]#1 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[reg1_afterForInsert]] : !quantum.reg
    qref.dealloc %reg_dyn_alloc : !qref.reg<1>


    %q0 = qref.get %reg[ 0] : !qref.reg<6> -> !qref.bit
    %q3 = qref.get %reg[ 3] : !qref.reg<6> -> !qref.bit

    // CHECK: [[q3:%.+]] = quantum.extract [[forOut]]#0[ 3] : !quantum.reg -> !quantum.bit
    // CHECK: [[q0:%.+]] = quantum.extract [[forOut]]#0[ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[SWAP:%.+]]:2 = quantum.custom "SWAP"() [[q3]], [[q0]] : !quantum.bit, !quantum.bit
    qref.custom "SWAP"() %q3, %q0 : !qref.bit, !qref.bit
    // CHECK: [[insert3_swap:%.+]] = quantum.insert [[forOut]]#0[ 3], [[SWAP]]#0 : !quantum.reg, !quantum.bit

    %q0_again = qref.get %reg[ 0] : !qref.reg<6> -> !qref.bit

    // CHECK: [[mres_init:%.+]], [[MEASURE_init:%.+]] = quantum.measure [[SWAP]]#1 : i1, !quantum.bit
    %mres_init = qref.measure %q0_again : i1
    // CHECK: [[insert0_swap:%.+]] = quantum.insert [[insert3_swap]][ 0], [[MEASURE_init]] : !quantum.reg, !quantum.bit

    // CHECK: [[whileOut:%.+]] = scf.while (%arg3 = [[mres_init]], %arg4 = [[insert0_swap]])
    // CHECK-SAME: (i1, !quantum.reg) -> !quantum.reg {
    scf.while (%mres = %mres_init) : (i1) -> () {

        // CHECK: scf.condition(%arg3) %arg4 : !quantum.reg
        scf.condition(%mres)

    // CHECK: do
    // CHECK: ^bb0([[reg6_whileArg:%.+]]: !quantum.reg):
    } do {
        ^bb0():
        %q0_again_again = qref.get %reg[0] : !qref.reg<6> -> !qref.bit

        // CHECK: [[extract:%.+]] = quantum.extract [[reg6_whileArg]][ 0] : !quantum.reg -> !quantum.bit
        // CHECK: [[mres_0:%.+]], [[MEASURE_0:%.+]] = quantum.measure [[extract]] : i1, !quantum.bit
        %mres_0 = qref.measure %q0_again_again : i1
        // CHECK: [[insert:%.+]] = quantum.insert [[reg6_whileArg]][ 0], [[MEASURE_0]] : !quantum.reg, !quantum.bit

        %q_arg1_again = qref.get %reg[%arg1] : !qref.reg<6>, i64 -> !qref.bit

        // CHECK: [[extract:%.+]] = quantum.extract [[insert]][%arg1] : !quantum.reg -> !quantum.bit
        // CHECK: [[mres_1:%.+]], [[MEASURE_1:%.+]] = quantum.measure [[extract]] : i1, !quantum.bit
        %mres_1 = qref.measure %q_arg1_again : i1
        // CHECK: [[insert_arg1:%.+]] = quantum.insert [[insert]][%arg1], [[MEASURE_1]] : !quantum.reg, !quantum.bit

        // CHECK: [[and:%.+]] = arith.andi [[mres_0]], [[mres_1]] : i1
        %and = arith.andi %mres_0, %mres_1 : i1

        // CHECK: scf.yield [[and]], [[insert_arg1]] : i1, !quantum.reg
        scf.yield %and : i1
    }

    // CHECK: [[obs:%.+]] = quantum.compbasis qreg [[whileOut]] : !quantum.obs
    // CHECK: [[probs:%.+]] = quantum.probs [[obs]] : tensor<64xf64>
    %obs = qref.compbasis (qreg %reg : !qref.reg<6>) : !quantum.obs
    %probs = quantum.probs %obs : tensor<64xf64>

    // CHECK: quantum.dealloc [[whileOut]] : !quantum.reg
    qref.dealloc %reg : !qref.reg<6>

    // CHECK: return [[probs]] : tensor<64xf64>
    return %probs : tensor<64xf64>
}
