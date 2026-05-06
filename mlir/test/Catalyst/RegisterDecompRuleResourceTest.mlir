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

// RUN: quantum-opt --pass-pipeline="builtin.module(register-decomp-rule-resource)" --split-input-file %s | FileCheck %s

// Basic decomposition rule

// CHECK: resources = {measurements = {}, num_alloc_qubits = 2 : i64, num_arg_qubits = 0 : i64, num_qubits = 2 : i64, operations = {"CNOT(2,0)" = 1 : i64, "Hadamard(1,0)" = 1 : i64, "S(1,0)" = 1 : i64, "T(1,0)" = 1 : i64}}, target_gate = "basic"
func.func @basic_gates() attributes {target_gate="basic"}  {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %4 = quantum.custom "T"() %3 : !quantum.bit
    %5 = quantum.custom "S"() %2 : !quantum.bit
    %6:2 = quantum.custom "CNOT"() %4, %5 : !quantum.bit, !quantum.bit
    %7 = quantum.insert %0[ 0], %6#0 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %6#1 : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}

// -----

// Rule with PBC ops

// CHECK: resources = {measurements = {}, num_alloc_qubits = 2 : i64, num_arg_qubits = 0 : i64, num_qubits = 2 : i64, operations = {"PPM(1,0)" = 2 : i64, "PPR-pi/4(1,0)" = 3 : i64, "PPR-pi/8(1,0)" = 1 : i64}}, target_gate = "pbc"
func.func @pbc_operations() attributes {target_gate="pbc"} {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = pbc.ppr ["Z"](4) %1 : !quantum.bit
    %4 = pbc.ppr ["X"](4) %3 : !quantum.bit
    %5 = pbc.ppr ["Z"](4) %2 : !quantum.bit
    %6 = pbc.ppr ["Z"](8) %4 : !quantum.bit
    %mres, %out = pbc.ppm ["Z"] %5 : i1, !quantum.bit
    %mres2, %out2 = pbc.ppm ["X"] %6 : i1, !quantum.bit
    %7 = quantum.insert %0[ 0], %out2 : !quantum.reg, !quantum.bit
    %8 = quantum.insert %7[ 1], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %8 : !quantum.reg
    return
}

// -----

// Rule with measure

// CHECK: resources = {measurements = {MidCircuitMeasure = 1 : i64},
func.func @rule_mcm() attributes {target_gate="gate"} {
    %0 = quantum.alloc( 1) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %mres, %out = quantum.measure %2 : i1, !quantum.bit
    %3 = quantum.insert %0[ 0], %out : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg
    return
}

// -----

// Rule depends on helper functions

// CHECK: @helper_func(%arg0: !quantum.bit) -> !quantum.bit
func.func private @helper_func(%arg0: !quantum.bit) -> !quantum.bit {
    %out = quantum.custom "Hadamard"() %arg0 : !quantum.bit
    return %out : !quantum.bit
}

// CHECK: attributes {qnode, resources
func.func @rule(%arg0: !quantum.bit) -> !quantum.bit attributes {qnode, target_gate="rule"} {
    %r1 = func.call @helper_func(%arg0) : (!quantum.bit) -> !quantum.bit
    %r2 = func.call @helper_func(%r1) : (!quantum.bit) -> !quantum.bit
    return %r2 : !quantum.bit
}

// -----

// Rules with for loop (static)

// CHECK: resources = {measurements = {}, num_alloc_qubits = 0 : i64, num_arg_qubits = 1 : i64, num_qubits = 1 : i64, operations = {"Hadamard(1,0)" = 5 : i64}}
func.func @rule_with_loop(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate="gate"} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %iter = %c0 to %c5 step %c1 iter_args(%arg1 = %arg0) -> (!quantum.bit) {
        %out = quantum.custom "Hadamard"() %arg1 : !quantum.bit
        scf.yield %out : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// Rules with branching (take max per op)

// CHECK: resources = {measurements = {}, num_alloc_qubits = 0 : i64, num_arg_qubits = 1 : i64, num_qubits = 1 : i64, operations = {"Hadamard(1,0)" = 3 : i64, "PauliX(1,0)" = 2 : i64}}
func.func @rule_with_branching(%arg0: !quantum.bit, %cond: i1) -> !quantum.bit attributes {target_gate="gate"} {
    %q = scf.if %cond -> !quantum.bit {
        // True branch: 2 Hadamard, 1 PauliX
        %t1 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        %t2 = quantum.custom "Hadamard"() %t1 : !quantum.bit
        %t3 = quantum.custom "PauliX"() %t2 : !quantum.bit
        scf.yield %t3 : !quantum.bit
    } else {
        // False branch: 3 Hadamard, 2 PauliX -> max(2,3)=3 Hadamard, max(1,2)=2 PauliX
        %f1 = quantum.custom "Hadamard"() %arg0 : !quantum.bit
        %f2 = quantum.custom "Hadamard"() %f1 : !quantum.bit
        %f3 = quantum.custom "Hadamard"() %f2 : !quantum.bit
        %f4 = quantum.custom "PauliX"() %f3 : !quantum.bit
        %f5 = quantum.custom "PauliX"() %f4 : !quantum.bit
        scf.yield %f5 : !quantum.bit
    }

    return %q : !quantum.bit
}

// -----

// Rules with static for loops

// CHECK:  operations = {"PauliX(1,0)" = 15 : i64}}
func.func @rule_with_nested_loop(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate="gate"} {
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %q = scf.for %i = %c0 to %c3 step %c1 iter_args(%a = %arg0) -> (!quantum.bit) {
        %q2 = scf.for %j = %c0 to %c5 step %c1 iter_args(%b = %a) -> (!quantum.bit) {
            %out = quantum.custom "PauliX"() %b : !quantum.bit
            scf.yield %out : !quantum.bit
        }
        scf.yield %q2 : !quantum.bit
    }

    return %q : !quantum.bit
}


// -----

// Rules with parametric ops
// CHECK:  operations = {"Adjoint(Rot)(4,3)" = 1 : i64, "Rot(4,3)" = 1 : i64}}
func.func @rule_with_parametric_ops(%arg0: !quantum.bit) -> !quantum.bit attributes {target_gate="gate"} {
    %cst_0 = arith.constant 0.1 : f64
    %cst_1 = arith.constant 0.2 : f64
    %cst_2 = arith.constant 0.3 : f64

    %true = arith.constant true
    %false = arith.constant false

    %reg = quantum.alloc(4) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %reg[2] : !quantum.reg -> !quantum.bit
    %q3 = quantum.extract %reg[3] : !quantum.reg -> !quantum.bit

    %res:4 = quantum.custom "Rot"(%cst_0, %cst_1, %cst_2) %q0, %q1
             ctrls(%q2, %q3)
             ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit

    %res_adj:4 = quantum.custom "Rot"(%cst_0, %cst_1, %cst_2) %res#0, %res#1 adj
                 ctrls(%res#2, %res#3)
                 ctrlvals(%true, %false) : !quantum.bit, !quantum.bit ctrls !quantum.bit, !quantum.bit

    return %res_adj#0 : !quantum.bit
}
