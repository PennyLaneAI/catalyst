// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// RUN: quantum-opt -resource-analysis=output-json --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s

// Adjoint lowering records wire indices in a classical LIFO list.
// CHECK-LABEL: "adjoint_wire_stack": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @adjoint_wire_stack() {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %stack = catalyst.list_init : <i64>
    catalyst.list_push %zero, %stack : <i64>
    catalyst.list_push %one, %stack : <i64>
    %reg = quantum.alloc(2) : !quantum.reg
    %index1 = catalyst.list_pop %stack : <i64>
    %wires1 = tensor.from_elements %index1 : tensor<1xi64>
    %reg1 = func.call @z_list_helper(%wires1, %reg) : (tensor<1xi64>, !quantum.reg) -> !quantum.reg
    %index0 = catalyst.list_pop %stack : <i64>
    %wires0 = tensor.from_elements %index0 : tensor<1xi64>
    %reg0 = func.call @z_list_helper(%wires0, %reg1) : (tensor<1xi64>, !quantum.reg) -> !quantum.reg
    catalyst.list_dealloc %stack : <i64>
    quantum.dealloc %reg0 : !quantum.reg
    return
}
func.func private @z_list_helper(%wires: tensor<1xi64>, %reg: !quantum.reg) -> !quantum.reg {
    %zero = arith.constant 0 : index
    %index = tensor.extract %wires[%zero] : tensor<1xi64>
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    %updated = quantum.insert %reg[%index], %out : !quantum.reg, !quantum.bit
    return %updated : !quantum.reg
}

// -----

// Each unrolled iteration resolves its current pop, rather than caching the first.
// CHECK-LABEL: "loop_wire_stack": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @loop_wire_stack() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %stack = catalyst.list_init : <index>
    catalyst.list_push %zero, %stack : <index>
    catalyst.list_push %one, %stack : <index>
    %reg = quantum.alloc(2) : !quantum.reg
    %updated = scf.for %i = %zero to %two step %one iter_args(%r = %reg) -> !quantum.reg {
        %position = catalyst.list_pop %stack : <index>
        %index = arith.index_cast %position : index to i64
        %q = quantum.extract %r[%index] : !quantum.reg -> !quantum.bit
        %out = pbc.ppr ["X"](8) %q : !quantum.bit
        %next = quantum.insert %r[%index], %out : !quantum.reg, !quantum.bit
        scf.yield %next : !quantum.reg
    }
    %q = quantum.extract %updated[0] : !quantum.reg -> !quantum.bit
    %last = pbc.ppr ["Z"](8) %q : !quantum.bit
    catalyst.list_dealloc %stack : <index>
    return
}

// -----

// A runtime wire value stays unknown after a push/pop round trip.
// CHECK-LABEL: "unknown_wire_stack": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "function_calls"
func.func @unknown_wire_stack(%index: i64) {
    %stack = catalyst.list_init : <i64>
    catalyst.list_push %index, %stack : <i64>
    %popped = catalyst.list_pop %stack : <i64>
    %reg = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %reg[%popped] : !quantum.reg -> !quantum.bit
    %out = pbc.ppr ["X"](8) %q : !quantum.bit
    catalyst.list_dealloc %stack : <i64>
    return
}

// -----

// A skipped classical region must not silently lose mutations of a captured list.
// CHECK-LABEL: "captured_list_while": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "function_calls"
func.func @captured_list_while() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %stack = catalyst.list_init : <index>
    catalyst.list_push %zero, %stack : <index>
    %count = scf.while (%i = %zero) : (index) -> index {
        %condition = arith.cmpi slt, %i, %one : index
        scf.condition(%condition) %i : index
    } do {
    ^bb0(%i: index):
        catalyst.list_push %one, %stack : <index>
        %next = arith.addi %i, %one : index
        scf.yield %next : index
    }
    %position = catalyst.list_pop %stack : <index>
    %index = arith.index_cast %position : index to i64
    %reg = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %b = pbc.ppr ["X"](8) %q0 : !quantum.bit
    return
}
