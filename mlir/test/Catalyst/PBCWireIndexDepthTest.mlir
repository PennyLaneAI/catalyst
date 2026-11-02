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

// RUN: quantum-opt -resource-analysis=output-json --split-input-file -verify-diagnostics %s -o /dev/null | FileCheck %s

// Propagate the selected element, not just the first element of a wire tensor.
// CHECK-LABEL: "constant_disjoint_wires": {
// CHECK: "qubit_disjoint_depth_total": 1
func.func @constant_disjoint_wires() {
    %a = arith.constant dense<[1, 0]> : tensor<2xi64>
    %b = arith.constant dense<[0, 1]> : tensor<2xi64>
    %reg = quantum.alloc(2) : !quantum.reg
    %r0 = func.call @rotate_second(%a, %reg) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    %r1 = func.call @rotate_second(%b, %r0) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    quantum.dealloc %r1 : !quantum.reg
    return
}

func.func private @rotate_second(%wires: tensor<2xi64>, %reg: !quantum.reg) -> !quantum.reg {
    %slice = stablehlo.slice %wires [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %scalar = stablehlo.reshape %slice : (tensor<1xi64>) -> tensor<i64>
    %index = tensor.extract %scalar[] : tensor<i64>
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %r = pbc.ppr ["Z"](8) %q : !quantum.bit
    %out = quantum.insert %reg[%index], %r : !quantum.reg, !quantum.bit
    return %out : !quantum.reg
}

// -----

// Boolean true is the unsigned one-bit value 1, not a signed one-bit integer.
// Exercise both dense constants and tensor construction before converting wires.
// CHECK-LABEL: "boolean_wire_construction": {
// CHECK: "qubit_disjoint_depth_total": 1
func.func @boolean_wire_construction() {
    %a = arith.constant dense<[false, true]> : tensor<2xi1>
    %false = arith.constant false
    %true = arith.constant true
    %b = tensor.from_elements %true, %false : tensor<2xi1>
    %reg = quantum.alloc(2) : !quantum.reg
    %r0 = func.call @rotate_boolean_second(%a, %reg) : (tensor<2xi1>, !quantum.reg) -> !quantum.reg
    %r1 = func.call @rotate_boolean_second(%b, %r0) : (tensor<2xi1>, !quantum.reg) -> !quantum.reg
    quantum.dealloc %r1 : !quantum.reg
    return
}

func.func private @rotate_boolean_second(%wires: tensor<2xi1>, %reg: !quantum.reg) -> !quantum.reg {
    %integer_wires = stablehlo.convert %wires : (tensor<2xi1>) -> tensor<2xi64>
    %slice = stablehlo.slice %integer_wires [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %scalar = stablehlo.reshape %slice : (tensor<1xi64>) -> tensor<i64>
    %index = tensor.extract %scalar[] : tensor<i64>
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %r = pbc.ppr ["Z"](8) %q : !quantum.bit
    %out = quantum.insert %reg[%index], %r : !quantum.reg, !quantum.bit
    return %out : !quantum.reg
}

// -----

// Propagate the selected element, not just the first element of a wire tensor.
// CHECK-LABEL: "constant_shared_wire": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @constant_shared_wire() {
    %a = arith.constant dense<[1, 0]> : tensor<2xi64>
    %b = arith.constant dense<[1, 0]> : tensor<2xi64>
    %reg = quantum.alloc(2) : !quantum.reg
    %r0 = func.call @rotate_second(%a, %reg) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    %r1 = func.call @rotate_second(%b, %r0) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    quantum.dealloc %r1 : !quantum.reg
    return
}

func.func private @rotate_second(%wires: tensor<2xi64>, %reg: !quantum.reg) -> !quantum.reg {
    %slice = stablehlo.slice %wires [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %scalar = stablehlo.reshape %slice : (tensor<1xi64>) -> tensor<i64>
    %index = tensor.extract %scalar[] : tensor<i64>
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %r = pbc.ppr ["Z"](8) %q : !quantum.bit
    %out = quantum.insert %reg[%index], %r : !quantum.reg, !quantum.bit
    return %out : !quantum.reg
}

// -----

// Cached classical outputs must keep the constant from the corresponding call.
// CHECK-LABEL: "constant_returned_wires": {
// CHECK: "qubit_disjoint_depth_total": 1
func.func @constant_returned_wires() {
    %a = arith.constant dense<[1, 0]> : tensor<2xi64>
    %b = arith.constant dense<[0, 1]> : tensor<2xi64>
    %wa = func.call @identity(%a) : (tensor<2xi64>) -> tensor<2xi64>
    %wb = func.call @identity(%b) : (tensor<2xi64>) -> tensor<2xi64>
    %again = func.call @identity(%b) : (tensor<2xi64>) -> tensor<2xi64>
    %reg = quantum.alloc(2) : !quantum.reg
    %r0 = func.call @rotate_second(%wa, %reg) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    %r1 = func.call @rotate_second(%again, %r0) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    return
}
func.func private @identity(%value: tensor<2xi64>) -> tensor<2xi64> {
    return %value : tensor<2xi64>
}

func.func private @rotate_second(%wires: tensor<2xi64>, %reg: !quantum.reg) -> !quantum.reg {
    %slice = stablehlo.slice %wires [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %scalar = stablehlo.reshape %slice : (tensor<1xi64>) -> tensor<i64>
    %index = tensor.extract %scalar[] : tensor<i64>
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %r = pbc.ppr ["Z"](8) %q : !quantum.bit
    %out = quantum.insert %reg[%index], %r : !quantum.reg, !quantum.bit
    return %out : !quantum.reg
}

// -----

// A loop-carried tensor can change: the first iteration's constant is not invariant.
// CHECK-LABEL: "changing_loop_wires": {
// CHECK: "qubit_disjoint_depth_total": 2
func.func @changing_loop_wires() {
    %a = arith.constant dense<[1, 0]> : tensor<2xi64>
    %b = arith.constant dense<[0, 1]> : tensor<2xi64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %reg = quantum.alloc(2) : !quantum.reg
    %out:2 = scf.for %i = %c0 to %c2 step %c1 iter_args(%r = %reg, %w = %a) -> (!quantum.reg, tensor<2xi64>) {
        %updated = func.call @rotate_second(%w, %r) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
        scf.yield %updated, %b : !quantum.reg, tensor<2xi64>
    }
    %q = quantum.extract %out#0[1] : !quantum.reg -> !quantum.bit
    %r = pbc.ppr ["X"](8) %q : !quantum.bit
    return
}

func.func private @rotate_second(%wires: tensor<2xi64>, %reg: !quantum.reg) -> !quantum.reg {
    %slice = stablehlo.slice %wires [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %scalar = stablehlo.reshape %slice : (tensor<1xi64>) -> tensor<i64>
    %index = tensor.extract %scalar[] : tensor<i64>
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %r = pbc.ppr ["Z"](8) %q : !quantum.bit
    %out = quantum.insert %reg[%index], %r : !quantum.reg, !quantum.bit
    return %out : !quantum.reg
}

// -----

// Distinct branch constants cannot be propagated as one known wire tensor.
// CHECK-LABEL: "changing_branch_wires": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @changing_branch_wires(%condition: i1) {
    %a = arith.constant dense<[1, 0]> : tensor<2xi64>
    %b = arith.constant dense<[0, 1]> : tensor<2xi64>
    %reg = quantum.alloc(2) : !quantum.reg
    %r0 = func.call @rotate_second(%b, %reg) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    %wires = scf.if %condition -> tensor<2xi64> {
        scf.yield %a : tensor<2xi64>
    } else {
        scf.yield %b : tensor<2xi64>
    }
    %r1 = func.call @rotate_second(%wires, %r0) : (tensor<2xi64>, !quantum.reg) -> !quantum.reg
    return
}

func.func private @rotate_second(%wires: tensor<2xi64>, %reg: !quantum.reg) -> !quantum.reg {
    %slice = stablehlo.slice %wires [1:2] : (tensor<2xi64>) -> tensor<1xi64>
    %scalar = stablehlo.reshape %slice : (tensor<1xi64>) -> tensor<i64>
    %index = tensor.extract %scalar[] : tensor<i64>
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %r = pbc.ppr ["Z"](8) %q : !quantum.bit
    %out = quantum.insert %reg[%index], %r : !quantum.reg, !quantum.bit
    return %out : !quantum.reg
}
