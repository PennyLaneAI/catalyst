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

// The report displays the QNode, whose captured constants live in its wrapper.
// CHECK-LABEL: "captured_qnode": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @jit_circuit() attributes {llvm.emit_c_interface} {
    %wire = arith.constant 1 : i64
    func.call @captured_qnode(%wire) : (i64) -> ()
    return
}
func.func @captured_qnode(%wire: i64) attributes {quantum.node} {
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %r = quantum.insert %reg[0], %a : !quantum.reg, !quantum.bit
    %q1 = quantum.extract %r[%wire] : !quantum.reg -> !quantum.bit
    %b = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    return
}

// -----

// Capture uses a launch operation outside the QNode's nested symbol table.
// CHECK-LABEL: "captured_kernel": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 1
func.func @jit_circuit() attributes {llvm.emit_c_interface} {
    %wire = stablehlo.constant dense<1> : tensor<i64>
    catalyst.launch_kernel @kernel_module::@captured_kernel(%wire) : (tensor<i64>) -> ()
    return
}
module @kernel_module {
    func.func @captured_kernel(%wire: tensor<i64>) attributes {quantum.node, quantum.kernel_entry_point} {
        %index = tensor.extract %wire[] : tensor<i64>
        %reg = quantum.alloc(2) : !quantum.reg
        %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
        %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
        %r = quantum.insert %reg[0], %a : !quantum.reg, !quantum.bit
        %q1 = quantum.extract %r[%index] : !quantum.reg -> !quantum.bit
        %b = pbc.ppr ["Z"](8) %q1 : !quantum.bit
        return
    }
}

// -----

// Shared captures may be used when all visible calls agree.
// CHECK-LABEL: "captured_qnode": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 2
func.func @jit_circuit() attributes {llvm.emit_c_interface} {
    %wire = arith.constant 0 : i64
    func.call @captured_qnode(%wire) : (i64) -> ()
    func.call @captured_qnode(%wire) : (i64) -> ()
    return
}
func.func @captured_qnode(%wire: i64) attributes {quantum.node} {
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %r = quantum.insert %reg[0], %a : !quantum.reg, !quantum.bit
    %q1 = quantum.extract %r[%wire] : !quantum.reg -> !quantum.bit
    %b = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    return
}

// -----

// Different call-site constants do not certify a single QNode depth.
// CHECK-LABEL: "captured_qnode": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "qnode":
func.func @jit_circuit() attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    func.call @captured_qnode(%zero) : (i64) -> ()
    func.call @captured_qnode(%one) : (i64) -> ()
    return
}
func.func @captured_qnode(%wire: i64) attributes {quantum.node} {
    %reg = quantum.alloc(2) : !quantum.reg
    %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %r = quantum.insert %reg[0], %a : !quantum.reg, !quantum.bit
    %q1 = quantum.extract %r[%wire] : !quantum.reg -> !quantum.bit
    %b = pbc.ppr ["Z"](8) %q1 : !quantum.bit
    return
}

// -----

// A runtime input cannot be replaced with a constant from another call.
// CHECK-LABEL: "captured_qnode": {
// CHECK: "qubit_disjoint_depth_status": "unavailable"
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "qnode":
func.func @jit_circuit(%wire: i64) attributes {llvm.emit_c_interface} {
    %one = arith.constant 1 : i64
    func.call @captured_qnode(%one) : (i64) -> ()
    func.call @captured_qnode(%wire) : (i64) -> ()
    return
}
func.func @captured_qnode(%wire: i64) attributes {quantum.node} {
    %reg = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %reg[%wire] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    return
}
