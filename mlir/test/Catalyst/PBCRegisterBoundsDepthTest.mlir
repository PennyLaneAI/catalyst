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

// Automatic qubit management can grow the register beyond its initial size.
// A later dynamic access may alias the newly added qubit.
// CHECK-LABEL: "growing_register": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @growing_register(%index: i64) {
    quantum.device ["librtd_null_qubit.so", "NullQubit", "{}"] {auto_qubit_management}
    %reg = quantum.alloc(1) : !quantum.reg
    %q1 = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %a0 = pbc.ppr ["X"](8) %q1 : !quantum.bit
    %a1 = pbc.ppr ["Z"](8) %a0 : !quantum.bit
    %r1 = quantum.insert %reg[1], %a1 : !quantum.reg, !quantum.bit
    %qd = quantum.extract %r1[%index] : !quantum.reg -> !quantum.bit
    %b0 = pbc.ppr ["X"](8) %qd : !quantum.bit
    %b1 = pbc.ppr ["Z"](8) %b0 : !quantum.bit
    %out = quantum.insert %r1[%index], %b1 : !quantum.reg, !quantum.bit
    quantum.dealloc %out : !quantum.reg
    quantum.device_release
    return
}

// -----

// Without a bounded register, even an in-range dynamic access is unsupported.
// CHECK-LABEL: "automatic_register_management": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @automatic_register_management(%index: i64) {
    quantum.device ["librtd_null_qubit.so", "NullQubit", "{}"] {auto_qubit_management}
    %reg = quantum.alloc(2) : !quantum.reg
    %q = quantum.extract %reg[%index] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %out = quantum.insert %reg[%index], %a : !quantum.reg, !quantum.bit
    quantum.dealloc %out : !quantum.reg
    quantum.device_release
    return
}

// -----

// Invalid static slots are not treated as extra independent physical qubits.
// CHECK-LABEL: "out_of_range_extract": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @out_of_range_extract() {
    %reg = quantum.alloc(1) : !quantum.reg
    %q = quantum.extract %reg[1] : !quantum.reg -> !quantum.bit
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    quantum.dealloc %reg : !quantum.reg
    return
}

// -----

// Insertions obey the same bound, including when the source is a separate qubit.
// CHECK-LABEL: "out_of_range_insert": {
// CHECK-NOT: "qubit_disjoint_depth_total"
// CHECK: "num_qubits"
func.func @out_of_range_insert(%q: !quantum.bit) {
    %reg = quantum.alloc(1) : !quantum.reg
    %a = pbc.ppr ["X"](8) %q : !quantum.bit
    %out = quantum.insert %reg[1], %a : !quantum.reg, !quantum.bit
    quantum.dealloc %out : !quantum.reg
    return
}
