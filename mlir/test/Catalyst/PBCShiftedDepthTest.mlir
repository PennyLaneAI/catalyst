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

// Every output depends on all four inputs after one forward/backward sweep.
// The resulting clocks differ only by a shift. Reusing their substitution must
// retain those shifts and the three layers preceding the loop.
// CHECK-LABEL: "shifted_dense_outputs": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 12291
func.func @shifted_dense_outputs(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count = arith.constant 2048 : index
    %pre0 = pbc.ppr ["X"](8) %q0 : !quantum.bit
    %pre1 = pbc.ppr ["X"](8) %pre0 : !quantum.bit
    %pre2 = pbc.ppr ["X"](8) %pre1 : !quantum.bit
    %out:4 = scf.for %i = %zero to %count step %one
        iter_args(%a = %pre2, %b = %q1, %c = %q2, %d = %q3)
        -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
        %g0:2 = pbc.ppr ["X", "X"](8) %a, %b : !quantum.bit, !quantum.bit
        %g1:2 = pbc.ppr ["X", "X"](8) %g0#1, %c : !quantum.bit, !quantum.bit
        %g2:2 = pbc.ppr ["X", "X"](8) %g1#1, %d : !quantum.bit, !quantum.bit
        %g3:2 = pbc.ppr ["X", "X"](8) %g2#0, %g2#1 : !quantum.bit, !quantum.bit
        %g4:2 = pbc.ppr ["X", "X"](8) %g1#0, %g3#0 : !quantum.bit, !quantum.bit
        %g5:2 = pbc.ppr ["X", "X"](8) %g0#0, %g4#0 : !quantum.bit, !quantum.bit
        scf.yield %g5#0, %g5#1, %g4#1, %g3#1 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    }
    return
}

// -----

// These dense clocks share their input IDs but have different relative offsets.
// The same footprint is insufficient to reuse a substitution.
// Every iteration adds five layers to the longest path.
// CHECK-LABEL: "different_relative_offsets": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 10240
func.func @different_relative_offsets(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count = arith.constant 2048 : index
    %out:4 = scf.for %i = %zero to %count step %one
        iter_args(%a = %q0, %b = %q1, %c = %q2, %d = %q3)
        -> (!quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit) {
        %left:2 = pbc.ppr ["X", "X"](8) %a, %b : !quantum.bit, !quantum.bit
        %right:2 = pbc.ppr ["X", "X"](8) %c, %d : !quantum.bit, !quantum.bit
        %slow0 = pbc.ppr ["X"](8) %left#0 : !quantum.bit
        %slow1 = pbc.ppr ["X"](8) %slow0 : !quantum.bit
        %slow2 = pbc.ppr ["X"](8) %slow1 : !quantum.bit
        %middle:2 = pbc.ppr ["X", "X"](8) %left#1, %right#0 : !quantum.bit, !quantum.bit
        %outer:2 = pbc.ppr ["X", "X"](8) %slow2, %right#1 : !quantum.bit, !quantum.bit
        scf.yield %outer#0, %middle#0, %middle#1, %outer#1 : !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    }
    return
}

// -----

// An internal ancilla can change the constant without changing relative input
// offsets. The constant therefore participates in normalized-clock equality.
// The first iteration takes five layers; later iterations add two each.
// CHECK-LABEL: "different_normalized_constants": {
// CHECK: "qubit_disjoint_depth_status": "exact"
// CHECK: "qubit_disjoint_depth_total": 4099
func.func @different_normalized_constants(%q0: !quantum.bit, %q1: !quantum.bit, %q2: !quantum.bit) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %count = arith.constant 2048 : index
    %out:3 = scf.for %i = %zero to %count step %one
        iter_args(%a = %q0, %b = %q1, %c = %q2) -> (!quantum.bit, !quantum.bit, !quantum.bit) {
        %pair:3 = pbc.ppr ["X", "X", "X"](8) %a, %b, %c : !quantum.bit, !quantum.bit, !quantum.bit
        %ancilla = quantum.alloc_qb : !quantum.bit
        %slow0 = pbc.ppr ["X"](8) %ancilla : !quantum.bit
        %slow1 = pbc.ppr ["X"](8) %slow0 : !quantum.bit
        %slow2 = pbc.ppr ["X"](8) %slow1 : !quantum.bit
        %slow3 = pbc.ppr ["X"](8) %slow2 : !quantum.bit
        %joined:2 = pbc.ppr ["X", "X"](8) %pair#0, %slow3 : !quantum.bit, !quantum.bit
        quantum.dealloc_qb %joined#1 : !quantum.bit
        scf.yield %joined#0, %pair#1, %pair#2 : !quantum.bit, !quantum.bit, !quantum.bit
    }
    return
}
