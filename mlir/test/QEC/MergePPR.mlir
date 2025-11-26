// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: quantum-opt --merge-ppr --split-input-file -verify-diagnostics %s | FileCheck %s
// RUN: quantum-opt --merge-ppr="max-pauli-size=3" --split-input-file -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK-MPS

//////////////////////////////////////////// MERGE TESTS ///////////////////////////////////////////
////////////////////////////////////// Single gate merge tests /////////////////////////////////////

// CHECK-LABEL: merge_X_pi_2
func.func public @merge_X_pi_2(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["X"](2)
    %0 = qec.ppr ["X"](2) %q1: !quantum.bit
    %1 = qec.ppr ["X"](2) %0: !quantum.bit
    func.return
}

// CHECK-LABEL: merge_Y_pi_4
func.func public @merge_Y_pi_4(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["Y"](4)
    // CHECK: qec.ppr ["Y"](2)
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %1 = qec.ppr ["Y"](4) %0: !quantum.bit
    func.return
}

// CHECK-LABEL: merge_Z_pi_8
func.func public @merge_Z_pi_8(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["Z"](8)
    // CHECK: qec.ppr ["Z"](4)
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
    %1 = qec.ppr ["Z"](8) %0: !quantum.bit
    func.return
}

// CHECK-LABEL: merge_X_minus_pi_8
func.func public @merge_X_minus_pi_8(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["X"](-8)
    // CHECK: qec.ppr ["X"](-4)
    %0 = qec.ppr ["X"](-8) %q1: !quantum.bit
    %1 = qec.ppr ["X"](-8) %0: !quantum.bit
    func.return
}

/////////////////////////////////// Single gate multi-merge tests //////////////////////////////////

// CHECK-LABEL: merge_multi_Z
func.func public @merge_multi_Z(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["Z"](8)
    // CHECK-NOT: qec.ppr ["Z"](4)
    // CHECK: qec.ppr ["Z"](2)
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
    %1 = qec.ppr ["Z"](8) %0: !quantum.bit
    %2 = qec.ppr ["Z"](4) %1: !quantum.bit
    func.return
}

// CHECK-LABEL: merge_multi_X
func.func public @merge_multi_X(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["X"](8)
    // CHECK: qec.ppr ["X"](2)
    %0 = qec.ppr ["X"](8) %q1: !quantum.bit
    %1 = qec.ppr ["X"](8) %0: !quantum.bit
    %2 = qec.ppr ["X"](8) %1: !quantum.bit
    %3 = qec.ppr ["X"](8) %2: !quantum.bit
    func.return
}

// CHECK-LABEL: merge_multi_Y
func.func public @merge_multi_Y(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %1 = qec.ppr ["Y"](4) %0: !quantum.bit
    %2 = qec.ppr ["Y"](8) %1: !quantum.bit
    %3 = qec.ppr ["Y"](8) %2: !quantum.bit
    %4 = qec.ppr ["Y"](8) %3: !quantum.bit
    %5 = qec.ppr ["Y"](8) %4: !quantum.bit
    func.return
}

// CHECK-LABEL: dont_merge
func.func public @dont_merge(%q1: !quantum.bit) {
    // CHECK: qec.ppr ["Y"](4)
    // CHECK: qec.ppr ["X"](4)
    // CHECK: qec.ppr ["Z"](8)
    // CHECK: qec.ppr ["X"](8)
    // CHECK: qec.ppr ["Y"](2)
    // CHECK: qec.ppr ["Z"](2)
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %1 = qec.ppr ["X"](4) %0: !quantum.bit
    %2 = qec.ppr ["Z"](8) %1: !quantum.bit
    %3 = qec.ppr ["X"](8) %2: !quantum.bit
    %4 = qec.ppr ["Y"](2) %3: !quantum.bit
    %5 = qec.ppr ["Z"](2) %4: !quantum.bit
    func.return
}

// CHECK-LABEL: merge_correct_references
func.func public @merge_correct_references(%q1: !quantum.bit) {
    // CHECK: %[[in:[a-zA-Z0-9_]+]] = qec.ppr ["Y"](4)
    // CHECK: %[[merge_out:[a-zA-Z0-9_]+]] = qec.ppr ["X"](2) %[[in]]
    // CHECK: qec.ppr ["Z"](4) %[[merge_out]]
    %0 = qec.ppr ["Y"](4) %q1: !quantum.bit
    %1 = qec.ppr ["X"](4) %0: !quantum.bit
    %2 = qec.ppr ["X"](4) %1: !quantum.bit
    %3 = qec.ppr ["Z"](4) %2: !quantum.bit
    func.return
}

///////////////////////////////////////// Multi-gate tests /////////////////////////////////////////

// CHECK-LABEL: merge_multi_ZX
func.func public @merge_multi_ZX(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0:2 = qec.ppr ["Z", "X"](8) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["Z", "X"](8) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["Z", "X"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Z", "X"](4) %2#0, %2#1: !quantum.bit, !quantum.bit
    %4:2 = qec.ppr ["Z", "X"](8) %3#0, %3#1: !quantum.bit, !quantum.bit
    %5:2 = qec.ppr ["Z", "X"](8) %4#0, %4#1: !quantum.bit, !quantum.bit
    func.return
}

// CHECK-LABEL: merge_multi_ZY
func.func public @merge_multi_ZY(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK: qec.ppr ["Z", "Y"](8)
    // CHECK: qec.ppr ["Z", "Y"](4)
    // CHECK: qec.ppr ["Z", "Y"](2)
    %0:2 = qec.ppr ["Z", "Y"](8) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["Z", "Y"](4) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["Z", "Y"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Z", "Y"](8) %2#0, %2#1: !quantum.bit, !quantum.bit
    %4:2 = qec.ppr ["Z", "Y"](8) %3#0, %3#1: !quantum.bit, !quantum.bit
    func.return
}

// CHECK-LABEL: merge_multi_XYZ
func.func public @merge_multi_XYZ(%q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit) {
    // CHECK: qec.ppr ["X", "Y", "Z"](4)
    // CHECK-NOT: qec.ppr ["X", "Y", "Z"](2)
    // CHECK-NOT: qec.ppr ["X", "Y", "Z"](8)
    %0:3 = qec.ppr ["X", "Y", "Z"](4) %q1, %q2, %q3: !quantum.bit, !quantum.bit, !quantum.bit
    %1:3 = qec.ppr ["X", "Y", "Z"](4) %0#0, %0#1, %0#2: !quantum.bit, !quantum.bit, !quantum.bit
    %2:3 = qec.ppr ["X", "Y", "Z"](8) %1#0, %1#1, %1#2: !quantum.bit, !quantum.bit, !quantum.bit
    %3:3 = qec.ppr ["X", "Y", "Z"](8) %2#0, %2#1, %2#2: !quantum.bit, !quantum.bit, !quantum.bit
    %4:3 = qec.ppr ["X", "Y", "Z"](2) %3#0, %3#1, %3#2: !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

/////////////////////////////////////////// CANCEL TESTS ///////////////////////////////////////////
///////////////////////////////////// Single gate cancel tests /////////////////////////////////////

// CHECK-LABEL: cancel_Z_pi_2
func.func public @cancel_Z_pi_2(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](2) %q1: !quantum.bit
    %1 = qec.ppr ["Z"](-2) %0: !quantum.bit
    func.return
}

// CHECK-LABEL: cancel_X_pi_4
func.func public @cancel_X_pi_4(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](4) %q1: !quantum.bit
    %1 = qec.ppr ["X"](-4) %0: !quantum.bit
    func.return
}

// CHECK-LABEL: cancel_Y_pi_8
func.func public @cancel_Y_pi_8(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](8) %q1: !quantum.bit
    %1 = qec.ppr ["Y"](-8) %0: !quantum.bit
    func.return
}

////////////////////////////////// Single gate multi-cancel tests //////////////////////////////////


// CHECK-LABEL: cancel_multi_Y
func.func public @cancel_multi_Y(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Y"](8) %q1: !quantum.bit
    %1 = qec.ppr ["Y"](-8) %0: !quantum.bit
    %2 = qec.ppr ["Y"](-8) %0: !quantum.bit
    %3 = qec.ppr ["Y"](8) %2: !quantum.bit
    func.return
}

// CHECK-LABEL: cancel_multi_X
func.func public @cancel_multi_X(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](8) %q1: !quantum.bit
    %1 = qec.ppr ["Y"](-4) %0: !quantum.bit
    %2 = qec.ppr ["Y"](4) %1: !quantum.bit
    %3 = qec.ppr ["X"](-8) %2: !quantum.bit
    func.return
}

// CHECK-LABEL: cancel_multi_Z
func.func public @cancel_multi_Z(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
    %1 = qec.ppr ["X"](-4) %0: !quantum.bit
    %2 = qec.ppr ["X"](4) %1: !quantum.bit
    %3 = qec.ppr ["Z"](-8) %2: !quantum.bit
    func.return
}

// CHECK-LABEL: dont_cancel
func.func public @dont_cancel(%q1: !quantum.bit) {
    // CHECK: qec.ppr ["Z"](8)
    // CHECK: qec.ppr ["X"](-4)
    // CHECK: qec.ppr ["Z"](-8)
    // CHECK: qec.ppr ["X"](4)
    %0 = qec.ppr ["Z"](8) %q1: !quantum.bit
    %1 = qec.ppr ["X"](-4) %0: !quantum.bit
    %2 = qec.ppr ["Z"](-8) %1: !quantum.bit
    %3 = qec.ppr ["X"](4) %2: !quantum.bit
    func.return
}

// CHECK-LABEL: cancel_correct_references
func.func public @cancel_correct_references(%q1: !quantum.bit) {
    // CHECK: %[[in:[a-zA-Z0-9_]+]] = qec.ppr ["Y"](2)
    // CHECK: qec.ppr ["Z"](4) %[[in]]
    %0 = qec.ppr ["Y"](2) %q1: !quantum.bit
    %1 = qec.ppr ["X"](8) %0: !quantum.bit
    %2 = qec.ppr ["X"](-8) %1: !quantum.bit
    %3 = qec.ppr ["Z"](4) %2: !quantum.bit
    func.return
}

/////////////////////////////////// Multi gate multi-cancel tests //////////////////////////////////

// CHECK-LABEL: cancel_multi_ZX
func.func public @cancel_multi_ZX(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0:2 = qec.ppr ["Z", "X"](8) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["Z", "X"](-4) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["Z", "X"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Z", "X"](-8) %2#0, %2#1: !quantum.bit, !quantum.bit
    func.return
}

// CHECK-LABEL: cancel_multi_XY
func.func public @cancel_multi_XY(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0:2 = qec.ppr ["X", "Y"](2) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["X", "Y"](-2) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["X", "Y"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["X", "Y"](-4) %2#0, %2#1: !quantum.bit, !quantum.bit
    func.return
}

// CHECK-LABEL: dont_cancel_multi
func.func public @dont_cancel_multi(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK: qec.ppr ["X", "Y"](2)
    // CHECK: qec.ppr ["Y", "X"](-2)
    // CHECK: qec.ppr ["Z", "Y"](4)
    // CHECK: qec.ppr ["Y", "Z"](-4)
    %0:2 = qec.ppr ["X", "Y"](2) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["Y", "X"](-2) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["Z", "Y"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Y", "Z"](-4) %2#0, %2#1: !quantum.bit, !quantum.bit
    func.return
}

///////////////////////////////////////// MERGE AND CANCEL /////////////////////////////////////////
//////////////////////////////////////////// single gate ///////////////////////////////////////////

// CHECK-LABEL: merge_and_cancel
func.func public @merge_and_cancel(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0 = qec.ppr ["X"](-4) %q1: !quantum.bit
    %1 = qec.ppr ["X"](-4) %0: !quantum.bit
    %2 = qec.ppr ["X"](2) %1: !quantum.bit
    func.return
}

// CHECK-LABEL: cancel_and_merge
func.func public @cancel_and_merge(%q1: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["X"]
    // CHECK: qec.ppr ["Z"](2)
    %0 = qec.ppr ["Z"](4) %q1: !quantum.bit
    %1 = qec.ppr ["X"](-2) %0: !quantum.bit
    %2 = qec.ppr ["X"](2) %1: !quantum.bit
    %3 = qec.ppr ["Z"](4) %2: !quantum.bit
    func.return
}

//////////////////////////////////////////// multi-gate ////////////////////////////////////////////

// CHECK-LABEL: multi_merge_and_cancel
func.func public @multi_merge_and_cancel(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK-NOT: qec.ppr
    %0:2 = qec.ppr ["Z", "X"](4) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["Z", "X"](-8) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["Z", "X"](-8) %1#0, %1#1: !quantum.bit, !quantum.bit
    func.return
}

// CHECK-LABEL: multi_cancel_and_merge
func.func public @multi_cancel_and_merge(%q1: !quantum.bit, %q2: !quantum.bit) {
    // CHECK-NOT: qec.ppr ["X", "Y"]
    // CHECK: qec.ppr ["Y", "Z"](4)
    %0:2 = qec.ppr ["Y", "Z"](8) %q1, %q2: !quantum.bit, !quantum.bit
    %1:2 = qec.ppr ["X", "Y"](-4) %0#0, %0#1: !quantum.bit, !quantum.bit
    %2:2 = qec.ppr ["X", "Y"](4) %1#0, %1#1: !quantum.bit, !quantum.bit
    %3:2 = qec.ppr ["Y", "Z"](8) %2#0, %2#1: !quantum.bit, !quantum.bit
    func.return
}

//////////////////////////////////////////////// MPS ///////////////////////////////////////////////

// CHECK-LABEL: max_pauli_size
func.func public @max_pauli_size(%q1: !quantum.bit, %q2: !quantum.bit, %q3: !quantum.bit, %q4: !quantum.bit) {
    // CHECK: qec.ppr ["Z", "X", "Y", "Z"](2)
    // CHECK-MPS: %[[in:[a-zA-Z0-9_]+]]:4 = qec.ppr ["Z", "X", "Y", "Z"](4) %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}
    // CHECK-MPS: qec.ppr ["Z", "X", "Y", "Z"](4) %[[in]]#0, %[[in]]#1, %[[in]]#2, %[[in]]#3
    // CHECK-MPS-NOT: qec.ppr ["Z", "X", "Y", "Z"](2)
    %0:4 = qec.ppr ["Z", "X", "Y", "Z"](4) %q1, %q2, %q3, %q4: !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    %1:4 = qec.ppr ["Z", "X", "Y", "Z"](4) %0#0, %0#1, %0#2, %0#3: !quantum.bit, !quantum.bit, !quantum.bit, !quantum.bit
    func.return
}

///////////////////////////////////////////// Other ////////////////////////////////////////////////

// CHECK-LABEL: mixed_operations
func.func public @mixed_operations(%q1: !quantum.bit) {
    // CHECK: qec.ppr ["Z"](4)
    // CHECK: quantum.custom "Hadamard"()
    // CHECK: qec.ppr ["Z"](4)
    %0 = qec.ppr ["Z"](4) %q1: !quantum.bit
    %1 = quantum.custom "Hadamard"() %0: !quantum.bit
    %2 = qec.ppr ["Z"](4) %1: !quantum.bit
    func.return
}
