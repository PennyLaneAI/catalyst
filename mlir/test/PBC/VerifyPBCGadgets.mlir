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

// RUN: quantum-opt %s --verify-pbc-gadgets --split-input-file --verify-diagnostics | FileCheck %s
// RUN: not quantum-opt %s --verify-pbc-gadgets="strict=true" --split-input-file 2>&1 | FileCheck %s --check-prefix=STRICT

// The pauli-corrected pi/8 gadget for a negative angle.

// CHECK-LABEL: @pauli_corrected
func.func @pauli_corrected(%q: !quantum.bit) -> !quantum.bit {
    // CHECK: pbc.gadget ["Z"] theta(-0.78539816339744828) verdict(verified)
    %0 = pbc.gadget ["Z"] theta(-0.78539816339744828) (%q) : !quantum.bit {
    ^bb0(%arg1: !quantum.bit):
        %1 = pbc.fabricate magic : !quantum.bit
        %m0, %o:2 = pbc.ppm ["Z", "Z"](-) %arg1, %1 : i1, !quantum.bit, !quantum.bit
        %m1, %a = pbc.select.ppm (%m0 ? ["Y"] : ["X"]) %o#1 : i1, !quantum.bit
        %2 = pbc.ppr ["Z"](2) %o#0 cond(%m1) : !quantum.bit
        quantum.dealloc_qb %a : !quantum.bit
        pbc.yield %2 : !quantum.bit
    }
    return %0 : !quantum.bit
}

// -----

// The pi/4 gadget on two qubits.

// CHECK-LABEL: @pi_over_four
func.func @pi_over_four(%q0: !quantum.bit, %q1: !quantum.bit) -> (!quantum.bit, !quantum.bit) {
    // CHECK: verdict(verified)
    %0:2 = pbc.gadget ["X", "Y"] theta(1.5707963267948966) (%q0, %q1) : !quantum.bit, !quantum.bit {
    ^bb0(%a: !quantum.bit, %b: !quantum.bit):
        %aux = quantum.alloc_qb : !quantum.bit
        %m0, %o:3 = pbc.ppm ["X", "Y", "Y"](-) %a, %b, %aux : i1, !quantum.bit, !quantum.bit, !quantum.bit
        %m1, %x = pbc.ppm ["X"] %o#2 : i1, !quantum.bit
        %c = arith.xori %m0, %m1 : i1
        %r:2 = pbc.ppr ["X", "Y"](2) %o#0, %o#1 cond(%c) : !quantum.bit, !quantum.bit
        quantum.dealloc_qb %x : !quantum.bit
        pbc.yield %r#0, %r#1 : !quantum.bit, !quantum.bit
    }
    return %0#0, %0#1 : !quantum.bit, !quantum.bit
}

// -----

// Catalyst's clifford-corrected pi/8 for a negative angle corrects with +pi/4, so one branch is
// off by a Pauli.

// CHECK-LABEL: @wrong_correction
// STRICT: error: gadget does not implement its source rotation
func.func @wrong_correction(%q: !quantum.bit) -> !quantum.bit {
    // CHECK: verdict(mismatch)
    // expected-warning@+1 {{gadget does not implement its source rotation: the branch with outcomes '1}}
    %0 = pbc.gadget ["Z"] theta(-0.78539816339744828) (%q) : !quantum.bit {
    ^bb0(%arg1: !quantum.bit):
        %1 = pbc.fabricate magic_conj : !quantum.bit
        %m0, %o:2 = pbc.ppm ["Z", "Z"] %arg1, %1 : i1, !quantum.bit, !quantum.bit
        %2 = pbc.ppr ["Z"](4) %o#0 cond(%m0) : !quantum.bit
        %m1, %a = pbc.ppm ["X"] %o#1 : i1, !quantum.bit
        %3 = pbc.ppr ["Z"](2) %2 cond(%m1) : !quantum.bit
        quantum.dealloc_qb %a : !quantum.bit
        pbc.yield %3 : !quantum.bit
    }
    return %0 : !quantum.bit
}

// -----

// An auxiliary qubit released while entangled with another.

// CHECK-LABEL: @entangled_release
func.func @entangled_release(%q: !quantum.bit) -> !quantum.bit {
    // CHECK: verdict(mismatch)
    // expected-warning@+1 {{entangled-release}}
    %0 = pbc.gadget ["Z"] theta(3.1415926535897931) (%q) : !quantum.bit {
    ^bb0(%arg1: !quantum.bit):
        %a = pbc.fabricate plus_i : !quantum.bit
        %b = quantum.alloc_qb : !quantum.bit
        %m, %o:2 = pbc.ppm ["X", "X"] %a, %b : i1, !quantum.bit, !quantum.bit
        quantum.dealloc_qb %o#0 : !quantum.bit
        quantum.dealloc_qb %o#1 : !quantum.bit
        %r = pbc.ppr ["Z"](2) %arg1 : !quantum.bit
        pbc.yield %r : !quantum.bit
    }
    return %0 : !quantum.bit
}

// -----

// A condition from outside the gadget cannot be followed.

// CHECK-LABEL: @outside_condition
func.func @outside_condition(%q: !quantum.bit, %c: i1) -> !quantum.bit {
    // CHECK: verdict(unchecked)
    // expected-remark@+1 {{gadget not checked: a condition comes from outside the gadget}}
    %0 = pbc.gadget ["Z"] theta(3.1415926535897931) (%q) : !quantum.bit {
    ^bb0(%arg1: !quantum.bit):
        %r = pbc.ppr ["Z"](2) %arg1 cond(%c) : !quantum.bit
        pbc.yield %r : !quantum.bit
    }
    return %0 : !quantum.bit
}
