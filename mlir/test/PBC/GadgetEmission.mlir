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

// RUN: quantum-opt %s --pass-pipeline="builtin.module(ppr-to-ppm{emit-gadgets=true},verify-pbc-gadgets)" | FileCheck %s
// RUN: quantum-opt %s --pass-pipeline="builtin.module(ppr-to-ppm{decompose-method=clifford-corrected emit-gadgets=true})" | FileCheck %s --check-prefix=NESTED
// RUN: quantum-opt %s --ppr-to-ppm | FileCheck %s --check-prefix=DEFAULT

// Each decomposed rotation becomes one gadget that records it, and pbc.ppr P(k) is
// exp(-i theta/2 P) with theta = 2 pi / k.

// DEFAULT-NOT: pbc.gadget

// CHECK-LABEL: @rotations
func.func @rotations(%q0: !quantum.bit, %q1: !quantum.bit, %c: i1) -> (!quantum.bit, !quantum.bit) {
    // CHECK: pbc.gadget ["X", "Z"] theta(0.78539816339744828) verdict(verified)
    %a:2 = pbc.ppr ["X", "Z"](8) %q0, %q1 : !quantum.bit, !quantum.bit
    // CHECK: pbc.gadget ["Y"] theta(-1.5707963267948966) verdict(verified)
    %b = pbc.ppr ["Y"](-4) %a#0 : !quantum.bit
    // A conditional pi/4 is a gadget inside the condition.
    // CHECK: scf.if
    // CHECK-NEXT: pbc.gadget ["Z"] theta(1.5707963267948966) verdict(verified)
    %d = pbc.ppr ["Z"](4) %a#1 cond(%c) : !quantum.bit
    return %b, %d : !quantum.bit, !quantum.bit
}

// The clifford-corrected pi/8 gadget holds a conditional pi/4, decomposed in the same gadget.

// NESTED-LABEL: @rotations
// NESTED: pbc.gadget ["X", "Z"] theta(0.78539816339744828)
// NESTED-NOT: pbc.gadget ["X", "Z"] theta(1.5707963267948966)
// NESTED: pbc.yield
