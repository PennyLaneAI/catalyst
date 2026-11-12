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

// RUN: quantum-opt --pbc-ref-collapse-if-ppr --split-input-file %s | FileCheck %s

// The PPR becomes conditional on the measurement result, ops without memory effects are hoisted,
// and the global phase is dropped.

// CHECK-LABEL: func.func @measurement_condition
// CHECK-SAME: ([[R:%.+]]: !qref.reg<2>, [[IDX_T:%.+]]: tensor<i64>)
func.func @measurement_condition(%r: !qref.reg<2>, %idx: tensor<i64>) {
    %phase = arith.constant 0.5 : f64
    %q0 = qref.get %r[0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[M:%.+]] = pbc.ref.ppm ["Z"]
    // CHECK-NOT: scf.if
    // CHECK: [[IDX:%.+]] = tensor.extract [[IDX_T]][]
    // CHECK: [[Q1:%.+]] = qref.get [[R]][[[IDX]]]
    // CHECK-NOT: qref.gphase
    // CHECK: pbc.ref.ppr ["X"](2) [[Q1]] cond([[M]])
    %m = pbc.ref.ppm ["Z"] %q0 : i1
    scf.if %m {
        %i = tensor.extract %idx[] : tensor<i64>
        %q1 = qref.get %r[%i] : !qref.reg<2>, i64 -> !qref.bit
        qref.gphase(%phase)
        pbc.ref.ppr ["X"](2) %q1
    }
    func.return
}

// -----

// The else branch is conditional on the negated condition, and conditions are and-ed.

// CHECK-LABEL: func.func @both_branches
// CHECK-SAME: ([[Q:%.+]]: !qref.bit, [[A:%.+]]: i1, [[B:%.+]]: i1)
func.func @both_branches(%q: !qref.bit, %a: i1, %b: i1) {
    // CHECK: [[AND:%.+]] = arith.andi [[B]], [[A]]
    // CHECK: pbc.ref.ppr ["X"](2) [[Q]] cond([[AND]])
    // CHECK: pbc.ref.ppr ["Y"](-2) [[Q]] cond([[A]])
    // CHECK: [[TRUE:%.+]] = arith.constant true
    // CHECK: [[NOT:%.+]] = arith.xori [[A]], [[TRUE]]
    // CHECK: pbc.ref.ppr ["Z"](2) [[Q]] cond([[NOT]])
    // CHECK-NOT: scf.if
    scf.if %a {
        pbc.ref.ppr ["X"](2) %q cond(%b)
        pbc.ref.ppr ["Y"](-2) %q
    } else {
        pbc.ref.ppr ["Z"](2) %q
    }
    func.return
}

// -----

// Nested branches collapse from the inside out.

// CHECK-LABEL: func.func @nested
// CHECK-SAME: ([[Q:%.+]]: !qref.bit, [[A:%.+]]: i1, [[B:%.+]]: i1)
func.func @nested(%q: !qref.bit, %a: i1, %b: i1) {
    // CHECK: [[AND:%.+]] = arith.andi [[B]], [[A]]
    // CHECK: pbc.ref.ppr ["X"](2) [[Q]] cond([[AND]])
    // CHECK: pbc.ref.ppr ["Z"](2) [[Q]] cond([[A]])
    // CHECK-NOT: scf.if
    scf.if %a {
        scf.if %b {
            pbc.ref.ppr ["X"](2) %q
        }
        pbc.ref.ppr ["Z"](2) %q
    }
    func.return
}

// -----

// Branches with other quantum operations, PPRs other than pi/2, controlled global phases or
// results are kept.

// CHECK-LABEL: func.func @kept
func.func @kept(%q: !qref.bit, %c: !qref.bit, %a: i1, %phase: f64) -> i1 {
    // CHECK: scf.if
    // CHECK: pbc.ref.ppm
    scf.if %a {
        %m = pbc.ref.ppm ["Z"] %q : i1
        pbc.ref.ppr ["X"](2) %q
    }
    // CHECK: scf.if
    // CHECK: pbc.ref.ppr ["X"](4)
    scf.if %a {
        pbc.ref.ppr ["X"](4) %q
    }
    // CHECK: scf.if
    // CHECK: qref.gphase
    scf.if %a {
        qref.gphase(%phase) ctrls(%c) ctrlvals(%a) : ctrls !qref.bit
        pbc.ref.ppr ["X"](2) %q
    }
    // CHECK: scf.if
    %r = scf.if %a -> i1 {
        pbc.ref.ppr ["X"](2) %q
        scf.yield %a : i1
    } else {
        scf.yield %a : i1
    }
    func.return %r : i1
}
