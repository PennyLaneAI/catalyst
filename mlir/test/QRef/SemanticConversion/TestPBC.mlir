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

// Test conversion to value semantics quantum dialect for PBC circuits.

// RUN: quantum-opt --convert-to-value-semantics --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_PPM_op
func.func @test_PPM_op(%angle: f64) -> (i1, i1, i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: [[qb:%.+]] = quantum.alloc_qb : !quantum.bit
    %a = qref.alloc(3) : !qref.reg<3>
    %q0 = qref.get %a[0] : !qref.reg<3> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<3> -> !qref.bit
    %qb = qref.alloc_qb : !qref.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[m0:%.+]], [[m0_out_qubit:%.+]] = pbc.ppm ["Z"] [[q0]] : i1, !quantum.bit
    %m0 = pbc.ref.ppm ["Z"] %q0 : i1

    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[m1:%.+]], [[m1_out_qubits:%.+]]:2 = pbc.ppm ["Z", "Y"] [[m0_out_qubit]], [[q1]] : i1, !quantum.bit, !quantum.bit
    %m1 = pbc.ref.ppm ["Z", "Y"] %q0, %q1 : i1

    // CHECK: [[m2:%.+]], [[m2_out_qubits:%.+]]:2 = pbc.ppm ["X", "Z"] [[m1_out_qubits]]#0, [[qb]] : i1, !quantum.bit, !quantum.bit
    %m2 = pbc.ref.ppm ["X", "Z"] %q0, %qb : i1
    // CHECK: [[insert1:%.+]] = quantum.insert [[qreg]][ 1], [[m1_out_qubits]]#1 : !quantum.reg, !quantum.bit
    // CHECK: [[insert2:%.+]] = quantum.insert [[insert1]][ 0], [[m2_out_qubits]]#0 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert2]] : !quantum.reg
    // CHECK: quantum.dealloc_qb [[m2_out_qubits]]#1 : !quantum.bit
    qref.dealloc %a : !qref.reg<3>
    qref.dealloc_qb %qb : !qref.bit

    // CHECK: return [[m0]], [[m1]], [[m2]] : i1, i1, i1
    return %m0, %m1, %m2 : i1, i1, i1
}

// -----

// CHECK-LABEL: test_PPR_op
func.func @test_PPR_op(%sw: i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: [[qb:%.+]] = quantum.alloc_qb : !quantum.bit
    %a = qref.alloc(3) : !qref.reg<3>
    %q0 = qref.get %a[0] : !qref.reg<3> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<3> -> !qref.bit
    %q2 = qref.get %a[2] : !qref.reg<3> -> !qref.bit
    %qb = qref.alloc_qb : !qref.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[q2:%.+]] = quantum.extract [[qreg]][ 2] : !quantum.reg -> !quantum.bit
    // CHECK: [[ppr0:%.+]]:3 = pbc.ppr ["X", "I", "Z"](4) [[q0]], [[q1]], [[q2]] : !quantum.bit, !quantum.bit, !quantum.bit
    pbc.ref.ppr ["X", "I", "Z"](4) %q0, %q1, %q2

    // CHECK: [[ppr1:%.+]]:2 = pbc.ppr ["Z", "Y"](-2) [[ppr0]]#0, [[qb]] cond(%arg0) : !quantum.bit, !quantum.bit
    pbc.ref.ppr ["Z", "Y"](-2) %q0, %qb cond(%sw)

    // CHECK: [[insert1:%.+]] = quantum.insert [[qreg]][ 1], [[ppr0]]#1 : !quantum.reg, !quantum.bit
    // CHECK: [[insert2:%.+]] = quantum.insert [[insert1]][ 2], [[ppr0]]#2 : !quantum.reg, !quantum.bit
    // CHECK: [[insert0:%.+]] = quantum.insert [[insert2]][ 0], [[ppr1]]#0 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert0]] : !quantum.reg
    // CHECK: quantum.dealloc_qb [[ppr1]]#1 : !quantum.bit
    qref.dealloc %a : !qref.reg<3>
    qref.dealloc_qb %qb : !qref.bit
    return
}

// -----

// CHECK-LABEL: test_select_PPM_op
func.func @test_select_PPM_op(%sw: i1) -> (i1, i1) attributes {quantum.node} {

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[m0:%.+]], [[m0_out:%.+]]:2 = pbc.select.ppm (%arg0 ? ["Z", "Y"] : ["X", "Z"](-)) [[q0]], [[q1]] : i1, !quantum.bit, !quantum.bit
    %m0 = pbc.ref.select.ppm (%sw ? ["Z", "Y"] : ["X", "Z"](-)) %q0, %q1 : i1

    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[m0_out]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[m1:%.+]], [[m1_out:%.+]] = pbc.select.ppm ([[m0]] ? ["X"] : ["Z"]) [[m0_out]]#1 : i1, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[m1_out]] : !quantum.reg, !quantum.bit
    %m1 = pbc.ref.select.ppm (%m0 ? ["X"] : ["Z"]) %q1 : i1

    // CHECK: quantum.dealloc [[insert1]] : !quantum.reg
    qref.dealloc %a : !qref.reg<2>

    // CHECK: return [[m0]], [[m1]] : i1, i1
    return %m0, %m1 : i1, i1
}

// -----

// CHECK-LABEL: test_fabricate_prepare_ops
func.func @test_fabricate_prepare_ops() -> (i1, i1) attributes {quantum.node} {

    // CHECK: [[magic:%.+]] = pbc.fabricate magic : !quantum.bit
    // CHECK: [[zeros:%.+]]:2 = pbc.prepare zero : !quantum.bit, !quantum.bit
    // CHECK-NOT: pbc.ref.fabricate
    // CHECK-NOT: pbc.ref.prepare
    %m = pbc.ref.fabricate magic : !qref.bit
    %z:2 = pbc.ref.prepare zero : !qref.bit, !qref.bit

    // CHECK: [[ppr:%.+]]:3 = pbc.ppr ["Z", "Z", "Z"](8) [[magic]], [[zeros]]#0, [[zeros]]#1 : !quantum.bit, !quantum.bit, !quantum.bit
    pbc.ref.ppr ["Z", "Z", "Z"](8) %m, %z#0, %z#1

    // CHECK: [[m0:%.+]], [[m0_out:%.+]] = pbc.ppm ["X"] [[ppr]]#0 : i1, !quantum.bit
    // CHECK: [[m1:%.+]], [[m1_out:%.+]]:2 = pbc.ppm ["Z", "X"] [[ppr]]#2, [[ppr]]#1 : i1, !quantum.bit, !quantum.bit
    %m0 = pbc.ref.ppm ["X"] %m : i1
    %m1 = pbc.ref.ppm ["Z", "X"] %z#1, %z#0 : i1

    // CHECK: quantum.dealloc_qb [[m0_out]] : !quantum.bit
    // CHECK: quantum.dealloc_qb [[m1_out]]#1 : !quantum.bit
    // CHECK: quantum.dealloc_qb [[m1_out]]#0 : !quantum.bit
    qref.dealloc_qb %m : !qref.bit
    qref.dealloc_qb %z#0 : !qref.bit
    qref.dealloc_qb %z#1 : !qref.bit

    // CHECK: return [[m0]], [[m1]] : i1, i1
    return %m0, %m1 : i1, i1
}

// -----

// CHECK-LABEL: test_PBC_control_flow
func.func @test_PBC_control_flow(%sw: i1, %n: index) -> i1 attributes {quantum.node} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[magic:%.+]] = pbc.fabricate magic : !quantum.bit
    %a = qref.alloc(2) : !qref.reg<2>
    %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
    %m = pbc.ref.fabricate magic : !qref.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[qreg]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[if:%.+]]:2 = scf.if %arg0 -> (!quantum.bit, !quantum.bit) {
    // CHECK:   [[ppr:%.+]]:2 = pbc.ppr ["Z", "Y"](8) [[magic]], [[q0]] : !quantum.bit, !quantum.bit
    // CHECK:   scf.yield [[ppr]]#0, [[ppr]]#1 : !quantum.bit, !quantum.bit
    // CHECK: } else {
    // CHECK:   scf.yield [[magic]], [[q0]] : !quantum.bit, !quantum.bit
    // CHECK: }
    // CHECK: [[insert:%.+]] = quantum.insert [[qreg]][ 0], [[if]]#1 : !quantum.reg, !quantum.bit
    scf.if %sw {
        pbc.ref.ppr ["Z", "Y"](8) %m, %q0
    }

    // CHECK: [[for:%.+]] = scf.for {{%.+}} = {{%.+}} to %arg1 step {{%.+}} iter_args([[iter:%.+]] = [[if]]#0) -> (!quantum.bit) {
    // CHECK:   [[aux:%.+]] = pbc.prepare plus : !quantum.bit
    // CHECK:   [[ppr:%.+]]:2 = pbc.ppr ["X", "X"](4) [[iter]], [[aux]] : !quantum.bit, !quantum.bit
    // CHECK:   {{%.+}}, [[sel_out:%.+]] = pbc.select.ppm (%arg0 ? ["Z"] : ["X"]) [[ppr]]#1 : i1, !quantum.bit
    // CHECK:   quantum.dealloc_qb [[sel_out]] : !quantum.bit
    // CHECK:   scf.yield [[ppr]]#0 : !quantum.bit
    // CHECK: }
    scf.for %i = %c0 to %n step %c1 {
        %aux = pbc.ref.prepare plus : !qref.bit
        pbc.ref.ppr ["X", "X"](4) %m, %aux
        %r = pbc.ref.select.ppm (%sw ? ["Z"] : ["X"]) %aux : i1
        qref.dealloc_qb %aux : !qref.bit
    }

    // CHECK: [[res:%.+]], [[res_out:%.+]] = pbc.ppm ["X"] [[for]] : i1, !quantum.bit
    %res = pbc.ref.ppm ["X"] %m : i1

    // CHECK: quantum.dealloc_qb [[res_out]] : !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc_qb %m : !qref.bit
    qref.dealloc %a : !qref.reg<2>

    // CHECK: return [[res]] : i1
    return %res : i1
}

// -----

// CHECK-LABEL: func.func private @PBC_subroutine
// CHECK-SAME: ([[q:%.+]]: !quantum.bit, [[r1:%.+]]: !quantum.bit) -> (!quantum.bit, !quantum.bit)
func.func private @PBC_subroutine(%r: !qref.reg<2>, %q: !qref.bit) {
    // CHECK: [[aux:%.+]] = pbc.fabricate magic_conj : !quantum.bit
    // CHECK: [[ppr:%.+]]:3 = pbc.ppr ["Z", "Z", "Y"](8) [[q]], [[r1]], [[aux]] : !quantum.bit, !quantum.bit, !quantum.bit
    // CHECK: {{%.+}}, [[aux_out:%.+]] = pbc.ppm ["X"] [[ppr]]#2 : i1, !quantum.bit
    // CHECK: quantum.dealloc_qb [[aux_out]] : !quantum.bit
    // CHECK: return [[ppr]]#0, [[ppr]]#1 : !quantum.bit, !quantum.bit
    %q1 = qref.get %r[1] : !qref.reg<2> -> !qref.bit
    %aux = pbc.ref.fabricate magic_conj : !qref.bit
    pbc.ref.ppr ["Z", "Z", "Y"](8) %q, %q1, %aux
    %mres = pbc.ref.ppm ["X"] %aux : i1
    qref.dealloc_qb %aux : !qref.bit
    return
}

// CHECK-LABEL: test_PBC_subroutine
func.func @test_PBC_subroutine() attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: [[magic:%.+]] = pbc.fabricate magic : !quantum.bit
    %a = qref.alloc(2) : !qref.reg<2>
    %m = pbc.ref.fabricate magic : !qref.bit

    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[call:%.+]]:2 = call @PBC_subroutine([[magic]], [[q1]]) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
    // CHECK: [[insert:%.+]] = quantum.insert [[qreg]][ 1], [[call]]#1 : !quantum.reg, !quantum.bit
    func.call @PBC_subroutine(%a, %m) : (!qref.reg<2>, !qref.bit) -> ()

    // CHECK: quantum.dealloc_qb [[call]]#0 : !quantum.bit
    // CHECK: quantum.dealloc [[insert]] : !quantum.reg
    qref.dealloc_qb %m : !qref.bit
    qref.dealloc %a : !qref.reg<2>
    return
}
