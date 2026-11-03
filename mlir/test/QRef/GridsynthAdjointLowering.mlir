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

// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.01}, convert-to-reference-semantics, adjoint-lowering)" %s | FileCheck %s

// Rotations discretized inside an adjoint region are inverted by the adjoint lowering: the gate
// sequence is computed for the original angle and applied in reverse with adjoint gates.

// CHECK-LABEL: func.func private @__catalyst_decompose_RZ.adjoint(
// CHECK-SAME:  [[Q_IN:%.+]]: !qref.bit, [[THETA:%.+]]: f64)
// CHECK:       [[SIZE:%.+]] = call @rs_decomposition_get_size([[THETA]]
// CHECK:       call @rs_decomposition_get_gates({{.*}}, [[THETA]]
// CHECK:       [[PHASE:%.+]] = call @rs_decomposition_get_phase([[THETA]]
// CHECK:       qref.gphase([[PHASE]]) adj
// CHECK: scf.for
// CHECK:         scf.index_switch
// CHECK:         case 0 {
// CHECK:           qref.custom "T"() [[Q_IN]] adj
// CHECK:         case 2 {
// CHECK:           qref.custom "T"() [[Q_IN]] adj
// CHECK:           qref.custom "Hadamard"() [[Q_IN]] adj
// CHECK:           qref.custom "S"() [[Q_IN]] adj
// CHECK: return

// CHECK-LABEL: func.func @test_adjoint_region
// CHECK-SAME:  ([[R_IN:%.+]]: !qref.reg<?>, [[X:%.+]]: f64)
// CHECK-NOT:   qref.adjoint
// CHECK:       [[Q:%.+]] = qref.get [[R_IN]][ 0]
// CHECK:       call @__catalyst_decompose_RZ.adjoint([[Q]], [[X]])
func.func @test_adjoint_region(%r: !quantum.reg, %x: f64) -> !quantum.reg {
    %out = quantum.adjoint(%r) : !quantum.reg {
    ^bb0(%arg0: !quantum.reg):
        %q = quantum.extract %arg0[0] : !quantum.reg -> !quantum.bit
        %q1 = quantum.custom "RZ"(%x) %q : !quantum.bit
        %r1 = quantum.insert %arg0[0], %q1 : !quantum.reg, !quantum.bit
        quantum.yield %r1 : !quantum.reg
    }
    return %out : !quantum.reg
}
