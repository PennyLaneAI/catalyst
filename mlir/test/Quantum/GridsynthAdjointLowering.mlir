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

// RUN: quantum-opt --pass-pipeline="builtin.module(gridsynth{epsilon=0.01}, adjoint-lowering)" %s | FileCheck %s

// Rotations discretized inside an adjoint region are inverted by the adjoint lowering: the gate
// sequence is computed for the original angle and applied in reverse with adjoint gates.

// CHECK-LABEL: func.func private @__catalyst_decompose_RZ.adjoint(
// CHECK-SAME:  [[Q_IN:%.+]]: !quantum.bit, [[THETA:%.+]]: f64) -> !quantum.bit
// CHECK:       [[SIZE:%.+]] = call @rs_decomposition_get_size([[THETA]]
// CHECK:       call @rs_decomposition_get_gates({{.*}}, [[THETA]]
// CHECK:       [[PHASE:%.+]] = call @rs_decomposition_get_phase([[THETA]]
// CHECK:       quantum.gphase([[PHASE]]) adj
// CHECK:       [[RES:%.+]] = scf.for {{.*}} iter_args([[L_QBIT:%.+]] = [[Q_IN]])
// CHECK:         scf.index_switch
// CHECK:         case 0 {
// CHECK:           quantum.custom "T"() [[L_QBIT]] adj
// CHECK:         case 2 {
// CHECK:           [[G1:%.+]] = quantum.custom "T"() [[L_QBIT]] adj
// CHECK:           [[G2:%.+]] = quantum.custom "Hadamard"() [[G1]] adj
// CHECK:           quantum.custom "S"() [[G2]] adj
// CHECK:       return [[RES]] : !quantum.bit

// CHECK-LABEL: func.func @test_adjoint_region
// CHECK-SAME:  ([[R_IN:%.+]]: !quantum.reg, [[X:%.+]]: f64)
// CHECK-NOT:   quantum.adjoint
// CHECK:       [[Q:%.+]] = quantum.extract [[R_IN]][ 0]
// CHECK:       [[Q_OUT:%.+]] = call @__catalyst_decompose_RZ.adjoint([[Q]], [[X]])
// CHECK:       quantum.insert [[R_IN]][ 0], [[Q_OUT]]
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
