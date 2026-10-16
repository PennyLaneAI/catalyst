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

// RUN: catalyst --tool=opt --pass-pipeline='builtin.module(graph-decomposition{gate-set=Toffoli=1.0,CNOT=1.0,PauliX=1.0,Hadamard=1.0,S=1.0,T=1.0,CZ=1.0})' %s | FileCheck %s

// A decomposition rule receives all base wires of the operator as one grouped operand, packed in
// the order of the operator's qubit operands (its declared wire arguments). SemiAdder declares
// x_wires, y_wires, work_wires, which is not alphabetical, so a rule that unpacks the group in
// sorted-name order would read the registers as work = [0:1], x = [1:2], y = [2:6] and act on the
// wrong qubits. The register sizes are all different so that only the declared order produces
// these three slices.

// CHECK-LABEL: func.func @circuit
// CHECK-NOT: quantum.operator "SemiAdder"
// CHECK: func.func private @"__builtin__semi_adder_SemiAdder{}{work_wires:2,x_wires:1,y_wires:3}{}
// CHECK-SAME: (%arg0: tensor<6xi64>
// CHECK-DAG: stablehlo.slice %arg0 [0:1] : (tensor<6xi64>) -> tensor<1xi64>
// CHECK-DAG: stablehlo.slice %arg0 [1:4] : (tensor<6xi64>) -> tensor<3xi64>
// CHECK-DAG: stablehlo.slice %arg0 [4:6] : (tensor<6xi64>) -> tensor<2xi64>
func.func @circuit() attributes {quantum.node} {
    %r = quantum.alloc(6) : !quantum.reg
    %q0 = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r[1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %r[2] : !quantum.reg -> !quantum.bit
    %q3 = quantum.extract %r[3] : !quantum.reg -> !quantum.bit
    %q4 = quantum.extract %r[4] : !quantum.reg -> !quantum.bit
    %q5 = quantum.extract %r[5] : !quantum.reg -> !quantum.bit
    %out:6 = quantum.operator "SemiAdder"() qubits(%q0, %q1, %q2, %q3, %q4, %q5)
      qubit_map = {x_wires = [0], y_wires = [1, 2, 3], work_wires = [4, 5]}
    %r0 = quantum.insert %r[0], %out#0 : !quantum.reg, !quantum.bit
    %r1 = quantum.insert %r0[1], %out#1 : !quantum.reg, !quantum.bit
    %r2 = quantum.insert %r1[2], %out#2 : !quantum.reg, !quantum.bit
    %r3 = quantum.insert %r2[3], %out#3 : !quantum.reg, !quantum.bit
    %r4 = quantum.insert %r3[4], %out#4 : !quantum.reg, !quantum.bit
    %r5 = quantum.insert %r4[5], %out#5 : !quantum.reg, !quantum.bit
    quantum.dealloc %r5 : !quantum.reg
    return
}
