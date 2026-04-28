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
    %m0 = qref.pbc.ppm ["Z"] %q0 : i1

    // CHECK: [[q1:%.+]] = quantum.extract [[qreg]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: [[m1:%.+]], [[m1_out_qubits:%.+]]:2 = pbc.ppm ["Z", "Y"] [[m0_out_qubit]], [[q1]] : i1, !quantum.bit, !quantum.bit
    %m1 = qref.pbc.ppm ["Z", "Y"] %q0, %q1 : i1
    // CHECK: [[insert0:%.+]] = quantum.insert [[qreg]][ 0], [[m1_out_qubits]]#0 : !quantum.reg, !quantum.bit
    // CHECK: [[insert1:%.+]] = quantum.insert [[insert0]][ 1], [[m1_out_qubits]]#1 : !quantum.reg, !quantum.bit

    // CHECK: [[q0:%.+]] = quantum.extract [[insert1]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: [[m2:%.+]], [[m2_out_qubits:%.+]]:2 = pbc.ppm ["X", "Z"] [[q0]], [[qb]] : i1, !quantum.bit, !quantum.bit
    %m2 = qref.pbc.ppm ["X", "Z"] %q0, %qb : i1
    // CHECK: [[insert2:%.+]] = quantum.insert [[insert1]][ 0], [[m2_out_qubits]]#0 : !quantum.reg, !quantum.bit

    // CHECK: quantum.dealloc [[insert2]] : !quantum.reg
    // CHECK: quantum.dealloc_qb [[m2_out_qubits]]#1 : !quantum.bit
    qref.dealloc %a : !qref.reg<3>
    qref.dealloc_qb %qb : !qref.bit

    // CHECK: return [[m0]], [[m1]], [[m2]] : i1, i1, i1
    return %m0, %m1, %m2 : i1, i1, i1
}
