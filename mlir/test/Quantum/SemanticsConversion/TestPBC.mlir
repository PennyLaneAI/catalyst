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

// RUN: quantum-opt --convert-to-reference-semantics --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_PPM_op
func.func @test_PPM_op(%angle: f64) -> (i1, i1, i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q2:%.+]] = qref.get [[qreg]][ 2] : !qref.reg<3> -> !qref.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    // CHECK: [[mres:%.+]] = pbc.ref.ppm ["Z"] [[q0]] : i1
    // CHECK: [[mres_0:%.+]] = pbc.ref.ppm ["Z", "Y"] [[q0]], [[q1]] : i1
    // CHECK: [[mres_2:%.+]] = pbc.ref.ppm ["Z", "Y", "X"] [[q0]], [[q1]], [[q2]] : i1
    %mres, %out_qubits = pbc.ppm ["Z"] %1 : i1, !quantum.bit
    %mres_0, %out_qubits_1:2 = pbc.ppm ["Z", "Y"] %out_qubits, %2 : i1, !quantum.bit, !quantum.bit
    %mres_2, %out_qubits_3:3 = pbc.ppm ["Z", "Y", "X"] %out_qubits_1#0, %out_qubits_1#1, %3 : i1, !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.insert
    %4 = quantum.insert %0[ 0], %out_qubits_3#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %out_qubits_3#1 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 2], %out_qubits_3#2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc %6 : !quantum.reg

    // CHECK: return [[mres]], [[mres_0]], [[mres_2]] : i1, i1, i1
    return %mres, %mres_0, %mres_2 : i1, i1, i1
}
