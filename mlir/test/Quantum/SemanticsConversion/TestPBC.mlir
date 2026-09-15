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

// Test conversion to reference semantics quantum dialect for PBC circuits.

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

// -----

// CHECK-LABEL: test_PPR_op
func.func @test_PPR_op(%sw: i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 3) : !qref.reg<3>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<3> -> !qref.bit
    // CHECK: [[q2:%.+]] = qref.get [[qreg]][ 2] : !qref.reg<3> -> !qref.bit
    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    // The reference PPR acts in place, so the same qubit references thread through
    // subsequent ops rather than being re-plumbed via new SSA values.
    // CHECK: pbc.ref.ppr ["X", "I", "Z"](4) [[q0]], [[q1]], [[q2]]
    // CHECK: pbc.ref.ppr ["Z", "Y", "X"](-2) [[q0]], [[q1]], [[q2]] cond(%arg0)
    %o:3 = pbc.ppr ["X", "I", "Z"](4) %1, %2, %3 : !quantum.bit, !quantum.bit, !quantum.bit
    %p:3 = pbc.ppr ["Z", "Y", "X"](-2) %o#0, %o#1, %o#2 cond(%sw) : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.insert
    %4 = quantum.insert %0[ 0], %p#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %p#1 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 2], %p#2 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<3>
    quantum.dealloc %6 : !quantum.reg
    return
}

// -----

// CHECK-LABEL: test_select_PPM_op
func.func @test_select_PPM_op(%sw: i1) -> (i1, i1) attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<2> -> !qref.bit
    // CHECK: [[q1:%.+]] = qref.get [[qreg]][ 1] : !qref.reg<2> -> !qref.bit
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit

    // CHECK: [[mres:%.+]] = pbc.ref.select.ppm (%arg0 ? ["Z", "Y"] : ["X", "Z"]) [[q0]], [[q1]] : i1
    // CHECK: [[mres_0:%.+]] = pbc.ref.select.ppm ([[mres]] ? ["X", "Y"] : ["Z", "X"]) [[q0]], [[q1]] : i1
    %mres, %oq:2 = pbc.select.ppm (%sw ? ["Z", "Y"] : ["X", "Z"]) %1, %2 : i1, !quantum.bit, !quantum.bit
    %mres_0, %oq2:2 = pbc.select.ppm (%mres ? ["X", "Y"] : ["Z", "X"]) %oq#0, %oq#1 : i1, !quantum.bit, !quantum.bit

    // CHECK-NOT: quantum.insert
    %3 = quantum.insert %0[ 0], %oq2#0 : !quantum.reg, !quantum.bit
    %4 = quantum.insert %3[ 1], %oq2#1 : !quantum.reg, !quantum.bit

    // CHECK: qref.dealloc [[qreg]] : !qref.reg<2>
    quantum.dealloc %4 : !quantum.reg

    // CHECK: return [[mres]], [[mres_0]] : i1, i1
    return %mres, %mres_0 : i1, i1
}

// -----

// CHECK-LABEL: test_fabricate_op
func.func @test_fabricate_op() -> (i1, i1) attributes {quantum.node} {
    // CHECK: [[q:%.+]] = pbc.ref.fabricate magic : !qref.bit
    %q = pbc.fabricate magic : !quantum.bit

    // CHECK: [[mres:%.+]] = pbc.ref.ppm ["Z"] [[q]] : i1
    // CHECK: [[mres_0:%.+]] = pbc.ref.ppm ["X"] [[q]] : i1
    %mres, %out_qubit = pbc.ppm ["Z"] %q : i1, !quantum.bit
    %mres_0, %out_qubit_0 = pbc.ppm ["X"] %out_qubit : i1, !quantum.bit

    // CHECK: qref.dealloc_qb [[q]] : !qref.bit
    quantum.dealloc_qb %out_qubit_0 : !quantum.bit

    // CHECK: return [[mres]], [[mres_0]] : i1, i1
    return %mres, %mres_0 : i1, i1
}

// -----

// CHECK-LABEL: test_prepare_op
func.func @test_prepare_op() -> (i1, i1) attributes {quantum.node} {
    // CHECK: [[q:%.+]] = pbc.ref.prepare zero : !qref.bit
    %q = pbc.prepare zero : !quantum.bit

    // CHECK: [[mres:%.+]] = pbc.ref.ppm ["X"] [[q]] : i1
    // CHECK: [[mres_0:%.+]] = pbc.ref.ppm ["Z"] [[q]] : i1
    %mres, %out_qubit = pbc.ppm ["X"] %q : i1, !quantum.bit
    %mres_0, %out_qubit_0 = pbc.ppm ["Z"] %out_qubit : i1, !quantum.bit

    // CHECK: qref.dealloc_qb [[q]] : !qref.bit
    quantum.dealloc_qb %out_qubit_0 : !quantum.bit

    // CHECK: return [[mres]], [[mres_0]] : i1, i1
    return %mres, %mres_0 : i1, i1
}
