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

// Test conversion to value semantics quantum dialect for MBQC circuits.

// RUN: quantum-opt --convert-to-reference-semantics --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: test_mbqc_measure_in_basis_op
func.func @test_mbqc_measure_in_basis_op(%arg0: f64) -> (i1, i1, i1) attributes {quantum.node} {

    // CHECK: [[qubit:%.+]] = qref.alloc_qb : !qref.bit
    // CHECK-NEXT: [[mres:%.+]] = mbqc.ref.measure_in_basis[ XY, %arg0] [[qubit]] : i1
    // CHECK-NEXT: qref.dealloc_qb [[qubit]] : !qref.bit
    %1 = quantum.alloc_qb : !quantum.bit
    %mres, %out_qubit = mbqc.measure_in_basis[ XY, %arg0] %1 : i1, !quantum.bit
    quantum.dealloc_qb %out_qubit : !quantum.bit

    // CHECK: [[qreg:%.+]] = qref.alloc( 1) : !qref.reg<1>
    // CHECK-NEXT: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<1> -> !qref.bit
    // CHECK-NEXT: [[mres_0:%.+]] = mbqc.ref.measure_in_basis[ YZ, %arg0] [[q0]] : i1
    // CHECK-NEXT: [[mres_2:%.+]] = mbqc.ref.measure_in_basis[ ZX, %arg0] [[q0]] : i1
    // CHECK-NEXT: qref.dealloc [[qreg]] : !qref.reg<1>
    %0 = quantum.alloc( 1) : !quantum.reg
    %2 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %mres_0, %out_qubit_1 = mbqc.measure_in_basis[ YZ, %arg0] %2 : i1, !quantum.bit
    %mres_2, %out_qubit_3 = mbqc.measure_in_basis[ ZX, %arg0] %out_qubit_1 : i1, !quantum.bit
    %3 = quantum.insert %0[ 0], %out_qubit_3 : !quantum.reg, !quantum.bit
    quantum.dealloc %3 : !quantum.reg

    // CHECK: return [[mres]], [[mres_0]], [[mres_2]] : i1, i1, i1
    return %mres, %mres_0, %mres_2 : i1, i1, i1
}


// -----


// CHECK-LABEL: test_mbqc_graph_state_prep
func.func @test_mbqc_graph_state_prep(%arg0: tensor<6xi1>, %arg1: f64) -> i1 attributes {quantum.node} {
    // CHECK: [[qreg:%.+]] = mbqc.ref.graph_state_prep(%arg0 : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !qref.reg<4>
    %0 = mbqc.graph_state_prep(%arg0 : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg

    // CHECK: [[q0:%.+]] = qref.get [[qreg]][ 0] : !qref.reg<4> -> !qref.bit
    // CHECK-NEXT: [[mres:%.+]] = mbqc.ref.measure_in_basis[ YZ, %arg1] [[q0]] : i1
    // CHECK-NOT: quantum.insert
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %mres, %out_qubit_1 = mbqc.measure_in_basis[ YZ, %arg1] %1 : i1, !quantum.bit
    %2 = quantum.insert %0[ 0], %out_qubit_1 : !quantum.reg, !quantum.bit

    // CHECK-NEXT: qref.dealloc [[qreg]] : !qref.reg<4>
    quantum.dealloc %2 : !quantum.reg

    // CHECK: return [[mres]] : i1
    return %mres : i1
}
